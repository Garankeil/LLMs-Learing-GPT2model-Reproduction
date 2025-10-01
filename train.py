import hashlib
from itertools import islice
import torch
import tiktoken
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
from model.Transformers import GPT, GPTconfig
import json
import os
import argparse
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


class Mydataset(Dataset):
    def __init__(self, path, max_seq_len: int = 512):
        #  读取1000行
        self.max_line = 2e5
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_seq_len = max_seq_len
        self.encoded_data = []
        # 特殊符号分割训练文本
        # <|endoftext|>
        self.eot_token = self.enc.encode(
            '<|endoftext|>',
            allowed_special={'<|endoftext|>'}
        )[0]

        # 读取json数据，按line读取，取出‘text’数据存入raw_data
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_line:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except Exception as e:
                    continue

        # 讲数据全部变成一行，方便分割
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eot_token])

        # max_seq_len 512 切割文本为512长度
        for i in range(0, len(full_encoded), self.max_seq_len):
            chunk = full_encoded[i:i + self.max_seq_len + 1]  # shift操作，但不知为何， 每一行实际长度是513
            if len(chunk) < self.max_seq_len + 1:
                chunk = chunk + [self.eot_token] * (self.max_seq_len + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        # x是文本前512个word， y是后512个， 共513个， 作loss以让模型学习下一个文本是什么
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token ID"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)


class StreamingGPT2Dataset(IterableDataset):
    def __init__(self, path, split='train', split_ratio=0.9, max_seq_len=512, max_lines=5e6):
        self.path = path
        self.max_seq_len = max_seq_len
        self.max_lines = int(max_lines)
        self.split = split  # 'train' 或 'val'
        self.split_ratio = split_ratio
        self.enc = tiktoken.get_encoding('gpt2')
        self.eot_token = self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

        # 预计算总样本数（可选，非必须）
        self._total_chunks = self._estimate_total_chunks()

    def _estimate_total_chunks(self):
        """估算总chunk数（遍历一次文件）"""
        chunk_count = 0
        with open(self.path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(islice(f, self.max_lines)):
                try:
                    text = json.loads(line.strip())['text']
                    encoded_len = len(self.enc.encode(text)) + 1  # +1 for EOT
                    chunk_count += (encoded_len // self.max_seq_len) + 1
                except:
                    continue
        return chunk_count

    def __iter__(self):
        buffer = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in islice(f, self.max_lines):
                # 通过哈希决定数据归属（确定性分流）
                line_hash = hashlib.md5(line.encode()).hexdigest()
                hash_ratio = int(line_hash, 16) / (1 << 128)  # 归一化到[0, 1)

                if (self.split == 'train' and hash_ratio < self.split_ratio) or \
                        (self.split == 'val' and hash_ratio >= self.split_ratio):
                    try:
                        text = json.loads(line.strip())['text']
                        encoded = self.enc.encode(text) + [self.eot_token]
                        buffer.extend(encoded)

                        # 动态分块
                        while len(buffer) >= self.max_seq_len + 1:
                            chunk = buffer[:self.max_seq_len + 1]
                            x = torch.tensor(chunk[:-1], dtype=torch.long)
                            y = torch.tensor(chunk[1:], dtype=torch.long)
                            yield x, y
                            buffer = buffer[self.max_seq_len:]  # 滑动窗口

                    except json.JSONDecodeError:
                        continue

        # 处理剩余不足一个chunk的数据
        if buffer:
            chunk = buffer + [self.eot_token] * (self.max_seq_len + 1 - len(buffer))
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y

    def __len__(self):
        return self._total_chunks  # 或返回估算值


class LineChunkDataset(Dataset):
    def __init__(self, chunks_dir, max_seq_len=512):
        """
        Args:
            chunks_dir: 存储.npy文件的目录
            max_seq_len: 每个chunk的序列长度
        """
        self.chunk_files = sorted(
            [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])  # 按文件名数字排序
        )
        self.max_seq_len = max_seq_len
        self.cumulative_chunks = self._precompute_offsets()  # 预计算每个文件的chunk偏移量

    def _precompute_offsets(self):
        """预计算每个文件的chunk起始索引"""
        cumulative_chunks = []
        total = 0
        for file in self.chunk_files:
            data = np.load(file, mmap_mode='r')
            cumulative_chunks.append((file, total, total + len(data)))
            total += len(data)
        return cumulative_chunks

    def __len__(self):
        return self.cumulative_chunks[-1][2] if self.cumulative_chunks else 0

    def __getitem__(self, idx):
        """根据全局idx定位到具体文件和chunk"""
        for file, start, end in self.cumulative_chunks:
            if start <= idx < end:
                data = np.load(file, mmap_mode='r')
                chunk = data[idx - start].copy()
                x = torch.from_numpy(chunk[:-1]).long()
                y = torch.from_numpy(chunk[1:]).long()
                return x, y
        raise IndexError("Index out of range")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT2')

    # 数据相关参数
    parser.add_argument('--dataset-path', type=str, default='dataset/dataset_chunk', help='数据集路径或目录')
    parser.add_argument('--max-seq-len', type=int, default=512, help='最大序列长度')

    # 训练相关参数
    parser.add_argument('--train-batch-size', type=int, default=12, help='训练batch大小')
    parser.add_argument('--val-batch-size', type=int, default=8, help='验证batch大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--num-workers', type=int, default=16, help='数据加载线程数')

    # 模型保存相关
    parser.add_argument('--save-dir', type=str, default='model_save', help='模型保存目录')
    parser.add_argument('--save-interval', type=int, default=1, help='每隔多少epoch保存一次模型')

    # 其他参数
    parser.add_argument('--resume-path', type=str, default=None, help='检查点路径')

    return parser.parse_args()


def sec_to_hms(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    m, s = divmod(seconds, 60)  # 分钟和秒
    h, m = divmod(m, 60)        # 小时和分钟
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def train(model_t, optimizer_t, scheduler_t, train_loader_t, rank_t, writer_t, scaler):
    model_t.train()
    log_step = 0
    print_time = time.time()
    batch_len = len(train_loader_t)
    for batch_idx, (x, y) in enumerate(train_loader_t):
        x, y = x.to(model_t.device), y.to(model_t.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):  # amp
            logits, loss = model_t(token_idx=x, target_idx=y)  # rank0广播model buffer给其他GPU
        # 反向传播
        optimizer_t.zero_grad()
        # loss.backward()  # 每个rank的每个参数梯度求平均（all reduce）进程是实时进行的， 并不是训练完一个batch在统一
        # optimizer_t.step()

        # amp (梯度缩放)
        scaler.scale(loss).backward()
        scaler.step(optimizer_t)
        scaler.update()

        # lr
        scheduler_t.step()
        if batch_idx % 100 == 0 and rank_t == 0:
            writer_t.add_scalar('Batch Loss', loss.item(), log_step)
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, Time: {sec_to_hms(time.time() - print_time)},'
                  f' Remaintime: {sec_to_hms((time.time() - print_time) * (batch_len - batch_idx + 1) // 100)}')
            print_time = time.time()
            log_step += 1
    dist.reduce(loss, dst=0)  # rank0汇总其他GPU的的loss
    return loss


def val(model_e, val_loader_e, rank_e):
    model_e.eval()
    eval_loss = 0
    with torch.no_grad():
        for x, y in val_loader_e:
            x, y = x.to(f'cuda:{rank_e}'), y.to(f'cuda:{rank_e}')
            logits, loss = model_e(token_idx=x, target_idx=y)
            eval_loss += loss.item()
    avg_loss = eval_loss / len(val_loader_e)
    return avg_loss


# dataset
# train_dataset = Mydataset('/home/jnu/jiananfu/project/GPT2/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl')
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])  # 将训练集分 10% 作为验证集
#
# train_loader = DataLoader(train_dataset, batch_size=24, num_workers=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=False)


def main():
    torch.cuda.empty_cache()

    args = parse_args()

    writer = SummaryWriter('logs_train')

    dist.init_process_group(backend='nccl')  # 集合通讯协议，其余GPU连接Master(互认)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    checkpoint = None
    if args.resume_path is not None:
        print(f'from {args.resume_path} recovering...')
        try:
            checkpoint = torch.load(args.resume_path, map_location={'cuda:0': 'cpu'})
        except Exception as e:
            if rank == 0:
                print(f'Error loading checkpoint: {e}')

    model = GPT(GPTconfig()).to(f'cuda:{rank}')
    # parameters calculate
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total params: {total_params / 1048576:.2f} M')
        print(f'Using train params：{args}')

    if checkpoint and rank == 0:  # rank0恢复参数
        model.load_state_dict(checkpoint['model'])

    model = DDP(model)  # 封装为DDP，rank0广播参数给其他GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint and rank == 0:
        optimizer.load_state_dict(checkpoint['optimizer'])  # 各GPU恢复优化器参数

    # chunkdataset
    chunkdataset = LineChunkDataset(args.dataset_path, max_seq_len=args.max_seq_len)
    train_dataset_chunk, val_dataset_chunk = torch.utils.data.random_split(chunkdataset, [0.9, 0.1])  # 将训练集分 10% 作为验证集

    sampler = DistributedSampler(train_dataset_chunk)  # 指派子集给各个GPU
    train_dataloader_chunk = DataLoader(train_dataset_chunk, batch_size=args.train_batch_size, num_workers=args.num_workers,
                                        sampler=sampler, persistent_workers=True)

    val_dataloader_chunk = DataLoader(val_dataset_chunk, batch_size=args.val_batch_size, num_workers=2,
                                      shuffle=True, persistent_workers=True)
    if rank == 0:
        print(f'rank{rank} train dataset size: {len(train_dataset_chunk)}')
        print(f'val dataset size: {len(val_dataset_chunk)}')
        print(f'rank{rank} train dataloader size: {len(train_dataloader_chunk)}')
        print(f'val dataloader size: {len(val_dataloader_chunk)}')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset_chunk) // args.train_batch_size)

    scaler = torch.cuda.amp.GradScaler(enabled=True)  # amp的梯度缩放器

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 生成随机种子， rank0广播给其他GPU
        if rank == 0:
            print('<----------Train Process---------->')
        train_loss = train(model, optimizer, scheduler, train_dataloader_chunk, rank, writer, scaler)
        if rank == 0:
            avg_train_loss = train_loss / world_size

            # evaluate
            print('<----------Val Process---------->')
            raw_model = model.module
            avg_val_loss = val(raw_model, val_dataloader_chunk, rank)
            print(f'Epoch: {epoch}, Train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}')
            writer.add_scalars('avg_Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch + 1)

            # save
            if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
                torch.save(checkpoint, save_path)
        dist.barrier()  # 等待rank0完成evaluate
    if rank == 0:
        print('Training Done.')


# torchrun --nproc_per_node=2 train.py
if __name__ == '__main__':
    main()
