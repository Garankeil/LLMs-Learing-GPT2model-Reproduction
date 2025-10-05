import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class GPTconfig:
    block_size: int = 512  # 序列最大长度
    max_seq_len = block_size
    batch_size: int = 12
    n_block: int = 12
    n_heads: int = 12
    dims: int = 768
    dropout: float = 0.1
    head_dim: int = dims // n_heads
    n_kv_heads: int = 4
    # vocab_size
    # GPT2官方tokenizer
    vocab_size = 50257
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eos_token = 50256


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.dims, config.head_dim)
        self.key = nn.Linear(config.dims, config.head_dim)
        self.value = nn.Linear(config.dims, config.head_dim)
        # self.qkv_proj = nn.Linear(config.dims, 3 * config.dims)
        self.head_dim = config.head_dim

        # attention mask 通过 register_bufffer注册
        # 不用计算梯度，运行更快，内存更省

        # self.register_buffer(
        #     'attention_mask',
        #     torch.tril(
        #         torch.ones(config.max_seq_len, config.max_seq_len)
        #     )
        # )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, head_dim = x.size()
        # qkv = self.qkv_proj(x)  # [B, T, 3*head_dim]
        # q, k, v = qkv.chunk(3, dim=-1)  # 拆分Q/K/V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        weight = torch.matmul(
            q, k.transpose(-1, -2)
        )  # Q 乘以 T的转置
        weight = weight.masked_fill(
            mask[:seq_len, :seq_len] == 0,
            float('-inf')
        )  # 加入mask（decoder）
        weight = weight / math.sqrt(self.head_dim)  # 除以根号dk
        weight = torch.softmax(weight, dim=-1)
        # dropout
        weight = self.dropout(weight)
        output = torch.matmul(
            weight, v
        )
        return output


class MutiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads_attention = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_heads)  # 以索引形式调用模块
            ]
        )
        self.WO_proj = nn.Linear(config.dims, config.dims)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads_attention],  # 妙哉， 列表表达式的用法
            dim=-1
        )
        output = self.WO_proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(config.dims, config.dims * 4),
            nn.GELU(),
            nn.Linear(config.dims * 4, config.dims),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.forward_net(x)


class GroupQueryAttention(nn.Module):  # MutiQueryAttention是GQA的n_kv_heads = 1 时的特殊形态， 损失效果更多， 而GQA是MHA和MQA的折中选择
    def __init__(self, config):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0
        assert config.dims % config.n_heads == 0
        self.query_proj = nn.Linear(config.dims, config.n_heads * config.head_dim)
        self.key_proj = nn.Linear(config.dims, config.n_kv_heads * config.head_dim)
        self.value_proj = nn.Linear(config.dims, config.n_kv_heads * config.head_dim)
        self.output_proj = nn.Linear(config.dims, config.dims)
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.dropout = nn.Dropout(config.dropout)
        self.head_dim = config.head_dim

    def forward(self, x, past_key_value=None, use_cache=False):
        batch, seq_len, _ = x.size()  # x (batch, seq_len, dims)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        # 所谓kv cache技术是在推理时应用，推理时第一次计算x序列的全部长度qkv，并缓存这个k和v
        # 第一次之后后仅计算最后一个token的qkv，然后将之前的kv和现在的拼接
        # 即每次仅需计算最后一个token，而之前的kv已经保存，可直接用于计算
        # 为何不缓存之前的q？因为历史token的q并不会被访问，从query的字面意思就能理解
        # kv cache在x的长度上体现，而不再attn的计算过程中体现

        # attention weight target shape(batch, n_heads, seq_len, seq_len)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dims)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dims)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dims)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k.detach(), v.detach()) if use_cache else None

        # n_kv_heads 是 n_heads 的分组 需要Repeat
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        total_len = k.size(2)
        mask = torch.tril(torch.ones(total_len, total_len, device=x.device))
        mask = mask[-seq_len:, :total_len]

        # attention score
        attention_score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention_score = attention_score.masked_fill(mask == 0,float('-inf'))  # 加入mask（decoder）
        attention_prob = torch.softmax(attention_score, dim=-1)
        # dropout
        attention_prob = self.dropout(attention_prob)

        output = attention_prob @ v  # attention_score(batch, n_heads, seq_len, seq_len) * v(batch, n_heads, seq_len, head_dims)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)  # (batch, seq_len, dims)
        output = self.output_proj(output)
        output = self.dropout(output)

        return output, present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.MHA = MutiHeadAttention(config)
        self.GQA = GroupQueryAttention(config)
        self.FFN = FeedForward(config)
        self.layernorm1 = nn.LayerNorm(config.dims)
        self.layernorm2 = nn.LayerNorm(config.dims)

    def forward(self, x, past_key_value=None, use_cache=False):
        attn_output, present = self.GQA(self.layernorm1(x), past_key_value=past_key_value, use_cache=use_cache)  # 输入是上一个token的kv, 输出现在的kv
        x = x + attn_output
        x = x + self.FFN(self.layernorm2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.dims)
        self.position_embedding_table = nn.Embedding(config.max_seq_len, config.dims)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_block)]
            # *号将列表解包，作为独立参数传进sequential，ModuleList接收list，而Sequential接收独立参数
        )
        self.layernorm_final = nn.LayerNorm(config.dims)
        self.FFN_final = nn.Linear(config.dims, config.vocab_size, bias=False)
        self.token_embedding_table.weight = self.FFN_final.weight  # tie_weight技术，因为Linear层（4->8）的权重其实是（8*4），所以可以合并权重，合并权重减少训练量

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # 偏置初始化为0
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, token_idx, target_idx=None, past_key_value=None, use_cache=False):
        # 两个输入的Shape要一样
        batch, seq_len = token_idx.size()  # (batch, seq_len)
        token_emb = self.token_embedding_table(token_idx)  # (batch, seq_len, dims) words的向量映射
        posi_emb = self.position_embedding_table(
            torch.arange(seq_len, device=token_idx.device)
        )
        # 经典题目：为何token_emb和posi_emb可以相加
        x = token_emb + posi_emb  # (batch, seq_len, dims)

        # x = self.blocks(x)
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_key_value[i] if past_key_value is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)  # Past -> Present
            if use_cache:
                presents.append(present)

        x = self.layernorm_final(x)
        logits = self.FFN_final(x)  # (batch, seq_len, vocab_size)

        if target_idx is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            target_idx = target_idx.view(batch * seq_len)
            # print("Logits shape:", logits.shape)  # 应为 [batch_size, num_classes, ...]
            # print("Target shape:", target_idx.shape)  # 应为 [batch_size, ...]
            loss = F.cross_entropy(logits, target_idx)  # logits(all_seq_len, vocab_size), targets(all_seq_len) but its value is vocab number

        return logits, loss, tuple(presents) if use_cache else None

    def generate(self, idx, max_new_tokens: int = 512):
        self.eval()
        past = None
        # 仅用于上下文，不再复制
        idx = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

        generated_tokens = []  # 专门存储新生成的token

        for i in range(max_new_tokens):
            # 首次：传入整个序列；之后只传入最后一个token
            if past is None:
                idx_current = idx
            else:
                idx_current = next_token  # 只传入上一个新token

            logits, _, past = self.forward(idx_current, None, past_key_value=past, use_cache=True)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated_tokens.append(next_token)

            # 如果遇到结束标志，则提前停止
            if next_token.item() == 50256:
                break

        # 拼接所有生成token并返回
        if len(generated_tokens) > 0:
            return torch.cat(generated_tokens, dim=1)
        else:
            return torch.empty((idx.size(0), 0), dtype=torch.long, device=idx.device)

# xx = torch.ones(12,12).to(torch.long)  # (batch, seq_len, dims)
# yy = torch.zeros(12,12).to(torch.long)
# net = GPT(GPTconfig)
# logitss, losss, _ = net(xx, yy)
# print(logitss, losss)

model = GPT(GPTconfig()).to('cpu')
input_ids = torch.tensor([[10, 20, 30]], device='cpu')
out = model.generate(input_ids)
print(out.shape)
