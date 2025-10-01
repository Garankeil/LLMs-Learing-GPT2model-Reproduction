import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

from torch.cuda.amp import autocast


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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MHA = MutiHeadAttention(config)
        self.FFN = FeedForward(config)
        self.layernorm1 = nn.LayerNorm(config.dims)
        self.layernorm2 = nn.LayerNorm(config.dims)

    def forward(self, x):
        x = x + self.MHA(self.layernorm1(x))
        x = x + self.FFN(self.layernorm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.dims)
        self.position_embedding_table = nn.Embedding(config.max_seq_len, config.dims)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_block)]
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

    def forward(self, token_idx, target_idx=None):
        # 两个输入的Shape要一样
        batch, seq_len = token_idx.size()  # (batch, seq_len)
        token_emb = self.token_embedding_table(token_idx)  # (batch, seq_len, dims) words的向量映射
        posi_emb = self.position_embedding_table(
            torch.arange(seq_len, device=token_idx.device)
        )
        # 经典题目：为何token_emb和posi_emb可以相加
        x = token_emb + posi_emb  # (batch, seq_len, dims)
        x = self.blocks(x)
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
            loss = F.cross_entropy(logits,
                                   target_idx)  # logits(all_seq_len, vocab_size), targets(all_seq_len) but its value is vocab number

        return logits, loss

    def generate(self, idx):
        i = 0
        for i in range(self.max_seq_len):
            idx_current = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # print(idx_current)
            logits, _ = self.forward(idx_current, None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            # print(idx_next)
            idx = torch.cat([idx_current, idx_next], dim=1)
            if idx_next == 50256:
                return idx[:, -i:-1]
        print('超过最大长度，生成结束')
        return idx[:, 1:]

