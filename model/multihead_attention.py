# encoding: utf-8
"""
@project = GPS$
@create_time = 2025-02-20$ 21:38$
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # query: (batch, seq_len_q, d_model)
        # key, value: (batch, seq_len_k, d_model)
        batch_size = query.size(0)

        # 线性映射并reshape为 (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, num_heads, seq_len_q, seq_len_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (batch, num_heads, seq_len_q, d_k)

        # 拼接多个头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 输出投影
        output = self.out_proj(context)
        return output, attn


# H_s: (batch, N_s, d_model)
# self_attention_output, K_ss = multihead_attn(H_s, H_s, H_s)

# H: (batch, N, d_model), H_s: (batch, N_s, d_model)
# cross_attention_output, K_st = multihead_attn(H, H_s, H_s)
