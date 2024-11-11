import torch
import torch.nn as nn
import numpy as np
import csv

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size_q = q.size(0)

        q = self.linear_q(q).view(batch_size_q, -1, self.num_heads, d_k)
        batch_size_k=k.size(0)
        k = self.linear_k(k)
        k=k.view(batch_size_k, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size_k, -1, self.num_heads, d_v)

        q = q.transpose(0, 2)  # [q_len, h, b_q, d_q]
        v = v.transpose(0, 2)  # [v_len, h, b_kv, d_v]
        k = k.transpose(0, 2).transpose(2, 3)  # [k_len, h, d_k, b_kv]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [q_len(k_len), h, b_q, b_kv]
        if attn_bias is not None:
            x = x + attn_bias
        x_att = torch.softmax(x, dim=3)
        # x = self.att_dropout(x_att)
        # x = x.matmul(v)  # [b, h, q_len, attn] [q_len(k_len), h, b_q, d_v]
        
        # x = x.transpose(0, 2).transpose(1,2).contiguous()  # [b, q_len, h, attn] [b_q, q_len(k_len),h, d_v]
        # x = x.view(batch_size_q, -1, self.num_heads * d_v)
        # x = self.output_layer(x)

        # x=x.squeeze(1)
        return x_att


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size=300, dropout_rate=0.1, attention_dropout_rate=0.1, num_heads=8):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y_att = self.self_attention(y, kv, kv, attn_bias)
        # y = self.self_attention_dropout(y)
        # x = x + y

        return y_att