import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, emb_dim):
        super(SingleHeadAttention, self).__init__()

        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.attention_norm = nn.LayerNorm(emb_dim)
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k):
        q=self.attention_norm(q) #query
        k=self.attention_norm(k) #key
        
        q = self.linear_q(q).view(-1, self.emb_dim) #[len_q, dim_q]
        k = self.linear_k(k).view( -1, self.emb_dim) #[len_k, dim_k]

        k = k.transpose(0, 1)  # [dim_k, len_k]

        # Attention_weight(Q, K) = softmax((QK^T)/sqrt(dim))
        q = q * self.scale
        x = torch.matmul(q, k)  # [len_q, len_k]

        x_att = torch.softmax(x, dim=1)

        return x_att