import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """
    Single-head attention mechanism.
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initialize SingleHeadAttention module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding vectors.
        """
        super(SingleHeadAttention, self).__init__()

        self.emb_dim = emb_dim
        self.scale = emb_dim**-0.5

        self.attention_norm = nn.LayerNorm(emb_dim)
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product attention weights.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape [num_q, emb_dim].
        k : torch.Tensor
            Key tensor of shape [num_k, emb_dim].

        Returns
        -------
        torch.Tensor
            Attention weight matrix of shape [num_q, num_k].
        """
        q = self.attention_norm(q)  # query
        k = self.attention_norm(k)  # key

        q = self.linear_q(q).view(-1, self.emb_dim)  # [len_q, dim_q]
        k = self.linear_k(k).view(-1, self.emb_dim)  # [len_k, dim_k]

        k = k.transpose(0, 1)  # [dim_k, len_k]

        # Attention_weight(Q, K) = softmax((QK^T)/sqrt(dim))
        q = q * self.scale
        x = torch.matmul(q, k)  # [len_q, len_k]

        x_att = torch.softmax(x, dim=1)

        return x_att
