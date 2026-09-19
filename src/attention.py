import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class ReactionSelfAttention(nn.Module):
    """
    Single-head self-attention over the compounds of a reaction.

    A reaction is treated as a sentence whose words are the compounds it
    contains: every reactant (reagents included) and every product attend to
    each other through one shared self-attention, so no side is privileged as
    query or key.
    """

    def __init__(self, emb_dim: int, dropout: float = 0.0) -> None:
        """
        Initialize ReactionSelfAttention module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding vectors.
        dropout : float, optional
            Dropout applied to the attention weights (default is 0.0).
        """
        super(ReactionSelfAttention, self).__init__()

        self.emb_dim = emb_dim
        self.scale = emb_dim**-0.5

        self.attention_norm = nn.LayerNorm(emb_dim)
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)
        self.linear_v = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute self-attended value vectors for every compound of a reaction.

        Parameters
        ----------
        x : torch.Tensor
            Compound embeddings of shape [batch_size, num_compounds, emb_dim].
        mask : torch.Tensor, optional
            Boolean tensor of shape [batch_size, num_compounds] that is True for
            the slots holding a real compound and False for padding slots.
            Padded slots are never attended to, and their output rows are zeroed
            so that a later sum ignores them.
        return_weights : bool, optional
            Whether to also return the attention weight matrix (default is False).

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Value vectors of shape [batch_size, num_compounds, emb_dim], and,
            when `return_weights` is True, the attention weights of shape
            [batch_size, num_compounds, num_compounds].
        """
        x = self.attention_norm(x)

        q = self.linear_q(x)  # [batch, len, dim]
        k = self.linear_k(x)  # [batch, len, dim]
        v = self.linear_v(x)  # [batch, len, dim]

        # Attention_weight(Q, K) = softmax((QK^T)/sqrt(dim))
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, len_q, len_k]

        if mask is not None:
            # Padding slots must not contribute as keys. A finite floor is used
            # instead of -inf so that an all-padded row stays finite.
            key_mask = mask.unsqueeze(1)  # [batch, 1, len_k]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        x_att = torch.softmax(scores, dim=-1)
        x_att = self.dropout(x_att)

        if mask is not None:
            # Padding slots must not contribute as queries either.
            x_att = x_att * mask.unsqueeze(-1)  # [batch, len_q, 1]

        out = torch.matmul(x_att, v)  # [batch, len, dim]

        if return_weights:
            return out, x_att

        return out
