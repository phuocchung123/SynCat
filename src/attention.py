import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class ReactionSelfAttention(nn.Module):
    """
    Multi-head self-attention over a set of compounds of a reaction.

    The compounds are treated as the words of a sentence that attend to each
    other; the model applies it to the reactants (reagents included) only, i.e.
    the compounds on the left of ">>".

    The embedding is split into `num_heads` heads of size `emb_dim // num_heads`
    that attend independently, so each head can learn its own interaction
    pattern. With a single head the module reduces exactly to plain single-head
    attention (no output projection), which keeps older checkpoints loadable.
    """

    def __init__(self, emb_dim: int, num_heads: int = 1, dropout: float = 0.0) -> None:
        """
        Initialize ReactionSelfAttention module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding vectors.
        num_heads : int, optional
            Number of attention heads; must divide `emb_dim` (default is 1).
        dropout : float, optional
            Dropout applied to the attention weights (default is 0.0).
        """
        super(ReactionSelfAttention, self).__init__()

        if num_heads < 1 or emb_dim % num_heads != 0:
            raise ValueError(
                "num_heads must be a positive divisor of emb_dim (%d), got %d"
                % (emb_dim, num_heads)
            )

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.attention_norm = nn.LayerNorm(emb_dim)
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)
        self.linear_v = nn.Linear(emb_dim, emb_dim)
        # Mixes the concatenated heads; unnecessary (and omitted) for one head.
        self.linear_o = nn.Linear(emb_dim, emb_dim) if num_heads > 1 else None

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
            so that a later masked average ignores them.
        return_weights : bool, optional
            Whether to also return the attention weight matrix (default is False).

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Value vectors of shape [batch_size, num_compounds, emb_dim], and,
            when `return_weights` is True, the attention weights averaged over
            heads, of shape [batch_size, num_compounds, num_compounds].
        """
        x = self.attention_norm(x)

        batch_size, length, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # [batch, len, dim] -> [batch, heads, len, head_dim]
            return t.view(batch_size, length, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        q = split_heads(self.linear_q(x))
        k = split_heads(self.linear_k(x))
        v = split_heads(self.linear_v(x))

        # Attention_weight(Q, K) = softmax((QK^T)/sqrt(head_dim)), per head
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, len_q, len_k]

        if mask is not None:
            # Padding slots must not contribute as keys. A finite floor is used
            # instead of -inf so that an all-padded row stays finite.
            key_mask = mask[:, None, None, :]  # [batch, 1, 1, len_k]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        x_att = torch.softmax(scores, dim=-1)
        x_att = self.dropout(x_att)

        if mask is not None:
            # Padding slots must not contribute as queries either.
            x_att = x_att * mask[:, None, :, None]  # [batch, 1, len_q, 1]

        out = torch.matmul(x_att, v)  # [batch, heads, len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, length, self.emb_dim)

        if self.linear_o is not None:
            out = self.linear_o(out)
            if mask is not None:
                # The output bias would otherwise make padded rows non-zero.
                out = out * mask.unsqueeze(-1)

        if return_weights:
            return out, x_att.mean(dim=1)

        return out
