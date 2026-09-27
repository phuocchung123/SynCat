import torch
import torch.nn as nn
from typing import Tuple, Union


class RelationPooling(nn.Module):
    """
    Relation-network pooling of a set of compounds into one vector.

    Every real compound contributes a main effect phi(x_i), and every unordered
    pair of real compounds i < j an interaction term g([x_i + x_j, x_i * x_j]):

        h = LayerNorm(sum_i phi(x_i) + sum_{i<j} g([x_i + x_j, x_i * x_j]))

    The pair descriptor is symmetric in i and j and no slot position is used, so
    the result does not depend on the order of the compounds, and padding slots
    take part in neither sum.
    """

    def __init__(self, emb_dim: int, dropout: float = 0.0) -> None:
        """
        Initialize RelationPooling module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the compound vectors and of the pooled vector.
        dropout : float, optional
            Dropout inside the phi and g networks (default is 0.0).
        """
        super(RelationPooling, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        # Shared by every pair.
        self.g = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm = nn.LayerNorm(emb_dim)

        # Upper-triangular (i < j) slot indices, built once per number of slots
        # and device; not part of the state dict.
        self._pair_index_cache = {}

    def pair_indices(self, num_slots: int, device: torch.device) -> torch.Tensor:
        """
        Slot indices of every unordered pair i < j.

        Parameters
        ----------
        num_slots : int
            Number of compound slots S.
        device : torch.device
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            Long tensor of shape [2, S * (S - 1) / 2] holding (i, j) column-wise;
            empty when S == 1.
        """
        key = (num_slots, device)
        if key not in self._pair_index_cache:
            self._pair_index_cache[key] = torch.triu_indices(
                num_slots, num_slots, offset=1, device=device
            )
        return self._pair_index_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_pairs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Pool the compounds of every reaction into one vector.

        Parameters
        ----------
        x : torch.Tensor
            Compound vectors of shape [batch_size, num_slots, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_slots], True for real slots.
        return_pairs : bool, optional
            Whether to also return the per-pair terms (default is False).

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Pooled vectors of shape [batch_size, emb_dim], and, when
            `return_pairs` is True, the pair terms g(...) of shape
            [batch_size, num_pairs, emb_dim] (zero for pairs that involve a
            padding slot) and their slot indices of shape [2, num_pairs].
        """
        padding = ~mask.unsqueeze(-1)  # [batch, slots, 1]
        # Zeroed first, so that whatever sits in a padding slot (even inf/NaN)
        # cannot leak into the sums through the products below.
        x = x.masked_fill(padding, 0.0)

        main = self.phi(x).masked_fill(padding, 0.0).sum(dim=1)

        i, j = self.pair_indices(x.shape[1], x.device)
        x_i, x_j = x[:, i], x[:, j]  # [batch, pairs, emb_dim]
        pair_terms = self.g(torch.cat((x_i + x_j, x_i * x_j), dim=-1))
        pair_padding = padding[:, i] | padding[:, j]  # [batch, pairs, 1]
        pair_terms = pair_terms.masked_fill(pair_padding, 0.0)

        # A sum over zero pairs (a single slot) is a zero vector.
        pooled = self.norm(main + pair_terms.sum(dim=1))

        if return_pairs:
            return pooled, pair_terms, torch.stack((i, j))

        return pooled
