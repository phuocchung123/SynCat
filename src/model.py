import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union
from gin import GIN
from attention import ReactionSelfAttention

# How the reactant vector r and the product vector p are combined into the
# reaction vector, with the size of the result in multiples of emb_dim.
REACTION_COMBINE_DIMS = {
    "concat": 2,  # [r, p]
    "sum": 1,  # r + p
    "sub": 1,  # p - r, the change from reactants to products
    "mul": 1,  # r * p, element-wise
    "concat_sub": 3,  # [r, p, p - r]
    "interaction": 4,  # [r, p, |p - r|, r * p]
}

# Which compounds take part in the self-attention.
ATTENTION_TARGETS = ("reactants", "products", "both", "all", "none")

# How the reactant slots are pooled into the reactant vector: masked mean, or a
# relation network over every compound and every pair of compounds.
REACTANT_POOLINGS = ("mean", "rn")

# Regression head on the reaction vector.
HEADS = ("linear", "mlp")


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

    def __init__(self, emb_dim: int, dropout: float) -> None:
        """
        Initialize RelationPooling module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the compound vectors and of the pooled vector.
        dropout : float
            Dropout inside the phi and g networks.
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

    def _pair_indices(
        self, num_slots: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slot indices (i, j) of every unordered pair i < j, each of shape
        [num_slots * (num_slots - 1) / 2]; empty when num_slots == 1.
        """
        key = (num_slots, device)
        if key not in self._pair_index_cache:
            i, j = torch.triu_indices(num_slots, num_slots, offset=1, device=device)
            self._pair_index_cache[key] = (i, j)
        return self._pair_index_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_pairs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, tuple]]:
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
        torch.Tensor or tuple
            Pooled vectors of shape [batch_size, emb_dim], and, when
            `return_pairs` is True, the pair terms g(...) of shape
            [batch_size, num_pairs, emb_dim] (zero for pairs that involve a
            padding slot) and the (i, j) slot index tensors, each of shape
            [num_pairs].
        """
        padding = ~mask.unsqueeze(-1)  # [batch, slots, 1]
        # Padding slots hold the embedding of a dummy graph, which is not zero;
        # zeroed first so that it cannot leak into the pair products.
        x = x.masked_fill(padding, 0.0)

        main = self.phi(x).masked_fill(padding, 0.0).sum(dim=1)

        i, j = self._pair_indices(x.shape[1], x.device)
        x_i, x_j = x[:, i], x[:, j]  # [batch, pairs, emb_dim]
        pair_terms = self.g(torch.cat((x_i + x_j, x_i * x_j), dim=-1))
        pair_padding = padding[:, i] | padding[:, j]  # [batch, pairs, 1]
        pair_terms = pair_terms.masked_fill(pair_padding, 0.0)

        # A sum over zero pairs (a single slot) is a zero vector.
        pooled = self.norm(main + pair_terms.sum(dim=1))

        if return_pairs:
            return pooled, pair_terms, (i, j)

        return pooled


class model(nn.Module):
    """
    Graph-based regression model with reaction-level self attention for
    reaction-yield prediction.

    Every compound of a reaction "reactants >> products" is encoded by the
    shared GNN. Which of them then attend to each other in the (multi-head)
    self-attention block is set by `attention_on`: by default only the reactants
    (reagents are treated as reactants), i.e. the compounds on the left of ">>".
    Each side is averaged into one vector - the attended value vectors where
    attention applies, the plain GNN embeddings where it does not - and the two
    vectors are combined into the reaction vector as set by `reaction_combine`
    (by default the concatenation [reactant vector, product vector]).
    `reactant_pooling` can replace the reactant average by a relation network
    (`RelationPooling`), and `head` the linear regressor by a small MLP; with
    `attention_on="none"` and `reactant_pooling="rn"` the relation network pools
    the GNN outputs directly.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float,
        num_attention_layer: int = 1,
        num_heads: int = 1,
        reaction_combine: str = "concat",
        attention_on: str = "reactants",
        reactant_pooling: str = "mean",
        head: str = "linear",
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        num_layer : int
            Number of GNN layers.
        emb_dim : int
            Embedding dimension.
        drop_ratio : float
            Dropout rate.
        num_attention_layer : int, optional
            Number of stacked self-attention layers over the reactants (default is 1).
            The sequence is only a handful of compounds long, so depth beyond 2-3
            layers tends to smooth every compound towards the same vector. It
            must be at least 1 even with `attention_on="none"`, where no
            attention layer is built.
        num_heads : int, optional
            Number of heads in every self-attention layer; must divide `emb_dim`
            (default is 1, i.e. single-head attention).
        reaction_combine : str, optional
            How the reactant vector r and the product vector p are combined
            into the reaction vector (default is "concat"):
            "concat" [r, p]; "sum" r + p; "sub" p - r; "mul" r * p;
            "concat_sub" [r, p, p - r]; "interaction" [r, p, |p - r|, r * p].
        attention_on : str, optional
            Which compounds attend to each other (default is "reactants"):
            "reactants" only the left of ">>", "products" only the right,
            "both" each side separately through the same attention layers,
            "all" every compound of the reaction in one shared attention, and
            "none" no attention at all (the vectors are then plain means of the
            GNN embeddings). A side that does not attend is averaged directly.
        reactant_pooling : str, optional
            How the reactant slots become the reactant vector (default is "mean"):
            "mean" the masked mean; "rn" a relation network (`RelationPooling`),
            LayerNorm of the summed per-compound terms phi(x_i) and per-pair
            terms g([x_i + x_j, x_i * x_j]) over all pairs i < j. It pools
            whatever the reactant slots hold after attention: attended vectors,
            or the raw GNN outputs with `attention_on` "products" or "none". The
            product side is always a masked mean.
        head : str, optional
            Regression head on the reaction vector (default is "linear"):
            "linear" a single linear layer; "mlp" Linear(k * emb_dim, emb_dim)
            -> ReLU -> Dropout -> Linear(emb_dim, 1), with k given by
            `REACTION_COMBINE_DIMS[reaction_combine]`.
        """
        super(model, self).__init__()
        # Everything needed to rebuild this architecture; saved in checkpoints
        # so that a model can be reloaded without repeating its settings.
        self.config = {
            "node_in_feats": int(node_in_feats),
            "edge_in_feats": int(edge_in_feats),
            "num_layer": int(num_layer),
            "emb_dim": int(emb_dim),
            "drop_ratio": float(drop_ratio),
            "num_attention_layer": int(num_attention_layer),
            "num_heads": int(num_heads),
            "reaction_combine": str(reaction_combine),
            "attention_on": str(attention_on),
            "reactant_pooling": str(reactant_pooling),
            "head": str(head),
        }
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            num_layer,
            emb_dim,
            drop_ratio,
        )

        if num_attention_layer < 1:
            raise ValueError(
                "num_attention_layer must be at least 1, got %d" % num_attention_layer
            )
        if attention_on not in ATTENTION_TARGETS:
            raise ValueError(
                "attention_on must be one of %s, got %r"
                % (list(ATTENTION_TARGETS), attention_on)
            )
        if reactant_pooling not in REACTANT_POOLINGS:
            raise ValueError(
                "reactant_pooling must be one of %s, got %r"
                % (list(REACTANT_POOLINGS), reactant_pooling)
            )
        if head not in HEADS:
            raise ValueError("head must be one of %s, got %r" % (list(HEADS), head))
        self.attention_on = attention_on
        # "both" runs the same layers over each side in turn, so that the weights
        # are shared and a one-compound side costs nothing extra.
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )

        # The reaction is summarised by combining the reactant vector (mean of
        # the self-attended reactant value vectors) with the product vector
        # (mean of the product GNN embeddings).
        if reaction_combine not in REACTION_COMBINE_DIMS:
            raise ValueError(
                "reaction_combine must be one of %s, got %r"
                % (sorted(REACTION_COMBINE_DIMS), reaction_combine)
            )
        self.reaction_combine = reaction_combine
        self.head = head
        if head == "linear":
            self.regressor = torch.nn.Linear(
                REACTION_COMBINE_DIMS[reaction_combine] * emb_dim, 1
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(REACTION_COMBINE_DIMS[reaction_combine] * emb_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(emb_dim, 1),
            )

        # Built last, so that every other layer starts from the same weights as
        # with the masked mean for the same seed, and a "mean" model has exactly
        # the parameters it had before this option existed.
        self.reactant_pooling = reactant_pooling
        self.reactant_pool = (
            RelationPooling(emb_dim, drop_ratio) if reactant_pooling == "rn" else None
        )

        # Optional bookkeeping for interpretation; disabled by default so that
        # training does not accumulate attention matrices. Only the last layer's
        # weights, averaged over heads, are kept: one matrix, or one per side
        # with `attention_on="both"`. With `reactant_pooling="rn"` the norms of
        # the per-pair terms are kept too, with their slot indices.
        self.store_attention = False
        self.last_attention = None
        self.last_pair_terms = None

    @classmethod
    def from_config(cls, config: dict) -> "model":
        """
        Rebuild a model from the `config` of another one (e.g. a checkpoint's
        "model_config").

        Parameters
        ----------
        config : dict
            Constructor arguments, as stored in `model.config`.

        Returns
        -------
        model
            A freshly initialised model with that architecture.
        """
        return cls(**config)

    def _attend(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run the stack of self-attention layers over one set of compounds.

        Intermediate layers refine the compound representations through a
        residual connection, which keeps a deeper stack stable; the last layer
        emits the value vectors that are averaged into a side's vector.

        Parameters
        ----------
        tokens : torch.Tensor
            Compound embeddings of shape [batch_size, num_slots, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_slots], True for real slots.

        Returns
        -------
        tuple
            The attended vectors, of the same shape as `tokens`, and the last
            layer's attention weights when `store_attention` is set (else None).
        """
        for attention_layer in self.attention_layers[:-1]:
            tokens = tokens + attention_layer(tokens, mask=mask)

        attended = self.attention_layers[-1](
            tokens, mask=mask, return_weights=self.store_attention
        )
        if self.store_attention:
            attended, att_weights = attended
            return attended, att_weights.detach().cpu().tolist()

        return attended, None

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Average the vectors of the real (non-padding) slots.

        Parameters
        ----------
        x : torch.Tensor
            Vectors of shape [batch_size, num_slots, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_slots], True for real slots.

        Returns
        -------
        torch.Tensor
            Mean vectors of shape [batch_size, emb_dim]; zero when a reaction has
            no real slot.
        """
        weights = mask.unsqueeze(-1).to(x.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / count

    def _pool_reactants(
        self, r_tokens: torch.Tensor, r_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool the reactant slots into the reactant vector, as set by
        `reactant_pooling`.

        Parameters
        ----------
        r_tokens : torch.Tensor
            Reactant vectors of shape [batch_size, num_slots, emb_dim].
        r_mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_slots], True for real slots.

        Returns
        -------
        torch.Tensor
            Reactant vectors of shape [batch_size, emb_dim].
        """
        if self.reactant_pooling == "mean":
            return self._masked_mean(r_tokens, r_mask)

        if not self.store_attention:
            return self.reactant_pool(r_tokens, r_mask)

        pooled, pair_terms, (i, j) = self.reactant_pool(
            r_tokens, r_mask, return_pairs=True
        )
        # Norms of shape [batch_size, num_pairs] (zero where a pair involves a
        # padding slot) and the (i, j) slots of every pair.
        self.last_pair_terms = {
            "norms": pair_terms.norm(dim=-1).detach().cpu().tolist(),
            "pairs": torch.stack((i, j), dim=1).cpu().tolist(),
        }
        return pooled

    def _combine(self, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Combine the reactant and product vectors into the reaction vector.

        Parameters
        ----------
        r : torch.Tensor
            Reactant vectors of shape [batch_size, emb_dim].
        p : torch.Tensor
            Product vectors of shape [batch_size, emb_dim].

        Returns
        -------
        torch.Tensor
            Reaction vectors of shape [batch_size, k * emb_dim], with k given by
            `REACTION_COMBINE_DIMS[self.reaction_combine]`.
        """
        mode = self.reaction_combine
        if mode == "concat":
            return torch.cat((r, p), dim=1)
        if mode == "sum":
            return r + p
        if mode == "sub":
            return p - r
        if mode == "mul":
            return r * p
        if mode == "concat_sub":
            return torch.cat((r, p, p - r), dim=1)
        return torch.cat((r, p, torch.abs(p - r), r * p), dim=1)  # interaction

    def forward(
        self,
        rmols: list,
        pmols: list,
        r_dummy: list,
        p_dummy: list,
        device: torch.device,
    ) -> tuple:
        """
        Forward pass of the model.

        Parameters
        ----------
        rmols : list
            List of reactant graph data objects.
        pmols : list
            List of product graph data objects.
        r_dummy : list
            Per-reaction list of reactant-slot flags, True where a reactant is present.
        p_dummy : list
            Per-reaction list of product-slot flags, True where a product is present.
        device : torch.device
            Device to run the computations.

        Returns
        -------
        tuple
            Predicted yields of shape [batch_size] and the reaction vectors of
            shape [batch_size, k * emb_dim] as a list (k depends on
            `reaction_combine`).
        """
        # Shape [batch_size, num_slots, emb_dim], one token per compound slot.
        r_tokens = torch.stack([self.gnn(rmol) for rmol in rmols], dim=1).to(device)
        p_tokens = torch.stack([self.gnn(pmol) for pmol in pmols], dim=1).to(device)

        # Padding slots carry a dummy graph, so they are excluded from attention
        # and from the averages.
        r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
        p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)

        weights = {}
        if self.attention_on == "all":
            # One shared attention over the whole reaction; the two sides are
            # only told apart afterwards, when each is averaged on its own.
            n_reactant_slots = r_tokens.shape[1]
            attended, weights["all"] = self._attend(
                torch.cat((r_tokens, p_tokens), dim=1),
                torch.cat((r_mask, p_mask), dim=1),
            )
            r_tokens = attended[:, :n_reactant_slots]
            p_tokens = attended[:, n_reactant_slots:]
        else:
            if self.attention_on in ("reactants", "both"):
                r_tokens, weights["reactants"] = self._attend(r_tokens, r_mask)
            if self.attention_on in ("products", "both"):
                p_tokens, weights["products"] = self._attend(p_tokens, p_mask)

        if self.store_attention:
            # One matrix for a single attended set, one per side for "both".
            self.last_attention = (
                weights if len(weights) > 1 else next(iter(weights.values()), None)
            )

        # A side that does not attend is pooled from its raw GNN embeddings; the
        # reactant side as set by `reactant_pooling`, the product side by the mean.
        reactant_vectors = self._pool_reactants(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)

        reaction_vectors = self._combine(reactant_vectors, product_vectors)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
