import numpy as np
import torch
import torch.nn as nn
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

# Reactant token representations: individual slots or with pairwise combination tokens.
REACTANT_TOKENS = ("ind", "comb")

# Reactant pooling methods: mean reduction or relation network.
REACTANT_POOLINGS = ("mean", "rn")

# Regression head architectures.
HEADS = ("linear", "mlp")


class RelationPooling(nn.Module):
    """
    Relation-network pooling for compound tokens.

    Computes token representations using main effects (per-token MLP) and
    pairwise relational effects (shared pair MLP over symmetric descriptors),
    masked appropriately, and combined via LayerNorm.
    """

    def __init__(self, emb_dim: int, dropout: float) -> None:
        """
        Initialize the relation-network pooling layer.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension of compound tokens.
        dropout : float
            Dropout probability for the MLPs.
        """
        super(RelationPooling, self).__init__()
        self.emb_dim = emb_dim
        self.phi = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.g = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_pairs: bool = False,
    ):
        """
        Pool tokens using relation network.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape [batch_size, T, emb_dim].
        mask : torch.Tensor
            Boolean mask of shape [batch_size, T], True for real tokens.
        return_pairs : bool, optional
            Whether to return per-pair outputs and slot indices (default is False).

        Returns
        -------
        torch.Tensor or tuple
            Pooled vector of shape [batch_size, emb_dim], or if return_pairs is True:
            (pooled_vector, pair_outputs, (a, b)).
        """
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        x = x * mask_f

        batch_size, num_tokens, emb_dim = x.shape
        phi_x = self.phi(x)
        main_sum = (phi_x * mask_f).sum(dim=1)

        if num_tokens < 2:
            pair_sum = x.new_zeros(batch_size, emb_dim)
            pair_out = x.new_zeros(batch_size, 0, emb_dim)
            a = torch.empty(0, dtype=torch.long, device=x.device)
            b = torch.empty(0, dtype=torch.long, device=x.device)
        else:
            a, b = torch.triu_indices(num_tokens, num_tokens, offset=1, device=x.device)
            x_a = x[:, a]
            x_b = x[:, b]
            pair_desc = torch.cat([x_a + x_b, x_a * x_b], dim=-1)
            pair_out = self.g(pair_desc)
            pair_mask = (mask[:, a] & mask[:, b]).unsqueeze(-1).to(x.dtype)
            pair_sum = (pair_out * pair_mask).sum(dim=1)

        out = self.norm(main_sum + pair_sum)
        if return_pairs:
            return out, pair_out, (a, b)
        return out


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
        reactant_tokens: str = "ind",
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
            layers tends to smooth every compound towards the same vector.
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
        reactant_tokens : str, optional
            Whether to use individual reactant tokens ("ind", default) or expand
            reactants with unordered pairwise combination tokens ("comb") before
            self-attention.
        reactant_pooling : str, optional
            How to pool attended reactant tokens into the reactant vector (default is "mean"):
            "mean" for masked average; "rn" for relation-network pooling.
        head : str, optional
            Regression head architecture (default is "linear"):
            "linear" for a single linear layer; "mlp" for a 2-layer MLP with
            ReLU and dropout.
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
            "reactant_tokens": str(reactant_tokens),
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
        self.attention_on = attention_on
        if reactant_tokens not in REACTANT_TOKENS:
            raise ValueError(
                "reactant_tokens must be one of %s, got %r"
                % (list(REACTANT_TOKENS), reactant_tokens)
            )
        self.reactant_tokens = reactant_tokens
        if reactant_pooling not in REACTANT_POOLINGS:
            raise ValueError(
                "reactant_pooling must be one of %s, got %r"
                % (list(REACTANT_POOLINGS), reactant_pooling)
            )
        self.reactant_pooling = reactant_pooling
        if head not in HEADS:
            raise ValueError(
                "head must be one of %s, got %r"
                % (list(HEADS), head)
            )
        self.head = head

        # "both" runs the same layers over each side in turn, so that the weights
        # are shared and a one-compound side costs nothing extra.
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )

        # Reactant pooling layer
        if reactant_pooling == "rn":
            self.reactant_pool = RelationPooling(emb_dim, drop_ratio)
        else:
            self.reactant_pool = None

        # The reaction is summarised by combining the reactant vector (mean of
        # the self-attended reactant value vectors) with the product vector
        # (mean of the product GNN embeddings).
        if reaction_combine not in REACTION_COMBINE_DIMS:
            raise ValueError(
                "reaction_combine must be one of %s, got %r"
                % (sorted(REACTION_COMBINE_DIMS), reaction_combine)
            )
        self.reaction_combine = reaction_combine
        input_dim = REACTION_COMBINE_DIMS[reaction_combine] * emb_dim
        if head == "linear":
            self.regressor = nn.Linear(input_dim, 1)
        elif head == "mlp":
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(emb_dim, 1),
            )

        # Optional bookkeeping for interpretation; disabled by default so that
        # training does not accumulate attention matrices. Only the last layer's
        # weights, averaged over heads, are kept: one matrix, or one per side
        # with `attention_on="both"`.
        self.store_attention = False
        self.last_attention = None
        self.last_token_slots = None
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
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool attended reactant tokens into a single reactant vector.

        Parameters
        ----------
        tokens : torch.Tensor
            Attended reactant tokens of shape [batch_size, num_tokens, emb_dim].
        mask : torch.Tensor
            Boolean mask of shape [batch_size, num_tokens], True for real tokens.

        Returns
        -------
        torch.Tensor
            Pooled reactant vector of shape [batch_size, emb_dim].
        """
        if self.reactant_pooling == "rn":
            if self.store_attention:
                pooled, pair_out, (a, b) = self.reactant_pool(
                    tokens, mask, return_pairs=True
                )
                pair_norms = pair_out.norm(dim=-1).detach().cpu().tolist()
                pair_indices = [
                    (int(u.item()), int(v.item())) for u, v in zip(a, b)
                ]
                self.last_pair_terms = (pair_norms, pair_indices)
                return pooled
            self.last_pair_terms = None
            return self.reactant_pool(tokens, mask)
        self.last_pair_terms = None
        return self._masked_mean(tokens, mask)

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

    def _add_pair_tokens(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        """
        Expand reactant tokens with unordered pairwise combination tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            Compound embeddings of shape [batch_size, S, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, S], True for real slots.

        Returns
        -------
        tuple
            (expanded_tokens, expanded_mask, token_slots) where expanded_tokens
            has shape [batch_size, S + S * (S - 1) // 2, emb_dim], expanded_mask
            has shape [batch_size, S + S * (S - 1) // 2], and token_slots is a list
            of slot index tuples: (0,), ..., (S-1,), (0, 1), (0, 2), ...
        """
        S = tokens.shape[1]
        i, j = torch.triu_indices(
            S,
            S,
            offset=1,
            device=tokens.device,
        )
        pair_tokens = tokens[:, i] + tokens[:, j]
        pair_mask = mask[:, i] & mask[:, j]

        expanded_tokens = torch.cat((tokens, pair_tokens), dim=1)
        expanded_mask = torch.cat((mask, pair_mask), dim=1)
        token_slots = [(s,) for s in range(S)] + [
            (int(u.item()), int(v.item())) for u, v in zip(i, j)
        ]
        return expanded_tokens, expanded_mask, token_slots

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

        if self.reactant_tokens == "comb":
            r_tokens, r_mask, token_slots = self._add_pair_tokens(r_tokens, r_mask)
        else:
            token_slots = [(s,) for s in range(r_tokens.shape[1])]

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
            self.last_token_slots = token_slots
        else:
            self.last_token_slots = None

        # A side that does not attend keeps the plain mean of its GNN embeddings.
        reactant_vectors = self._pool_reactants(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)

        reaction_vectors = self._combine(reactant_vectors, product_vectors)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
