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

# How the reactant tokens are built before self-attention: "ind" keeps one
# token per reactant slot, "comb" adds every pairwise sum (r_i + r_j).
REACTANT_TOKENS = ("ind", "comb")

# Which regression head turns the reaction vector into a yield: a single linear
# layer ("linear") or a two-layer perceptron ("mlp").
HEADS = ("linear", "mlp")

# How the attended reactant tokens are pooled into the reactant vector: a plain
# masked mean ("mean") or a relation network over all token pairs ("rn").
REACTANT_POOLINGS = ("mean", "rn")


class RelationPooling(nn.Module):
    """
    Relation-network pooling of a set of tokens.

    A plain masked mean treats the tokens as independent. This module also models
    every unordered token pair: a shared network scores the concatenation of the
    pair's sum and element-wise product, the pair vectors are summed, and the
    result is added to the summed per-token (main) effects and normalised. The
    module is permutation-invariant, ignores masked tokens and pairs that touch
    them, and works for a single token (where the pair set is empty).
    """

    def __init__(self, emb_dim: int, dropout: float) -> None:
        """
        Initialize the RelationPooling module.

        Parameters
        ----------
        emb_dim : int
            Dimension of the token vectors.
        dropout : float
            Dropout rate inside the per-token and per-pair networks.
        """
        super(RelationPooling, self).__init__()
        self.emb_dim = emb_dim
        # Per-token main effect.
        self.phi = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        # Per-pair relation, shared across all pairs.
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
    ) -> tuple:
        """
        Pool a batch of token sets into one vector per set.

        Parameters
        ----------
        x : torch.Tensor
            Token vectors of shape [batch_size, num_tokens, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_tokens], True for real
            tokens. Masked tokens, and pairs touching a masked token, do not
            contribute.
        return_pairs : bool, optional
            Whether to also return the per-pair outputs and the pair indices
            (default is False).

        Returns
        -------
        torch.Tensor or tuple
            Pooled vectors of shape [batch_size, emb_dim] and, when
            `return_pairs` is True, the per-pair outputs of shape
            [batch_size, num_pairs, emb_dim] together with the
            (a, b) index tensors used to build them.
        """
        weights = mask.unsqueeze(-1).to(x.dtype)
        # Masked tokens must not leak through their biases.
        x = x * weights

        # Main effects, summed over the real tokens only.
        main = self.phi(x)
        main_sum = (main * weights).sum(dim=1)

        num_tokens = x.shape[1]
        a, b = torch.triu_indices(
            num_tokens, num_tokens, offset=1, device=x.device
        )
        if a.numel() > 0:
            x_a = x[:, a]
            x_b = x[:, b]
            # Symmetric in a and b; concat[x_a + x_b, x_a * x_b].
            pair_desc = torch.cat((x_a + x_b, x_a * x_b), dim=-1)
            pair_out = self.g(pair_desc)
            pair_mask = mask[:, a] & mask[:, b]
            pair_out = pair_out * pair_mask.unsqueeze(-1).to(pair_out.dtype)
            pair_sum = pair_out.sum(dim=1)
        else:
            # No pairs (num_tokens == 1): avoid summing over an empty dimension.
            pair_out = x.new_zeros((x.shape[0], 0, self.emb_dim))
            pair_sum = x.new_zeros((x.shape[0], self.emb_dim))

        pooled = self.norm(main_sum + pair_sum)
        if return_pairs:
            return pooled, pair_out, (a, b)
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
            How the reactant tokens are built before self-attention
            (default is "ind"): "ind" keeps one token per reactant slot, while
            "comb" appends every unordered pair sum (r_i + r_j) after the
            individual tokens, growing the sequence from S to
            S + S * (S - 1) / 2. A pair that touches a padding slot is masked
            out, and only the reactants are expanded (the products are left
            untouched).
        reactant_pooling : str, optional
            How the attended reactant tokens are pooled into the reactant
            vector (default is "mean"): "mean" is the masked mean of the
            token vectors, while "rn" runs a shared relation network over every
            unordered pair of tokens (individual and pair tokens alike) and adds
            the summed per-pair outputs to the summed per-token main effects.
        head : str, optional
            Regression head applied to the reaction vector (default is
            "linear"): "linear" is a single linear layer, "mlp" is a two-layer
            perceptron with a ReLU and dropout between its layers.
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
        if head not in HEADS:
            raise ValueError("head must be one of %s, got %r" % (list(HEADS), head))
        if reactant_pooling not in REACTANT_POOLINGS:
            raise ValueError(
                "reactant_pooling must be one of %s, got %r"
                % (list(REACTANT_POOLINGS), reactant_pooling)
            )
        self.reactant_tokens = reactant_tokens
        self.reactant_pooling = reactant_pooling
        self.head = head
        # "both" runs the same layers over each side in turn, so that the weights
        # are shared and a one-compound side costs nothing extra.
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )
        # Only built for the relation-network pooling, so the default "mean"
        # state dict is unchanged and old checkpoints keep loading.
        if reactant_pooling == "rn":
            self.reactant_pool = RelationPooling(emb_dim, drop_ratio)

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
        else:
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
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool the attended reactant tokens into one reactant vector.

        Dispatches between the masked mean and the relation network as set by
        `reactant_pooling`. The relation network runs over all attended reactant
        tokens, individual and pair alike; when `store_attention` is set, the
        per-pair output norms and their token indices are kept in
        `last_pair_terms`.

        Parameters
        ----------
        tokens : torch.Tensor
            Attended reactant tokens of shape [batch_size, num_tokens, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_tokens], True for real tokens.

        Returns
        -------
        torch.Tensor
            Reactant vectors of shape [batch_size, emb_dim].
        """
        self.last_pair_terms = None
        if self.reactant_pooling == "rn":
            pooled, pair_out, (a, b) = self.reactant_pool(
                tokens, mask, return_pairs=True
            )
            if self.store_attention:
                # Per-pair output norm, averaged over the batch, one entry per
                # pair; the indices are the token slots the pair was built from.
                norms = pair_out.detach().norm(dim=-1).mean(dim=0).cpu().tolist()
                self.last_pair_terms = [
                    (int(i), int(j), float(norm))
                    for i, j, norm in zip(a.tolist(), b.tolist(), norms)
                ]
            return pooled
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
        Append every unordered pair sum of the reactant tokens.

        The individual tokens come first, followed by the pairwise sums in the
        order given by `torch.triu_indices(S, S, offset=1)`; a pair token is
        masked out unless both of its slots hold a real reactant.

        Parameters
        ----------
        tokens : torch.Tensor
            Reactant embeddings of shape [batch_size, num_slots, emb_dim].
        mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_slots], True for real slots.

        Returns
        -------
        tuple
            The expanded tokens of shape
            [batch_size, S + S * (S - 1) // 2, emb_dim], the expanded mask of
            the same length, and the slot metadata as a list of tuples:
            (0,), (1,), ..., (S-1,), (0, 1), (0, 2), ...
        """
        num_slots = tokens.shape[1]
        i, j = torch.triu_indices(
            num_slots,
            num_slots,
            offset=1,
            device=tokens.device,
        )
        pair_tokens = tokens[:, i] + tokens[:, j]
        pair_mask = mask[:, i] & mask[:, j]

        expanded_tokens = torch.cat((tokens, pair_tokens), dim=1)
        expanded_mask = torch.cat((mask, pair_mask), dim=1)

        token_slots = [(slot,) for slot in range(num_slots)] + list(
            zip(i.tolist(), j.tolist())
        )
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

        # Optionally expand the reactants with their pairwise sums. This runs
        # before every attention branch so that "all" records the expanded slot
        # count as the reactant/product boundary below.
        if self.reactant_tokens == "comb":
            r_tokens, r_mask, token_slots = self._add_pair_tokens(r_tokens, r_mask)
        else:
            token_slots = [(slot,) for slot in range(r_tokens.shape[1])]
        self.last_token_slots = token_slots if self.store_attention else None

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

        # A side that does not attend keeps the plain mean of its GNN embeddings.
        reactant_vectors = self._pool_reactants(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)

        reaction_vectors = self._combine(reactant_vectors, product_vectors)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
