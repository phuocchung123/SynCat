import numpy as np
import torch
import torch.nn as nn
from gin import GIN
from attention import CompoundCrossAttention, ReactionSelfAttention

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

# How a reaction is turned into one vector.
ARCHITECTURES = ("attention_pool", "cross_center")


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
        architecture: str = "attention_pool",
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
        architecture : str, optional
            How a reaction becomes one vector (default is "attention_pool"):

            - "attention_pool": self attention as set by `attention_on`, each
              side averaged, the two sides merged by `reaction_combine`.
            - "cross_center": reactants and products weight each other through
              cross attention and are pooled by a weighted sum into r and p; the
              reaction centre is c = r - p; a second cross attention, with c as
              the query and the reactants as keys and values, pools the reactants
              once more into a, and the reaction vector is [a, c]. With this
              architecture `attention_on` and `reaction_combine` are unused.
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
            "architecture": str(architecture),
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
        if architecture not in ARCHITECTURES:
            raise ValueError(
                "architecture must be one of %s, got %r"
                % (list(ARCHITECTURES), architecture)
            )
        self.architecture = architecture
        if attention_on not in ATTENTION_TARGETS:
            raise ValueError(
                "attention_on must be one of %s, got %r"
                % (list(ATTENTION_TARGETS), attention_on)
            )
        self.attention_on = attention_on
        # "both" runs the same layers over each side in turn, so that the weights
        # are shared and a one-compound side costs nothing extra.
        uses_self_attention = architecture == "attention_pool" and attention_on != "none"
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if uses_self_attention else 0)
            ]
        )

        if architecture == "cross_center":
            # One module for the reactant/product pass, in both directions as in
            # the original SynCat model, and one for the reaction-centre query.
            self.pair_attention = CompoundCrossAttention(emb_dim)
            self.center_attention = CompoundCrossAttention(emb_dim)

        # The reaction is summarised by combining the reactant vector (mean of
        # the self-attended reactant value vectors) with the product vector
        # (mean of the product GNN embeddings).
        if reaction_combine not in REACTION_COMBINE_DIMS:
            raise ValueError(
                "reaction_combine must be one of %s, got %r"
                % (sorted(REACTION_COMBINE_DIMS), reaction_combine)
            )
        self.reaction_combine = reaction_combine
        self.regressor = torch.nn.Linear(
            2 * emb_dim
            if architecture == "cross_center"
            else REACTION_COMBINE_DIMS[reaction_combine] * emb_dim,
            1,
        )

        # Optional bookkeeping for interpretation; disabled by default so that
        # training does not accumulate attention matrices. Only the last layer's
        # weights, averaged over heads, are kept: one matrix, or one per side
        # with `attention_on="both"`.
        self.store_attention = False
        self.last_attention = None

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
    def _pooled_weights(
        weights: torch.Tensor, q_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Average an attention matrix over its real queries, giving one weight per
        key, as in the original SynCat model.

        Parameters
        ----------
        weights : torch.Tensor
            Attention weights of shape [batch_size, len_q, len_k]; the rows of
            padding queries are zero.
        q_mask : torch.Tensor
            Boolean tensor of shape [batch_size, len_q], True for a real query.

        Returns
        -------
        torch.Tensor
            One weight per key, of shape [batch_size, len_k].
        """
        n_queries = q_mask.sum(dim=1, keepdim=True).clamp(min=1).to(weights.dtype)
        return weights.sum(dim=1) / n_queries

    def _cross_center(
        self,
        r_tokens: torch.Tensor,
        p_tokens: torch.Tensor,
        r_mask: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the reaction vector of the "cross_center" architecture.

        Reactants and products weight each other through cross attention; each
        side is pooled by the weighted sum of its compound embeddings, giving r
        and p. Their difference c = r - p is the reaction centre, which then
        queries the reactants through a second cross attention, pooling them into
        a. The reaction vector is [a, c].

        Parameters
        ----------
        r_tokens : torch.Tensor
            Reactant embeddings of shape [batch_size, num_r_slots, emb_dim].
        p_tokens : torch.Tensor
            Product embeddings of shape [batch_size, num_p_slots, emb_dim].
        r_mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_r_slots].
        p_mask : torch.Tensor
            Boolean tensor of shape [batch_size, num_p_slots].

        Returns
        -------
        torch.Tensor
            Reaction vectors of shape [batch_size, 2 * emb_dim].
        """
        # Every product attends to every reactant and vice versa; averaging over
        # the queries leaves one weight per compound.
        w_r = self._pooled_weights(
            self.pair_attention(p_tokens, r_tokens, q_mask=p_mask, k_mask=r_mask),
            p_mask,
        )
        w_p = self._pooled_weights(
            self.pair_attention(r_tokens, p_tokens, q_mask=r_mask, k_mask=p_mask),
            r_mask,
        )

        # Weighted sum of the compounds of each side.
        reactant_vectors = torch.sum(w_r.unsqueeze(-1) * r_tokens, dim=1)
        product_vectors = torch.sum(w_p.unsqueeze(-1) * p_tokens, dim=1)

        center_vectors = reactant_vectors - product_vectors

        # The reaction centre asks the reactants once more: query is the centre,
        # keys and values are the reactant compounds.
        w_center = self.center_attention(
            center_vectors.unsqueeze(1), r_tokens, k_mask=r_mask
        )
        attended_reactants = torch.sum(w_center.transpose(1, 2) * r_tokens, dim=1)

        if self.store_attention:
            self.last_attention = {
                "reactants": w_r.detach().cpu().tolist(),
                "products": w_p.detach().cpu().tolist(),
                "center": w_center.squeeze(1).detach().cpu().tolist(),
            }

        return torch.cat((attended_reactants, center_vectors), dim=1)

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

        if self.architecture == "cross_center":
            reaction_vectors = self._cross_center(r_tokens, p_tokens, r_mask, p_mask)
            out = self.regressor(reaction_vectors).squeeze(-1)
            return out, reaction_vectors.tolist()

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
        reactant_vectors = self._masked_mean(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)

        reaction_vectors = self._combine(reactant_vectors, product_vectors)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
