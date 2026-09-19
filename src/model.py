import numpy as np
import torch
import torch.nn as nn
from gin import GIN
from attention import ReactionSelfAttention


class model(nn.Module):
    """
    Graph-based regression model with reaction-level self attention for
    reaction-yield prediction.

    A reaction is read like a sentence whose words are its compounds: every
    reactant (reagents are treated as reactants) and every product is encoded by
    the shared GNN, the resulting compound embeddings attend to each other in a
    single self-attention block, and the value vectors of the whole reaction are
    summed into one reaction vector.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float,
        num_attention_layer: int = 1,
        use_role_embedding: bool = True,
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
            Number of stacked self-attention layers over the reaction (default is 1).
            The sequence is only a handful of compounds long, so depth beyond 2-3
            layers tends to smooth every compound towards the same vector.
        use_role_embedding : bool, optional
            Whether to add a learned role embedding (reactant vs. product) to
            each compound embedding before self attention (default is True).
            Self attention is permutation invariant, so without it the model
            cannot tell which side of ">>" a compound comes from.
        """
        super(model, self).__init__()
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
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, dropout=drop_ratio)
                for _ in range(num_attention_layer)
            ]
        )

        # Reactants and products are distinguished by a role embedding rather
        # than by being routed through separate attention branches.
        self.use_role_embedding = use_role_embedding
        if use_role_embedding:
            self.role_embedding = nn.Embedding(2, emb_dim)

        # The reaction is summarised by one vector: the sum of the self-attended
        # value vectors of all its compounds.
        self.regressor = torch.nn.Linear(emb_dim, 1)

        # Optional bookkeeping for interpretation; disabled by default so that
        # training does not accumulate attention matrices. Only the last layer's
        # weights are kept.
        self.store_attention = False
        self.last_attention = None

    def _build_mask(
        self, r_dummy: list, p_dummy: list, device: torch.device
    ) -> torch.Tensor:
        """
        Build the boolean mask marking the slots that hold a real compound.

        Parameters
        ----------
        r_dummy : list
            Per-reaction list of reactant-slot flags, True where a reactant is present.
        p_dummy : list
            Per-reaction list of product-slot flags, True where a product is present.
        device : torch.device
            Device to place the mask on.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [batch_size, rmol_max_cnt + pmol_max_cnt].
        """
        r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
        p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)

        return torch.cat((r_mask, p_mask), dim=1)

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
            List of masks/indices for reactants in batch.
        p_dummy : list
            List of masks/indices for products in batch.
        device : torch.device
            Device to run the computations.

        Returns
        -------
        tuple
            Predicted yields of shape [batch_size] and the reaction vectors of
            shape [batch_size, emb_dim] as a list.
        """
        r_graph_feats = torch.stack([self.gnn(rmol) for rmol in rmols])
        p_graph_feats = torch.stack([self.gnn(pmol) for pmol in pmols])

        # The "sentence" of a reaction: reactant slots followed by product slots,
        # one token per compound. Shape [batch_size, num_compounds, emb_dim].
        tokens = torch.cat((r_graph_feats, p_graph_feats), dim=0).transpose(0, 1)
        tokens = tokens.to(device)

        # Padding slots carry a dummy graph, so they are excluded from attention
        # and from the final sum.
        mask = self._build_mask(r_dummy, p_dummy, device)

        if self.use_role_embedding:
            roles = torch.cat(
                (
                    torch.zeros(len(rmols), dtype=torch.long, device=device),
                    torch.ones(len(pmols), dtype=torch.long, device=device),
                )
            )
            tokens = tokens + self.role_embedding(roles).unsqueeze(0)

        # Self attention over the whole reaction: reactants and products are
        # queries and keys for each other, with no separate cross-attention step.
        # Intermediate layers refine the compound representations through a
        # residual connection, which keeps a deeper stack stable; the last layer
        # emits the value vectors that are summed into the reaction vector.
        for attention_layer in self.attention_layers[:-1]:
            tokens = tokens + attention_layer(tokens, mask=mask)

        attended = self.attention_layers[-1](
            tokens, mask=mask, return_weights=self.store_attention
        )
        if self.store_attention:
            attended, att_weights = attended
            self.last_attention = att_weights.detach().cpu().tolist()

        # One unique vector per reaction: the sum of the value vectors of all its
        # compounds (padded slots were already zeroed out by the attention).
        reaction_vectors = torch.sum(attended, dim=1)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
