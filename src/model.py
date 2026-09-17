import torch
import torch.nn as nn
from gin import GIN


class model(nn.Module):
    """
    Graph-based regression model for reaction-yield prediction.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float,
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
        """
        super(model, self).__init__()
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            num_layer,
            emb_dim,
            drop_ratio,
        )

        # The reaction representation preserves both sides independently:
        # [reactant_embedding || product_embedding].
        self.regressor = torch.nn.Linear(2 * emb_dim, 1)

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
            Predicted yields of shape [batch_size] and the concatenated reaction
            vectors of shape [batch_size, 2 * emb_dim] as a list.
        """
        r_graph_feats = torch.stack([self.gnn(rmol) for rmol in rmols])
        p_graph_feats = torch.stack([self.gnn(pmol) for pmol in pmols])

        reaction_vectors = torch.empty((0, self.regressor.in_features), device=device)
        for batch in range(r_graph_feats.shape[1]):
            # The initial reactant's embeddings of each reaction in a batch
            r_graph_feats_1 = r_graph_feats[:, batch, :][r_dummy[batch]].to(device)
            # The initial product's embeddings of each reaction in a batch
            p_graph_feats_1 = p_graph_feats[:, batch, :][p_dummy[batch]].to(device)

            # Each side is pooled by summing the embeddings of the molecules it
            # actually contains, so no attention weighting is involved.
            reactant_tensor = torch.sum(r_graph_feats_1, dim=0, keepdim=True)
            product_tensor = torch.sum(p_graph_feats_1, dim=0, keepdim=True)

            # Reaction embedding: concatenate the pooled reactant and product
            # representations instead of collapsing them through subtraction.
            reaction_vector = torch.cat((reactant_tensor, product_tensor), dim=1)
            reaction_vectors = torch.cat((reaction_vectors, reaction_vector), dim=0)
        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()
