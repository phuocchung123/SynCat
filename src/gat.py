import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import global_add_pool


class GAT(nn.Module):
    """
    Graph Attention Network with edge features.

    Same interface and output shape as `gin.GIN`, so the two encoders are
    interchangeable inside the reaction model: every compound graph is encoded
    into one node_hid_feats vector per graph.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        depth: int,
        node_hid_feats: int,
        dr: float,
        heads: int = 4,
    ) -> None:
        """
        Initialize GAT model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        depth : int
            Number of GAT layers.
        node_hid_feats : int
            Hidden feature dimension for nodes.
        dr : float
            Dropout rate.
        heads : int
            Number of attention heads per layer (default is 4). The heads are
            averaged (`concat=False`), so the hidden dimension stays
            `node_hid_feats` for any number of heads.
        """
        super(GAT, self).__init__()

        self.depth = depth
        self.heads = heads

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats)
        )

        self.gnn_layers = nn.ModuleList(
            [
                GATConv(
                    node_hid_feats,
                    node_hid_feats,
                    heads=self.heads,
                    concat=False,
                    edge_dim=node_hid_feats,
                    dropout=dr,
                )
                for _ in range(self.depth)
            ]
        )

        self.dropout = nn.Dropout(dr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GAT model.

        Parameters
        ----------
        data : torch.Tensor
            Batch of graph data objects with x, edge_attr, edge_index, and batch attributes.

        Returns
        -------
        torch.Tensor
            Readout vector after global pooling, shape [batch_size, node_hid_feats].
        """
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch

        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, data.edge_index, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        return readout
