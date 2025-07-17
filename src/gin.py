import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_add_pool


class GIN(nn.Module):
    """
    Graph Isomorphism Network with edge features.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        depth: int,
        node_hid_feats: int,
        dr: float,
    ) -> None:
        """
        Initialize GIN model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        depth : int
            Number of GIN layers.
        node_hid_feats : int
            Hidden feature dimension for nodes.
        dr : float
            Dropout rate.
        """
        super(GIN, self).__init__()

        self.depth = depth

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats)
        )

        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=torch.nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    )
                )
                for _ in range(self.depth)
            ]
        )

        self.dropout = nn.Dropout(dr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GIN model.

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

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        edge_feats = self.project_edge_feats(edge_feats_orig)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, data.edge_index, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        return readout
