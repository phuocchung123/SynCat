import torch
import torch.nn as nn
from dmpnn import DMPNN



def get_rev_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute rev_edge_index from PyG edge_index.

    Args:
        edge_index: Tensor of shape [2, num_edges]
            edge_index[0] = source nodes
            edge_index[1] = destination nodes

    Returns:
        rev_edge_index: Tensor of shape [num_edges]
            rev_edge_index[e] is the index of the reverse edge of edge e.
    """

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {edge_index.shape}"
        )

    source = edge_index[0]
    destination = edge_index[1]

    edge_to_index = {}

    for i, (src, dst) in enumerate(zip(source.tolist(), destination.tolist())):
        edge_to_index[(src, dst)] = i

    rev_edge_index = []

    for i, (src, dst) in enumerate(zip(source.tolist(), destination.tolist())):
        reverse_edge = (dst, src)

        if reverse_edge not in edge_to_index:
            raise ValueError(
                f"Missing reverse edge for edge {i}: {src} -> {dst}"
            )

        rev_edge_index.append(edge_to_index[reverse_edge])

    return torch.tensor(
        rev_edge_index,
        dtype=torch.long,
        device=edge_index.device,
    )

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

        self.gnn = DMPNN(
            in_channels=node_in_feats,
            edge_channels=edge_in_feats,
            hidden_channels=node_hid_feats,
            num_layers=depth,
            out_channels=node_hid_feats,
            dropout=dr,
            pool="sum",
            ffn_num_layers=1,
        )

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
        edge_index = data.edge_index
        rev_edge_index = get_rev_edge_index(edge_index)

        self.gnn(node_feats_orig,
        edge_index = edge_index,
        edge_attr= edge_feats_orig,
        batch=batch,
        rev_edge_index= rev_edge_index)

