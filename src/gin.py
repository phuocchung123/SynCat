import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_add_pool


def make_gine_conv(node_hid_feats: int) -> GINEConv:
    """
    Build one GINE convolution with the repository's standard MLP.

    Parameters
    ----------
    node_hid_feats : int
        Hidden feature dimension.

    Returns
    -------
    GINEConv
        A GINE convolution whose edge-conditioned update MLP is
        Linear -> ReLU -> Linear, as used by the shared molecular encoder.
    """
    return GINEConv(
        nn=nn.Sequential(
            nn.Linear(node_hid_feats, node_hid_feats),
            nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats),
        )
    )


def build_gine_conv_stack(depth: int, node_hid_feats: int) -> nn.ModuleList:
    """
    Build `depth` independent GINE convolutions.

    The list is what `GIN` exposes as `gnn_layers`; the hierarchical model uses
    one such stack per pass, so the construction (and therefore the parameter
    initialisation order) is shared instead of duplicated.

    Parameters
    ----------
    depth : int
        Number of GINE layers.
    node_hid_feats : int
        Hidden feature dimension.

    Returns
    -------
    nn.ModuleList
        The GINE stack.
    """
    return nn.ModuleList(
        [make_gine_conv(node_hid_feats) for _ in range(depth)]
    )


def propagate_gine(
    conv_stack: nn.ModuleList,
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_feats: torch.Tensor,
    dropout: nn.Module,
) -> torch.Tensor:
    """
    Run a GINE stack on the current atom states.

    ReLU is applied between convolutions and omitted after the last one; dropout
    is applied after every convolution. The returned tensor keeps the atom
    dimension, which is what lets the hierarchical model re-use the current atom
    states instead of pooling them away as `GIN.forward` does.

    Parameters
    ----------
    conv_stack : nn.ModuleList
        A stack built by `build_gine_conv_stack`.
    node_feats : torch.Tensor
        Current atom states, shape [N, node_hid_feats].
    edge_index : torch.Tensor
        Chemical edges, shape [2, E].
    edge_feats : torch.Tensor
        Projected edge features, shape [E, node_hid_feats].
    dropout : nn.Module
        Dropout module applied after every convolution.

    Returns
    -------
    torch.Tensor
        Updated atom states, shape [N, node_hid_feats].
    """
    depth = len(conv_stack)
    for i in range(depth):
        node_feats = conv_stack[i](node_feats, edge_index, edge_feats)
        if i < depth - 1:
            node_feats = nn.functional.relu(node_feats)
        node_feats = dropout(node_feats)
    return node_feats


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

        self.gnn_layers = build_gine_conv_stack(self.depth, node_hid_feats)

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

        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig)

        node_feats = propagate_gine(
            self.gnn_layers, node_feats, data.edge_index, edge_feats, self.dropout
        )

        readout = global_add_pool(node_feats, batch)

        return readout
