import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool


def _reverse_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Index of the reverse of every directed edge.

    The molecule graphs store each bond as both (u, v) and (v, u), so for the
    directed edge at position `i` there is an opposite edge at `rev[i]`. Entries
    whose reverse is missing (an isolated one-way edge, which should not occur
    in these graphs) are set to -1.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    num_nodes : int
        Number of nodes in the (batched) graph, used to build unique keys.

    Returns
    -------
    torch.Tensor
        Long tensor of shape [num_edges] with the reverse-edge positions.
    """
    src, dst = edge_index
    key = src * num_nodes + dst
    reverse_key = dst * num_nodes + src

    order = torch.argsort(key)
    sorted_key = key[order]
    pos = torch.searchsorted(sorted_key, reverse_key).clamp(max=key.numel() - 1)

    rev = torch.full_like(key, -1)
    found = sorted_key[pos] == reverse_key
    rev[found] = order[pos[found]]
    return rev


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network with edge features.

    Messages live on directed bonds rather than on atoms: for every directed
    edge v -> w the hidden state h_vw is updated from the incoming messages of
    the neighbours of v, excluding the reverse edge w -> v, so that information
    never flows straight back along the bond it came from. After `depth` steps
    the atom vectors pool the incoming messages and are summed per graph.

    Same interface and output shape as `gin.GIN`, so it is interchangeable with
    the other encoders inside the reaction model.
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
        Initialize DMPNN model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        depth : int
            Number of directed message passing steps.
        node_hid_feats : int
            Hidden feature dimension for nodes and directed edges.
        dr : float
            Dropout rate.
        """
        super(DMPNN, self).__init__()

        self.depth = depth

        # Initial directed-edge state from the source atom and the bond.
        self.W_i = nn.Linear(node_in_feats + edge_in_feats, node_hid_feats)
        # Initial atom state, added to every message update.
        self.W_o = nn.Linear(node_in_feats, node_hid_feats)
        # Message transforms, one per propagation step.
        self.W_h = nn.ModuleList(
            [nn.Linear(node_hid_feats, node_hid_feats) for _ in range(self.depth)]
        )
        # Final atom state from the atom and its incoming message sum.
        self.W_f = nn.Linear(node_in_feats + node_hid_feats, node_hid_feats)

        self.dropout = nn.Dropout(dr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DMPNN model.

        Parameters
        ----------
        data : torch.Tensor
            Batch of graph data objects with x, edge_attr, edge_index, and batch attributes.

        Returns
        -------
        torch.Tensor
            Readout vector after global pooling, shape [batch_size, node_hid_feats].
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        num_nodes = x.shape[0]
        src, dst = edge_index

        node_feats_init = torch.relu(self.W_o(x))
        edge_feats = torch.relu(self.W_i(torch.cat((x[src], edge_attr), dim=-1)))

        rev = _reverse_edge_index(edge_index, num_nodes)

        for i in range(self.depth):
            # Sum of all directed messages that arrive at a node.
            incoming = x.new_zeros(num_nodes, edge_feats.shape[-1])
            incoming = incoming.index_add(0, dst, edge_feats)
            # Exclude the reverse edge, so a message is not sent back along
            # the bond it just came from. `edge_feats[rev]` with the missing
            # entries zeroed keeps this differentiable.
            valid = (rev >= 0).unsqueeze(-1).to(edge_feats.dtype)
            excluded = edge_feats[rev.clamp(min=0)] * valid

            messages = incoming[src] - excluded
            edge_feats = torch.relu(
                node_feats_init[src] + self.W_h[i](messages)
            )
            edge_feats = self.dropout(edge_feats)

        # Final atom state: its own features plus the messages arriving at it.
        incoming = x.new_zeros(num_nodes, edge_feats.shape[-1])
        incoming = incoming.index_add(0, dst, edge_feats)
        node_feats = torch.relu(
            self.W_f(torch.cat((x, incoming), dim=-1))
        )

        readout = global_add_pool(node_feats, batch)

        return readout
