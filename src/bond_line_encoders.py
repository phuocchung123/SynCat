"""Molecular encoders for the bond-line graph representation.

Every encoder consumes a single ``BondLineData``/``Batch`` of ``BondLineData``
and returns one embedding per graph, of shape [num_graphs, D]. They share the
same input projection (``Linear(line_node_dim, D) -> ReLU``) and use molecular
**mean** pooling, unlike the atom-graph encoders which sum-pool.

The atom-graph GIN/GAT/DMPNN in ``gin.py``, ``gat.py`` and ``dmpnn.py`` are kept
unchanged; these are separate classes for the transformed topology-only graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv, GINConv
from torch_geometric.nn.pool import global_mean_pool


def _batch_and_count(data):
    """Return the graph-assignment vector and the number of graphs."""
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = data.x.new_zeros(data.x.shape[0], dtype=torch.long)
    num_graphs = int(getattr(data, "num_graphs", 0))
    if num_graphs <= 0:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    return batch, num_graphs


class BondLineGIN(nn.Module):
    """
    Standard GIN on the bond-line graph (topology only).

    Uses ``GINConv`` with a two-layer MLP, ``train_eps=True`` and the default
    sum aggregation. No chemical-bond features and no explicit self-loops are
    used: the representation already carries bond features in node features.
    """

    def __init__(
        self,
        line_node_dim: int,
        depth: int,
        node_hid_feats: int,
        dr: float,
    ) -> None:
        """
        Initialize BondLineGIN.

        Parameters
        ----------
        line_node_dim : int
            Width of the transformed node features.
        depth : int
            Number of GIN layers.
        node_hid_feats : int
            Hidden/embedding dimension D.
        dr : float
            Dropout rate.
        """
        super(BondLineGIN, self).__init__()

        self.depth = depth
        self.project_node_feats = nn.Sequential(
            nn.Linear(line_node_dim, node_hid_feats), nn.ReLU()
        )
        self.gnn_layers = nn.ModuleList(
            [
                GINConv(
                    nn=nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    ),
                    train_eps=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.dropout = nn.Dropout(dr)

    def forward(self, data) -> torch.Tensor:
        """Encode a (batched) bond-line graph into [num_graphs, D]."""
        node_feats = self.project_node_feats(data.x)
        edge_index = data.edge_index

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, edge_index)
            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)
            node_feats = self.dropout(node_feats)

        batch, num_graphs = _batch_and_count(data)
        return global_mean_pool(node_feats, batch, size=num_graphs)


class BondLineGAT(nn.Module):
    """
    Standard GAT on the bond-line graph (topology only).

    Uses ``GATConv`` with ``concat=True`` (unlike the atom-graph GAT, whose
    heads are averaged) and no edge features. ``gat_heads`` graph-attention
    heads are concatenated, so the output width is ``D // gat_heads * gat_heads
    == D`` and ``D`` must be divisible by ``gat_heads``.
    """

    def __init__(
        self,
        line_node_dim: int,
        depth: int,
        node_hid_feats: int,
        dr: float,
        gat_heads: int = 4,
    ) -> None:
        """
        Initialize BondLineGAT.

        Parameters
        ----------
        line_node_dim : int
            Width of the transformed node features.
        depth : int
            Number of GAT layers.
        node_hid_feats : int
            Hidden/embedding dimension D.
        dr : float
            Dropout rate.
        gat_heads : int
            Number of concatenated graph-attention heads (default is 4).
        """
        super(BondLineGAT, self).__init__()

        if gat_heads < 1 or node_hid_feats % gat_heads != 0:
            raise ValueError(
                "gat_heads (%d) must be a positive divisor of emb_dim (%d)"
                % (gat_heads, node_hid_feats)
            )

        self.depth = depth
        self.gat_heads = gat_heads
        self.project_node_feats = nn.Sequential(
            nn.Linear(line_node_dim, node_hid_feats), nn.ReLU()
        )
        self.gnn_layers = nn.ModuleList(
            [
                GATConv(
                    node_hid_feats,
                    node_hid_feats // self.gat_heads,
                    heads=self.gat_heads,
                    concat=True,
                    add_self_loops=True,
                    dropout=dr,
                )
                for _ in range(self.depth)
            ]
        )
        self.dropout = nn.Dropout(dr)

    def forward(self, data) -> torch.Tensor:
        """Encode a (batched) bond-line graph into [num_graphs, D]."""
        node_feats = self.project_node_feats(data.x)
        edge_index = data.edge_index

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, edge_index)
            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)
            node_feats = self.dropout(node_feats)

        batch, num_graphs = _batch_and_count(data)
        return global_mean_pool(node_feats, batch, size=num_graphs)


class BondLineDMPNN(nn.Module):
    """
    D-MPNN adapted to the bond-line graph.

    Directed messages live on the directed line-graph edges. With ``K = depth``
    (``K >= 1``) the directed states are initialized once and then updated
    ``K - 1`` times:

        s0      = relu(W_in(z[src]))
        state   = s0
        repeat K-1 times:
            incoming = zeros[num_line_nodes].index_add(0, dst, state)
            message  = incoming[src] - state[rev_edge]   # exclude the reverse
            state    = dropout(relu(s0 + W_message(message)))
        incoming = zeros[num_line_nodes].index_add(0, dst, state)
        u        = readout_mlp(concat(z, incoming))

    ``W_message`` is bias-free and shared across all updates, and is only
    allocated when ``K > 1``. The final transformed-node embeddings ``u`` are
    mean-pooled by molecule. Isolated transformed nodes (E == 0) still use
    their own projection ``z`` through the readout.
    """

    def __init__(
        self,
        line_node_dim: int,
        depth: int,
        node_hid_feats: int,
        dr: float,
    ) -> None:
        """
        Initialize BondLineDMPNN.

        Parameters
        ----------
        line_node_dim : int
            Width of the transformed node features.
        depth : int
            Number of directed message states K (>= 1).
        node_hid_feats : int
            Hidden/embedding dimension D.
        dr : float
            Dropout rate.
        """
        super(BondLineDMPNN, self).__init__()

        if depth < 1:
            raise ValueError("BondLineDMPNN requires depth >= 1, got %d" % depth)

        self.depth = depth
        self.project_node_feats = nn.Sequential(
            nn.Linear(line_node_dim, node_hid_feats), nn.ReLU()
        )
        self.W_in = nn.Linear(node_hid_feats, node_hid_feats)
        # Allocated only when there is at least one update (K > 1).
        self.W_message = (
            nn.Linear(node_hid_feats, node_hid_feats, bias=False)
            if self.depth > 1
            else None
        )
        self.readout_mlp = nn.Sequential(
            nn.Linear(2 * node_hid_feats, node_hid_feats),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(node_hid_feats, node_hid_feats),
        )
        self.dropout = nn.Dropout(dr)

    def forward(self, data) -> torch.Tensor:
        """Encode a (batched) bond-line graph into [num_graphs, D]."""
        z = self.project_node_feats(data.x)
        num_line_nodes = z.shape[0]
        src, dst = data.edge_index

        state = torch.relu(self.W_in(z[src]))
        s0 = state

        for _ in range(self.depth - 1):
            incoming = z.new_zeros((num_line_nodes, z.shape[1]))
            incoming = incoming.index_add(0, dst, state)
            message = incoming[src] - state[data.rev_edge]
            state = self.dropout(torch.relu(s0 + self.W_message(message)))

        incoming = z.new_zeros((num_line_nodes, z.shape[1]))
        incoming = incoming.index_add(0, dst, state)
        u = self.readout_mlp(torch.cat((z, incoming), dim=-1))

        batch, num_graphs = _batch_and_count(data)
        return global_mean_pool(u, batch, size=num_graphs)
