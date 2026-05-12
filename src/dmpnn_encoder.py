import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter_add


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if num_layers == 1:
            self.net = nn.Linear(in_channels, out_channels)
        else:
            layers = []
            dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(activation())
                    layers.append(nn.Dropout(dropout))

            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network encoder.

    Inputs:
        x:              [num_nodes, in_channels]
        edge_index:     [2, num_edges]
        edge_attr:      [num_edges, edge_channels]
        batch:          [num_nodes]
        rev_edge_index: [num_edges]

    Output:
        graph_embedding: [num_graphs, hidden_channels]
    """

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: int = 300,
        num_layers: int = 3,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool

        # Initial directed bond embedding.
        # Chemprop-style D-MPNN initializes bond messages from source atom + bond features.
        self.W_i = nn.Linear(in_channels + edge_channels, hidden_channels)

        # Message update.
        self.W_h = nn.Linear(hidden_channels, hidden_channels)

        # Atom readout after directed bond message passing.
        self.W_o = nn.Linear(in_channels + hidden_channels, hidden_channels)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch=None,
        rev_edge_index=None,
    ):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if rev_edge_index is None:
            rev_edge_index = self._infer_reverse_edges(edge_index)

        src, dst = edge_index  # edge: src -> dst
        num_edges = edge_index.size(1)
        num_nodes = x.size(0)

        # Initial directed edge state h_0 for each edge src -> dst.
        h0 = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0

        for _ in range(self.num_layers - 1):
            # Sum incoming edge states for each node.
            # incoming_to_node[v] = sum_{u -> v} h_{u->v}
            incoming_to_node = scatter_add(
                h,
                dst,
                dim=0,
                dim_size=num_nodes,
            )

            # For edge src -> dst, candidate message is all messages into src.
            # Then subtract reverse edge dst -> src to avoid immediate backtracking.
            message = incoming_to_node[src] - h[rev_edge_index]

            h = F.relu(h0 + self.W_h(message))
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Aggregate final directed bond states into destination atoms.
        atom_message = scatter_add(
            h,
            dst,
            dim=0,
            dim_size=num_nodes,
        )

        atom_hidden = F.relu(self.W_o(torch.cat([x, atom_message], dim=-1)))
        atom_hidden = F.dropout(atom_hidden, p=self.dropout, training=self.training)

        if self.pool == "mean":
            graph_hidden = global_mean_pool(atom_hidden, batch)
        elif self.pool == "sum":
            graph_hidden = global_add_pool(atom_hidden, batch)
        elif self.pool == "max":
            graph_hidden = global_max_pool(atom_hidden, batch)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        return graph_hidden

    @staticmethod
    def _infer_reverse_edges(edge_index):
        """
        Infers reverse edge indices.

        This assumes every directed edge has a matching reverse edge.
        It is better to precompute this in your dataset for speed.
        """
        src, dst = edge_index
        edge_to_idx = {
            (int(s), int(d)): i
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))
        }

        rev = []
        for s, d in zip(src.tolist(), dst.tolist()):
            key = (int(d), int(s))
            if key not in edge_to_idx:
                raise ValueError(f"Missing reverse edge for edge {s} -> {d}")
            rev.append(edge_to_idx[key])

        return torch.tensor(rev, dtype=torch.long, device=edge_index.device)