import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_add_pool


class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats=125,
        edge_in_feats=5,
        depth=3,
        node_hid_feats=300,
        readout_feats=512,
        dr=0.1,
        readout_option=False,
    ):
        super(GNN, self).__init__()

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

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)
        self.readout_option = readout_option
        
    def super_node_rep(self, node_rep, batch):
        """
        Aggregates node representations to form super node representations.

        This method aggregates the node representations of each graph in the batch to form a super node
        representation by taking the representation of the last node for each graph in the batch.

        Args:
            node_rep (Tensor): The node representations of all nodes in the batch.
            batch (Tensor): The batch vector, which maps each node to its respective graph in the batch.

        Returns:
            Tensor: The super node representations for each graph in the batch.
        """
        super_group = []
        for i in range(len(batch)):
            if i != (len(batch) - 1) and batch[i] != batch[i + 1]:
                super_group.append(node_rep[i, :])
            elif i == (len(batch) - 1):
                super_group.append(node_rep[i, :])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep

    def forward(self, data):
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

        readout = self.super_node_rep(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout

