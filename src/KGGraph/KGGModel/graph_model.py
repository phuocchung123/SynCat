import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F


class GINConv(MessagePassing):
    """
    GINConv is an extension of the Graph Isomorphism Network (GIN) that incorporates edge information
    by concatenating edge embeddings with node features before aggregation. This class extends the
    MessagePassing class to enable edge feature utilization in message passing.

    Args:
        dataset: The dataset object, which is used to derive the number of edge features or embeddings.
        emb_dim (int): The dimensionality of embeddings for nodes and edges.
        aggr (str): The aggregation scheme to use ('add', 'mean', 'max').

    Reference:
        Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?
        https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add", edge_features=5):
        """
        Initializes the GINConv layer with the specified parameters.

        Args:
            dataset: The dataset object, used for deriving the number of edge features.
            emb_dim (int): The dimensionality of node and edge embeddings.
            aggr (str): The aggregation method to use ('add', 'mean', 'max').
        """
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )

        # Initialize a list of edge MLPs
        # self.edge_mlps = torch.nn.ModuleList(
        #     [
        #         torch.nn.Sequential(
        #             torch.nn.Linear(1, 2 * emb_dim),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(2 * emb_dim, emb_dim),
        #         )
        #         for _ in range(edge_features)  # number of edge features
        #     ]
        # )
        self.edge_mlps = torch.nn.Sequential(torch.nn.Linear(5, emb_dim))
        self.emb_dim = emb_dim
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GINConv layer.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
            edge_attr (Tensor): The edge attributes (features).

        Returns:
            Tensor: The updated node representations.
        """
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 8  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Apply each MLP to its corresponding edge feature slice
        # edge_embeddings = torch.zeros(edge_attr.size(0), self.emb_dim).to(
        #     edge_attr.device
        # )
        # for i, mlp in enumerate(self.edge_mlps):
        #     edge_embeddings += mlp(edge_attr[:, i].view(-1, 1))
        edge_embeddings = self.edge_mlps(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """
        Constructs the messages to a node in a graph.

        Args:
            x_j (Tensor): The features of neighboring nodes.
            edge_attr (Tensor): The features of the edges.

        Returns:
            Tensor: The message to be aggregated.
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Updates node features based on aggregated messages.

        Args:
            aggr_out (Tensor): The aggregated messages for each node.

        Returns:
            Tensor: The updated node features.
        """
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        JK="last",
        drop_ratio=0,
        gnn_type="gin",
        x_features=7,
        edge_features=5,
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # Initialize a list of x MLPs
        # self.x_mlps = torch.nn.ModuleList(
        #     [
        #         torch.nn.Sequential(
        #             torch.nn.Linear(1, 2 * emb_dim),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(2 * emb_dim, emb_dim),
        #         )
        #         for _ in range(x_features)  # number of x features
        #     ]
        # )

        self.x_mlps = torch.nn.Sequential(
            torch.nn.Linear(7, emb_dim),)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add", edge_features=edge_features))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

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

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # Apply each MLP to its corresponding edge feature slice
        # x_embeddings = torch.zeros(x.size(0), self.emb_dim).to(x.device)
        # for i, mlp in enumerate(self.x_mlps):
        #     x_embeddings += mlp(x[:, i].view(-1, 1))
        x_embeddings = self.x_mlps(x)

        h_list = [x_embeddings]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.elu(h), self.drop_ratio, training=self.training
                )  # relu->elu

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
        super_node = self.super_node_rep(node_representation, data.batch)
        # print(super_node.shape)

        return super_node


class GraphModel(torch.nn.Module):
    """
    A GIN model extension that incorporates edge information by concatenation for graph-level prediction tasks.

    This class defines a GNN model which can handle node features, edge features, and graph connectivity to
    produce embeddings for graph-level prediction tasks. It supports different types of GNN layers (e.g., GIN, GCN)
    and aggregation methods for node representations.

    Args:
        num_layer (int): The number of GNN layers.
        emb_dim (int): The dimensionality of node embeddings.
        num_tasks (int): The number of tasks for multi-task learning, typically corresponding to the number of output features.
        JK (str): Choice of how to aggregate node representations across layers. Options are 'last', 'concat', 'max', or 'sum'.
        drop_ratio (float): The dropout rate applied after GNN layers.
        gnn_type (str): The type of GNN layer to use. Options include 'gin', 'gcn', 'graphsage', and 'gat'.
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK,
        drop_ratio,
        gnn_type,
        x_features,
        edge_features,
    ):
        """
        Initializes the GraphModel model with the specified architecture and parameters.

        Args:
            num_layer (int): The number of GNN layers.
            emb_dim (int): The dimensionality of node embeddings.
            num_tasks (int): The number of tasks for multi-task learning.
            JK (str): The aggregation method for node representations.
            drop_ratio (float): The dropout rate after GNN layers.
            gnn_type (str): The type of GNN layer to use.
        """
        super(GraphModel, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(
            num_layer,
            emb_dim,
            JK,
            drop_ratio,
            gnn_type=gnn_type,
            x_features=x_features,
            edge_features=edge_features,
        )

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, (self.emb_dim) // 2),
                torch.nn.ELU(),
                torch.nn.Linear((self.emb_dim) // 2, self.num_tasks),
            )

    def from_pretrained(self, model_file):
        print("Loading pre-trained model from %s" % model_file)
        self.gnn.load_state_dict(torch.load(model_file))

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

    def forward(self, *argv):
        """
        Forward pass of the GraphModel model.

        The method can accept either a data object or the components of the data object as separate parameters.
        It processes the input through GNN layers, aggregates the node representations to form super node
        representations, and applies a final prediction layer.

        Args:
            *argv: Variable length argument list. Can be a single data object or four separate components
                   of the data object (x, edge_index, edge_attr, batch).

        Returns:
            Tensor: The output predictions for each graph in the batch.
        """
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)

        return self.graph_pred_linear(super_rep)
