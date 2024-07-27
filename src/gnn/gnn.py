import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
# from torch_scatter import scatter_ad
from torch_geometric.nn.inits import glorot, zeros

from gnn.gat import GATConv
from gnn.gcn import GCNConv
from gnn.gin import GINConv
from gnn.graphsage import GraphSAGEConv

num_atom_type = 118 #including the extra mask tokens
num_charge = 4
num_degree=6
num_hybridization=5
num_h=4
num_valence=6
num_h_bond=2
num_chirality=2
num_ringsize=6
num_aromatic=2

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

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
    def __init__(self,num_layer,emb_dim,node_in_feats, edge_in_feats,JK ='sum',drop_ratio=0,gnn_type='gin'):
        super(GNN,self).__init__()
        self.num_layer = num_layer
        self.drop_ratio=drop_ratio
        self.JK=JK

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        
        # self.x_embedding1= torch.nn.Embedding(num_atom_type, emb_dim)
        # self.x_embedding2= torch.nn.Embedding(num_chirality_tag, emb_dim)

        # torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns= torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.gnns.append(GINConv(emb_dim, aggr='add'))
            elif gnn_type == 'gcn':
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))


        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.project_node_feats=torch.nn.Sequential(torch.nn.Linear(node_in_feats,emb_dim),torch.nn.ReLU())
        self.project_edge_feats=torch.nn.Sequential(torch.nn.Linear(edge_in_feats,emb_dim),torch.nn.ReLU())


    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) ==1:
            data=argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data. edge_attr
        else:
            raise ValueError('unmatched number of arguments.')
        
        # x= self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        x=self.project_node_feats(x)
        edge_attr=self.project_edge_feats(edge_attr)


        h_list=[x]
        for layer in range(self.num_layer):
            h=self.gnns[layer](h_list[layer],edge_index,edge_attr)
            h=self.batch_norms[layer](h)

            if layer == self.num_layer -1:
                #remove relu for the last layer
                h = F.dropout(h,self.drop_ratio, training = self.training)

            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training =self.training)

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK =='concat':
            node_representation= torch.cat(h_list, dim=1)
        elif self.JK =='last':
            node_representation= h_list[-1]
        elif self.JK == 'max':
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0),dim=0)[0]
        elif self.JK =='sum':
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation= torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        
        return node_representation
    
