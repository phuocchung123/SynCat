import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
# from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot,zeros


num_atom_type=120 
num_chirality_tag=3

num_bond_type=6
num_bond_direction=3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self,emb_dim, aggr='add'):
        super(GINConv,self).__init__()
        #multi-layer perception
        self.mlp=torch.nn.Sequential(torch.nn.Linear(emb_dim,2*emb_dim),torch.nn.ReLU(),torch.nn.Linear(2*emb_dim,emb_dim))
        # self.edge_embedding1=torch.nn.Embedding(num_bond_type,emb_dim)
        # self.edge_embedding2=torch.nn.Embedding(num_bond_direction,emb_dim)

        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self,x,edge_index,edge_attr):
        #add self loops in edge space
        # edge_index=add_self_loops(edge_index,num_nodes=x.size(0))

        # # #add features corresponding to self-loop edges.
        # # self_loop_attr=torch.zeros(x.size(0),2)
        # # self_loop_attr[:,0] = 4 #bond type for self-loop edge
        # # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # # edge_attr = torch.cat((edge_attr,self_loop_attr),dim=0)

        # # edge_embbeddings=self.edge_embedding1(edge_attr[:,0])+self.edge_embedding2(edge_attr[:,1])
        # self_loop_attr=torch.zeros(x.size(0),9)
        # self_loop_attr[:,4]=1
        # print(edge_attr.shape)
        # print(self_loop_attr.shape)
        # edge_embbeddings = torch.cat((edge_attr,self_loop_attr),dim=0)

        return self.propagate(edge_index, x=x,edge_attr=edge_attr)
    
    def message(self,x_j, edge_attr):
        return x_j + edge_attr
    
    def update(self,aggr_out):
        return self.mlp(aggr_out)
    