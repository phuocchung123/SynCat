import itertools
import torch
import torch.nn as nn
from gin import GIN
from attention import EncoderLayer


class model(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        out_dim,
        num_layer,
        emb_dim,
        drop_ratio,
    ):
        super(model, self).__init__()
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            num_layer,
            emb_dim,
            drop_ratio,
        )

        self.predict= torch.nn.Linear(emb_dim,out_dim)
        self.attention=EncoderLayer()
        self.atts_reactant=[]
        self.atts_product=[]

    def forward(self, rmols, pmols, r_dummy, p_dummy, device):
        r_graph_feats = torch.stack([self.gnn(rmol) for rmol in rmols])
        p_graph_feats = torch.stack([self.gnn(pmol) for pmol in pmols])

        reaction_vectors=torch.tensor([]).to(device)
        for batch in range(r_graph_feats.shape[1]):
            #The initial reactant's embeddings of each reaction in a batch
            r_graph_feats_1=r_graph_feats[:,batch,:][r_dummy[batch]].to(device)
            num_r= r_graph_feats_1.shape[0]
            # Add pairwise embeddings into the initial reactant's embedding
            for i, j in itertools.combinations(range(num_r), 2):
                pairwise_r=r_graph_feats_1[i] + r_graph_feats_1[j]
                pairwise_r=pairwise_r.reshape(1,-1)
                r_graph_feats_1=torch.cat((r_graph_feats_1,pairwise_r),dim=0)

            # The initial product's emdeddings of each reaction in a batch
            p_graph_feats_1=p_graph_feats[:,batch,:][p_dummy[batch]].to(device)
            num_p= p_graph_feats_1.shape[0]
            #Add pairwise embeddings into the initial product's embeddings
            for i, j in itertools.combinations(range(num_p), 2):
                pairwise_p=p_graph_feats_1[i]+p_graph_feats_1[j]
                pairwise_p=pairwise_p.reshape(1,-1)
                p_graph_feats_1=torch.cat((p_graph_feats_1,pairwise_p),dim=0)

            #attention of reactants
            att_r=self.attention(p_graph_feats_1,r_graph_feats_1)
            att_r=att_r.squeeze(0,1)
            att_reactant = torch.sum(att_r,dim=0)/att_r.shape[0]
            att_reactant=att_reactant.reshape(-1).to(device)

            #attention of products
            att_p = self.attention(r_graph_feats_1, p_graph_feats_1)
            att_p = att_p.squeeze(0,1)
            att_procduct=torch.sum(att_p,dim=0)/att_p.shape[0]
            att_procduct=att_procduct.reshape(-1).to(device)
            
            #reactant's embeddings with attention weights
            reactant_tensor=torch.zeros(1,r_graph_feats_1.shape[1]).to(device)
            for idx in range(r_graph_feats_1.shape[0]):
                reactant_tensor+=att_reactant[idx]*r_graph_feats_1[idx]

            ##product's embeddings with attention weights
            product_tensor=torch.zeros(1,p_graph_feats_1.shape[1]).to(device)
            for idx in range(p_graph_feats_1.shape[0]):
                product_tensor+=att_procduct[idx]*p_graph_feats_1[idx]
            
            # Reaction center
            reaction_center=torch.sub(reactant_tensor,product_tensor)
            reaction_vectors=torch.cat((reaction_vectors,reaction_center),dim=0)
            self.atts_reactant.append(att_reactant.tolist())
            self.atts_product.append(att_procduct.tolist())
        react_graph_feats = torch.sub(r_graph_feats, p_graph_feats)
        out = self.predict(react_graph_feats)
        return out, self.atts_reactant, self.atts_product, reaction_vectors.tolist()

