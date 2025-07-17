import itertools
import torch
import torch.nn as nn
from gin import GIN
from attention import SingleHeadAttention


class model(nn.Module):
    """
    Graph-based classification model with cross attention.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        out_dim: int,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        out_dim : int
            Output dimension for prediction.
        num_layer : int
            Number of GNN layers.
        emb_dim : int
            Embedding dimension.
        drop_ratio : float
            Dropout rate.
        """
        super(model, self).__init__()
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            num_layer,
            emb_dim,
            drop_ratio,
        )

        self.predict = torch.nn.Linear(emb_dim, out_dim)
        self.attention = SingleHeadAttention(emb_dim)
        self.atts_reactant = []
        self.atts_product = []

    def forward(
        self,
        rmols: list,
        pmols: list,
        r_dummy: list,
        p_dummy: list,
        device: torch.device,
    ) -> tuple:
        """
        Forward pass of the model.

        Parameters
        ----------
        rmols : list
            List of reactant graph data objects.
        pmols : list
            List of product graph data objects.
        r_dummy : list
            List of masks/indices for reactants in batch.
        p_dummy : list
            List of masks/indices for products in batch.
        device : torch.device
            Device to run the computations.

        Returns
        -------
        tuple
            Output logits, reactant attentions, product attentions, and reaction vectors as list.
        """
        r_graph_feats = torch.stack([self.gnn(rmol) for rmol in rmols])
        p_graph_feats = torch.stack([self.gnn(pmol) for pmol in pmols])

        reaction_vectors = torch.tensor([]).to(device)
        for batch in range(r_graph_feats.shape[1]):
            # The initial reactant's embeddings of each reaction in a batch
            r_graph_feats_1 = r_graph_feats[:, batch, :][r_dummy[batch]].to(device)
            num_r = r_graph_feats_1.shape[0]
            # Add pairwise embeddings into the initial reactant's embedding
            for i, j in itertools.combinations(range(num_r), 2):
                pairwise_r = r_graph_feats_1[i] + r_graph_feats_1[j]
                pairwise_r = pairwise_r.reshape(1, -1)
                r_graph_feats_1 = torch.cat((r_graph_feats_1, pairwise_r), dim=0)

            # The initial product's emdeddings of each reaction in a batch
            p_graph_feats_1 = p_graph_feats[:, batch, :][p_dummy[batch]].to(device)
            num_p = p_graph_feats_1.shape[0]
            # Add pairwise embeddings into the initial product's embeddings
            for i, j in itertools.combinations(range(num_p), 2):
                pairwise_p = p_graph_feats_1[i] + p_graph_feats_1[j]
                pairwise_p = pairwise_p.reshape(1, -1)
                p_graph_feats_1 = torch.cat((p_graph_feats_1, pairwise_p), dim=0)

            # attention of reactants
            att_r = self.attention(p_graph_feats_1, r_graph_feats_1)
            att_reactant = torch.sum(att_r, dim=0) / att_r.shape[0]
            att_reactant = att_reactant.reshape(-1).to(device)

            # attention of products
            att_p = self.attention(r_graph_feats_1, p_graph_feats_1)
            att_procduct = torch.sum(att_p, dim=0) / att_p.shape[0]
            att_procduct = att_procduct.reshape(-1).to(device)

            # reactants embeddings with attention weights
            reactant_tensor = torch.zeros(1, r_graph_feats_1.shape[1]).to(device)
            for idx in range(r_graph_feats_1.shape[0]):
                reactant_tensor += att_reactant[idx] * r_graph_feats_1[idx]

            # products embeddings with attention weights
            product_tensor = torch.zeros(1, p_graph_feats_1.shape[1]).to(device)
            for idx in range(p_graph_feats_1.shape[0]):
                product_tensor += att_procduct[idx] * p_graph_feats_1[idx]

            # Reaction center
            reaction_center = torch.sub(reactant_tensor, product_tensor)
            reaction_vectors = torch.cat((reaction_vectors, reaction_center), dim=0)
            self.atts_reactant.append(att_reactant.tolist())
            self.atts_product.append(att_procduct.tolist())
        out = self.predict(reaction_vectors)
        return out, self.atts_reactant, self.atts_product, reaction_vectors.tolist()
