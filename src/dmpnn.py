import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN).

    The GINE encoder passes messages between atoms; a D-MPNN passes them
    between directed bonds instead. Every bond is read in both directions (the
    graphs already carry the reverse edge of every bond), so a bond state
    `u -> v` is built from atom `u` and the bond features, and is then updated
    from the states of the bonds entering `u`, minus the state of the bond
    `v -> u` so that a message never immediately backtracks. The final bond
    states are summed into their destination atoms together with the atom
    features, and the atoms are pooled into one vector per graph.

    The pooling is a sum, as in the GINE encoder, so that swapping the encoder
    changes only the message passing and not the readout.
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
        Initialize the D-MPNN model.

        Parameters
        ----------
        node_in_feats : int
            Input feature dimension for nodes.
        edge_in_feats : int
            Input feature dimension for edges.
        depth : int
            Number of message passing steps: the initial bond states plus
            `depth - 1` update rounds. With the default of 3 this reads the
            neighbourhood within three bonds, like the three GINE layers.
        node_hid_feats : int
            Hidden feature dimension, used for atoms and for bond states.
        dr : float
            Dropout rate.
        """
        super(DMPNN, self).__init__()

        self.depth = depth
        self.node_hid_feats = node_hid_feats

        # Initial directed bond state, from the source atom and the bond features.
        self.project_bond_feats = nn.Linear(
            node_in_feats + edge_in_feats, node_hid_feats
        )

        # Update of a bond state from the bond states that enter its source atom.
        self.update_bond_feats = nn.Linear(node_hid_feats, node_hid_feats)

        # Atom readout, from the atom features and the bond states entering it.
        self.project_atom_feats = nn.Linear(
            node_in_feats + node_hid_feats, node_hid_feats
        )

        self.dropout = nn.Dropout(dr)

    @staticmethod
    def _scatter_sum(
        values: torch.Tensor, index: torch.Tensor, size: int
    ) -> torch.Tensor:
        """
        Sum the rows of `values` into `size` rows according to `index`.

        Parameters
        ----------
        values : torch.Tensor
            Rows to add, of shape [num_values, feats].
        index : torch.Tensor
            Target row of every value, of shape [num_values].
        size : int
            Number of rows in the result.

        Returns
        -------
        torch.Tensor
            Summed rows of shape [size, feats].
        """
        out = values.new_zeros(size, values.size(1))
        return out.index_add_(0, index, values)

    @staticmethod
    def reverse_edge_index(
        edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """
        Map every directed edge to the index of its reverse edge.

        The molecules are preprocessed with one edge per bond direction, so
        every `u -> v` has a matching `v -> u`. Batched graphs number their
        nodes consecutively, which keeps the (source, destination) pairs
        globally unique and makes the lookup a search over sorted keys.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        num_nodes : int
            Number of nodes in the (batched) graphs.

        Returns
        -------
        torch.Tensor
            Reverse edge indices of shape [num_edges].
        """
        src, dst = edge_index
        key = src * num_nodes + dst
        reverse_key = dst * num_nodes + src

        order = torch.argsort(key)
        sorted_key = key[order]
        position = torch.searchsorted(sorted_key, reverse_key)
        position = position.clamp(max=sorted_key.numel() - 1)
        reverse = order[position]

        if not torch.equal(key[reverse], reverse_key):
            raise ValueError(
                "every directed edge needs its reverse edge, but at least one "
                "is missing; the graphs must contain both bond directions"
            )

        return reverse

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the D-MPNN model.

        Parameters
        ----------
        data : torch.Tensor
            Batch of graph data objects with x, edge_attr, edge_index, and batch attributes.

        Returns
        -------
        torch.Tensor
            Readout vector after global pooling, shape [batch_size, node_hid_feats].
        """
        node_feats = data.x
        batch = data.batch
        src, dst = data.edge_index
        num_nodes = node_feats.size(0)
        num_edges = data.edge_index.size(1)

        if num_edges > 0:
            # Initial bond state of every directed edge u -> v.
            bond_feats = F.relu(
                self.project_bond_feats(
                    torch.cat((node_feats[src], data.edge_attr), dim=-1)
                )
            )
            bond_feats_init = bond_feats
            reverse = self.reverse_edge_index(data.edge_index, num_nodes)

            for _ in range(self.depth - 1):
                # For the bond u -> v, sum the states entering u and subtract
                # the state of v -> u, so that a message never backtracks.
                into_atoms = self._scatter_sum(bond_feats, dst, num_nodes)
                message = into_atoms[src] - bond_feats[reverse]
                bond_feats = F.relu(bond_feats_init + self.update_bond_feats(message))
                bond_feats = self.dropout(bond_feats)

            atom_messages = self._scatter_sum(bond_feats, dst, num_nodes)
        else:
            # Graphs without bonds (single atoms, dummy padding graphs).
            atom_messages = node_feats.new_zeros(num_nodes, self.node_hid_feats)

        atom_feats = F.relu(
            self.project_atom_feats(torch.cat((node_feats, atom_messages), dim=-1))
        )
        atom_feats = self.dropout(atom_feats)

        if batch is None:
            return atom_feats.sum(dim=0, keepdim=True)

        return global_add_pool(atom_feats, batch)
