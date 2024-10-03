# Import necessary modules and functions
import sys
from pathlib import Path
import torch
import math
import random
from rdkit import Chem
from typing import Tuple, List
import numpy as np

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGDecompose.brics_decompose import BRCISDecomposition
from KGGraph.KGGDecompose.jin_decompose import TreeDecomposition
from KGGraph.KGGDecompose.motif_decompose import MotifDecomposition
from KGGraph.KGGDecompose.smotif_decompose import SMotifDecomposition
from KGGraph.KGGChem.bond_features import bond_type_feature


# allowable edge features
allowable_features = {
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_inring": [None, False, True],
}


class EdgeFeature:
    def __init__(self, mol: Chem.Mol, decompose_type: str):
        """
        Initializes the class with the given molecule and sets up the decompose type for further processing.

        Args:
            mol (Chem.Mol): The input molecule for the class.
            decompose_type (str): The type of decomposition to use, e.g., 'motif', 'brics', 'jin', 'smotif'.
        """
        self.mol = mol
        self.decompose_type = decompose_type
        self._cliques, self._clique_edges = None, None
        self._num_edge_features = 5

    def decompose(self) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        if self.decompose_type == "motif":
            return MotifDecomposition.defragment(self.mol)
        elif self.decompose_type == "brics":
            return BRCISDecomposition.defragment(self.mol)
        elif self.decompose_type == "jin":
            return TreeDecomposition.defragment(self.mol)
        elif self.decompose_type == "smotif":
            return SMotifDecomposition.defragment(self.mol)
        else:
            raise ValueError(
                f"Unknown decomposition type: {self.decompose_type}. It should be motif, brics, jin or smotif."
            )

    @property
    def cliques(self) -> List[List[int]]:
        if self._cliques is None:
            self._cliques, self._clique_edges = self.decompose()
        return self._cliques

    @property
    def clique_edges(self) -> List[Tuple[int, int]]:
        if self._clique_edges is None:
            self._cliques, self._clique_edges = self.decompose()
        return self._clique_edges

    @property
    def num_edge_features(self) -> int:
        return self._num_edge_features

    @property
    def num_motif(self) -> int:
        return len(self.cliques)

    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def num_bonds(self) -> int:
        return self.mol.GetNumBonds()

    @staticmethod
    def get_edge_node_feature(
        mol: Chem.Mol, num_edge_features: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the edge features for the molecule.

        Returns:
            edge_attr_list: A tensor of edge attributes.
            edges_index_list: A tensor of edge indices.
        """
        if len(mol.GetAtoms()) > 0:
            # Initialize lists to store edge attributes and indices
            edge_attr_list = []
            edges_index_list = []

            # Iterate over all bonds in the molecule
            for bond in mol.GetBonds():
                # Combine all features into a single list
                bond_type = bond.GetBondType()
                if bond_type in allowable_features["possible_bonds"]:
                    bond_type_int = [
                        allowable_features["possible_bonds"].index(bond.GetBondType())
                    ]
                else:
                    bond_type_int = [4]  # 4 is the index for OTHER bond type

                combined_features = (
                    bond_type_int
                    + [
                        allowable_features["possible_bond_inring"].index(
                            bond.IsInRing()
                        )
                    ]
                    + bond_type_feature(bond)
                )

                # Get the indices of the atoms involved in the bond
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # Add the indices and features to the respective lists
                edges_index_list.extend([(i, j), (j, i)])
                edge_attr_list.extend([combined_features, combined_features])
            # Convert the lists to tensors
            edge_attr_node = torch.tensor(np.array(edge_attr_list), dtype=torch.float32)
            edges_index_node = torch.tensor(
                np.array(edges_index_list).T, dtype=torch.long
            )
        else:
            edges_index_node = torch.empty((2, 0), dtype=torch.long)
            edge_attr_node = torch.empty((0, num_edge_features), dtype=torch.float32)
        return edge_attr_node, edges_index_node

    @staticmethod
    def get_edge_index(
        edge_index_node: torch.Tensor,
        num_motif: int,
        cliques: List[List[int]],
        num_atoms: int,
        clique_edges: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct edge indices for a molecule with motif supernodes.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            torch.Tensor: Tensor representing the edge indices including motif supernodes.
        """
        # If there are motifs, create edges between atoms and motifs
        if num_motif > 0:
            # Initialize the motif_edge_index list
            motif_edge_index = []
            for k, motif_nodes in enumerate(cliques):
                motif_edge_index.extend([[i, num_atoms + k] for i in motif_nodes])

            motif_edge_index = torch.tensor(
                np.array(motif_edge_index).T, dtype=torch.long
            ).to(edge_index_node.device)

            # Create edges between motif and a supernode
            super_edge_index = [
                [num_atoms + i, num_atoms + num_motif] for i in range(num_motif)
            ]
            super_edge_index = torch.tensor(
                np.array(super_edge_index).T, dtype=torch.long
            ).to(edge_index_node.device)

            # Concatenate all edges
            edge_index = torch.cat(
                (edge_index_node, motif_edge_index, super_edge_index), dim=1
            )
        else:
            motif_edge_index = torch.empty((0, 0), dtype=torch.long)
            # Create edges between atoms and the supernode
            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(
                np.array(super_edge_index).T, dtype=torch.long
            ).to(edge_index_node.device)
            edge_index = torch.cat((edge_index_node, super_edge_index), dim=1)

        return motif_edge_index, edge_index

    @staticmethod
    def get_edge_attr(
        edge_attr_node: torch.Tensor,
        motif_edge_index: torch.Tensor,
        num_motif: int,
        clique_edges: List[Tuple[int, int]],
        num_edge_features: int,
        num_atoms: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate edge attributes for a molecule, including attributes for edges connecting
        atoms, motifs, and a super supernode.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple containing tensors for motif edge attributes, super edge attributes,
            and the concatenated edge attributes for the entire molecular graph.
        """
        if num_motif > 0:
            # Initialize motif edge attributes
            motif_edge_attr = torch.zeros((motif_edge_index.size(1), num_edge_features))
            motif_edge_attr[:, 0] = (
                6  # Set bond type for the edge between atoms and motif,
            )

            # Initialize super edge attributes
            super_edge_attr = torch.zeros((num_motif, num_edge_features))
            super_edge_attr[:, 0] = 5
            motif_edge_attr = motif_edge_attr.to(edge_attr_node.dtype).to(
                edge_attr_node.device
            )
            super_edge_attr = super_edge_attr.to(edge_attr_node.dtype).to(
                edge_attr_node.device
            )
            # Concatenate edge attributes for the entire graph
            edge_attr = torch.cat(
                (edge_attr_node, motif_edge_attr, super_edge_attr), dim=0
            )

        else:
            motif_edge_attr = torch.empty((0, 0))
            # Initialize super edge attributes when there are no motifs
            super_edge_attr = torch.zeros((num_atoms, num_edge_features))
            super_edge_attr[:, 0] = (
                5  # Set bond type for the edge between nodes and supernode,
            )
            super_edge_attr = super_edge_attr.to(edge_attr_node.dtype).to(
                edge_attr_node.device
            )

            # Concatenate edge attributes for the entire graph
            edge_attr = torch.cat((edge_attr_node, super_edge_attr), dim=0)

        return edge_attr

    @staticmethod
    def masked_edge_feature(
        edge_index_node: torch.Tensor,
        edge_attr_node: torch.Tensor,
        num_bonds: int,
        num_edge_features: int,
        mask_edge_ratio: float,
        fix_ratio: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask a portion of the edges in the graph by setting their attributes to zero.

        Args:
            edge_index_node (torch.Tensor): The edge index tensor for the node graph.
            edge_attr_node (torch.Tensor): The edge attribute tensor for the node graph.
            num_bonds (int): The number of bonds in the graph.
            fix_ratio (bool): Whether to use a fixed ratio for masking or a random ratio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The masked edge attribute tensor and masked edge index tensor.
        """
        # Calculate the number of edges to be masked
        if fix_ratio:
            num_masked_edges = max(0, math.floor(mask_edge_ratio * num_bonds))
        else:
            num_masked_edges = random.randint(
                0, math.floor(mask_edge_ratio * num_bonds)
            )

        # Sample the indices of edges to be masked
        masked_edges_single = random.sample(list(range(num_bonds)), num_masked_edges)
        masked_edges = [2 * i for i in masked_edges_single] + [
            2 * i + 1 for i in masked_edges_single
        ]

        # Initialize the masked edge index and attribute tensors
        edge_index_masked = torch.zeros(
            (2, 2 * (num_bonds - num_masked_edges)), dtype=torch.long
        )
        edge_attr_masked = torch.zeros(
            (2 * (num_bonds - num_masked_edges), num_edge_features), dtype=torch.float32
        )
        count = 0

        # Iterate over all edges and copy those not to be masked to the masked tensors
        for bond_idx in range(2 * num_bonds):
            if bond_idx not in masked_edges:
                edge_index_masked[:, count] = edge_index_node[:, bond_idx]
                edge_attr_masked[count, :] = edge_attr_node[bond_idx, :]
                count += 1

        return edge_attr_masked, edge_index_masked


def edge_feature(mol, decompose_type, mask_edge, mask_edge_ratio, fix_ratio):
    """
    This function generates edge features for a given molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        decompose_type (str): The type of decomposition to use.
        mask_edge (bool): Whether to mask edges or not.
        fix_ratio (bool): Whether to use a fixed ratio or a random ratio for masking.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The edge attribute tensor for the node graph,
            the edge index tensor for the node graph,
            the edge index tensor for the graph,
            the edge attribute tensor for the graph.
    """
    # Create an instance of the EdgeFeature class
    obj = EdgeFeature(mol, decompose_type=decompose_type)

    # Get the edge attribute tensor and edge index tensor for the node graph
    edge_attr_node, edge_index_node = obj.get_edge_node_feature(
        mol, obj.num_edge_features
    )

    # If masking is not enabled
    if not mask_edge:
        # Get the edge index tensor for the graph and the edge attribute tensor for the graph
        motif_edge_index, edge_index = obj.get_edge_index(
            edge_index_node, obj.num_motif, obj.cliques, obj.num_atoms, obj.clique_edges
        )
        edge_attr = obj.get_edge_attr(
            edge_attr_node,
            motif_edge_index,
            obj.num_motif,
            obj.clique_edges,
            obj.num_edge_features,
            obj.num_atoms,
        )
    # If masking is enabled
    else:
        # Get the masked edge attribute tensor and masked edge index tensor
        edge_attr_masked, edge_index_masked = obj.masked_edge_feature(
            edge_index_node,
            edge_attr_node,
            obj.num_bonds,
            obj.num_edge_features,
            mask_edge_ratio,
            fix_ratio=fix_ratio,
        )
        # Get the edge index tensor for the graph and the edge attribute tensor for the graph
        motif_edge_index, edge_index = obj.get_edge_index(
            edge_index_masked,
            obj.num_motif,
            obj.cliques,
            obj.num_atoms,
            obj.clique_edges,
        )
        # Get the edge attribute tensor for the graph
        edge_attr = obj.get_edge_attr(
            edge_attr_masked,
            motif_edge_index,
            obj.num_motif,
            obj.clique_edges,
            obj.num_edge_features,
            obj.num_atoms,
        )

    return edge_attr_node, edge_index_node, edge_index, edge_attr


def main():
    import time
    from KGGraph.KGGProcessor.loader import load_tox21_dataset
    from pathlib import Path
    import sys

    # Get the root directory
    root_dir = Path(__file__).resolve().parents[2]
    # Add the root directory to the system path
    sys.path.append(str(root_dir))
    smiles_list, mols_list, labels = load_tox21_dataset(
        "./Data/classification/tox21/raw/tox21.csv"
    )
    t1 = time.time()
    for mol in mols_list:
        edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(
            mol,
            decompose_type="motif",
            mask_edge=True,
            mask_edge_ratio=0.1,
            fix_ratio=False,
        )
        print(edge_attr.size())
        print(edge_index.size())
    t2 = time.time()
    print(t2 - t1)


if __name__ == "__main__":
    main()
