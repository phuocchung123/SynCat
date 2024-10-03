import sys
from pathlib import Path
import networkx as nx
from rdkit import Chem
from typing import Tuple, List, Dict, Set
from KGGraph.KGGChem.atom_utils import (
    get_inter_label,
    set_atommap,
    get_smiles,
    get_assm_cands,
)

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGDecompose.MotitDcp.smotif_decompose import SMotifDecomposition
from KGGraph.KGGDecompose.MotitDcp.jin_decompose import TreeDecomposition
from KGGraph.KGGDecompose.MotitDcp.brics_decompose import BRCISDecomposition
from KGGraph.KGGDecompose.MotitDcp.motif_decompose import MotifDecomposition


class JTConstruct:
    def __init__(self, mol: Chem.Mol, fragment_type: str = "smotif"):
        """
        Initialize JTConstruct with a molecule.

        Args mol: RDKit Molecule object.
        """
        self.mol = mol
        self.fragment_type = fragment_type
        self.mol_tree = None
        self.graph = None
        self.order = None
        self.BOND_LIST = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

    @staticmethod
    def build_mol_graph(
        mol: Chem.Mol, bond_list: List[Chem.rdchem.BondType]
    ) -> nx.DiGraph:
        """
        Constructs a molecular graph from the RDKit Molecule object.

        Args mol: RDKit Molecule object.
        Return: NetworkX Directed Graph representing the molecule.
        """
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]["label"] = (
                atom.GetSymbol(),
                atom.GetFormalCharge(),
            )

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = bond_list.index(bond.GetBondType())
            graph[a1][a2]["label"] = btype
            graph[a2][a1]["label"] = btype

        return graph

    @staticmethod
    def find_clusters(
        mol: Chem.Mol, fragment_type: str = "smotif"
    ) -> Tuple[List[Set[int]], Dict[Tuple[int, int], int]]:
        """
        Identifies clusters in the molecule.

        Args mol: RDKit Molecule object.
        Return: A tuple containing a list of clusters and edges.
        """
        if fragment_type == "smotif":
            cliques, edges = SMotifDecomposition().defragment(mol)
            # print(cliques)
        elif fragment_type == "brics":
            cliques, edges = BRCISDecomposition().defragment(mol)
            # print(cliques)
        elif fragment_type == "motif":
            cliques, edges = MotifDecomposition().defragment(mol)
            # print(cliques)
        elif fragment_type == "jin":
            cliques, edges = TreeDecomposition().defragment(mol)
            # print(cliques)
        return list(cliques), edges

    @staticmethod
    def tree_decomp(mol: Chem.Mol, fragment_type: str = "smotif") -> nx.Graph:
        """
        Performs tree decomposition on the molecule.

        Args mol: RDKit Molecule object.
        Return: NetworkX Graph representing the tree decomposition.
        """
        clusters, edges = JTConstruct.find_clusters(mol, fragment_type)
        tree_graph = nx.empty_graph(len(clusters))

        for edge in edges:
            tree_graph.add_edge(edge[0], edge[1], weight=edges[edge])
        n, m = len(tree_graph.nodes), len(tree_graph.edges)
        assert n - m <= 1  # must be connected
        return tree_graph if n - m == 1 else nx.maximum_spanning_tree(tree_graph)

    @staticmethod
    def dfs(
        order: List[Tuple[int, int, int]],
        pa: Dict[int, int],
        prev_sib: List[List[int]],
        x: int,
        fa: int,
        mol_tree: nx.DiGraph,
    ):
        """
        Depth-first search for labeling tree nodes.

        Args:
        - order: Order of traversal.
        - pa: Parent nodes.
        - prev_sib: Previous siblings.
        - x: Current node.
        - fa: Parent of current node.
        - mol_tree: NetworkX DiGraph of the molecule tree.
        """
        pa[x] = fa
        sorted_child = sorted(
            [y for y in mol_tree[x] if y != fa]
        )  # better performance with fixed order
        for idx, y in enumerate(sorted_child):
            mol_tree[x][y]["label"] = 0
            mol_tree[y][x]["label"] = idx + 1  # position encoding
            prev_sib[y] = sorted_child[:idx]
            prev_sib[y] += [x, fa] if fa >= 0 else [x]
            order.append((x, y, 1))
            JTConstruct.dfs(order, pa, prev_sib, y, x, mol_tree)
            order.append((y, x, 0))

    def label_tree(self):
        """
        Labels the tree decomposed graph of the molecule.
        """
        self.mol_tree = nx.DiGraph(self.tree_decomp(self.mol, self.fragment_type))
        self.order, pa = [], {}
        clusters, _ = self.find_clusters(self.mol)
        prev_sib = [[] for _ in range(len(clusters))]
        self.dfs(self.order, pa, prev_sib, 0, -1, self.mol_tree)

        self.order.append((0, None, 0))  # last backtrack at root

        self.graph = self.build_mol_graph(self.mol, self.BOND_LIST)
        for a in self.mol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)

        for i, cls in enumerate(clusters):
            inter_atoms = set(cls) & set(clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label(self.mol, cls, inter_atoms)
            self.mol_tree.nodes[i]["ismiles"] = ismiles = get_smiles(cmol)
            self.mol_tree.nodes[i]["inter_label"] = inter_label
            self.mol_tree.nodes[i]["smiles"] = smiles = get_smiles(set_atommap(cmol))
            self.mol_tree.nodes[i]["label"] = (
                (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            )
            self.mol_tree.nodes[i]["cluster"] = cls
            self.mol_tree.nodes[i]["assm_cands"] = []

            if (
                pa[i] >= 0 and len(clusters[pa[i]]) > 2
            ):  # uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in clusters[c]]
                pa_cls = clusters[pa[i]]
                self.mol_tree.nodes[i]["assm_cands"] = get_assm_cands(
                    self.mol, hist, inter_label, pa_cls, len(inter_atoms)
                )

                child_order = self.mol_tree[i][pa[i]]["label"]
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.graph[ch_atom][fa_atom]["label"]
                            if (
                                type(label) is int
                            ):  # in case one bond is assigned multiple times
                                self.graph[ch_atom][fa_atom]["label"] = (
                                    label,
                                    child_order,
                                )
        return self.order, self.mol_tree, self.graph
