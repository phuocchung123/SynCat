from rdkit import Chem
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import List, Tuple, Dict


class TreeDecomposition:
    @staticmethod
    def create_initial_cliques(mol: Chem.Mol) -> List[List[int]]:
        """
        Create initial cliques for each non-ring bond in the molecule.

        Args:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        List[List[int]]: List of initial cliques.
        """
        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])
        return cliques

    @staticmethod
    def add_ring_cliques(mol: Chem.Mol, cliques: List[List[int]]) -> List[List[int]]:
        """
        Add ring cliques to the existing cliques list.

        Args:
        mol (Chem.Mol): RDKit molecule object.
        cliques (List[List[int]]): Existing cliques list.

        Returns:
        List[List[int]]: Updated cliques list with ring cliques added.
        """
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        return cliques

    @staticmethod
    def merge_cliques(
        cliques: List[List[int]], nei_list: List[List[int]]
    ) -> List[List[int]]:
        """
        Merge cliques with more than 2 atoms in common.

        Args:
        cliques (List[List[int]]): List of cliques.
        nei_list (List[List[int]]): Neighbor list for each atom.

        Returns:
        List[List[int]]: Merged list of cliques.
        """
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2:
                continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2:
                        continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []
        return [c for c in cliques if c]

    @staticmethod
    def create_neighbor_list(n_atoms: int, cliques: List[List[int]]) -> List[List[int]]:
        """
        Create a neighbor list for each atom based on cliques.

        Args:
        n_atoms (int): Number of atoms in the molecule.
        cliques (List[List[int]]): List of cliques.

        Returns:
        List[List[int]]: Neighbor list for each atom.
        """
        nei_list = [[] for _ in range(n_atoms)]
        for i, clique in enumerate(cliques):
            for atom in clique:
                nei_list[atom].append(i)
        return nei_list

    @staticmethod
    def initialize_edges(
        n_atoms: int, cliques: List[List[int]], nei_list: List[List[int]]
    ) -> Tuple[Dict[Tuple[int, int], int], List[List[int]]]:
        """
        Initialize edges between cliques.

        Args:
        n_atoms (int): Number of atoms in the molecule.
        cliques (List[List[int]]): List of cliques.
        nei_list (List[List[int]]): Neighbor list for each atom.

        Returns:
        Tuple[Dict[Tuple[int, int], int], List[List[int]]]: Dictionary of edges and updated cliques.
        """
        # print(cliques)
        # print(nei_list)
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            # for c in cnei:
            #     print(c)
            #     if len(cliques[c]) == 2:
            #         print(cnei)
            #         print(c)
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]

            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)
        return edges, cliques

    @staticmethod
    def compute_mst(
        cliques: List[List[int]], edges: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int]]:
        """
        Compute the maximum spanning tree of the clique graph.

        Args:
        cliques (List[List[int]]): List of cliques.
        edges (Dict[Tuple[int, int], int]): Dictionary of edges.

        Returns:
        List[Tuple[int, int]]: List of edges in the maximum spanning tree.
        """
        MST_MAX_WEIGHT = 100
        edge_data = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edge_data) == 0:
            return []

        row, col, data = zip(*edge_data)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        return [(row[i], col[i]) for i in range(len(row))]

    @staticmethod
    def defragment(
        mol: Chem.Mol,
        merge_rings: bool = True,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """
        Perform tree decomposition on a molecule.

        Args:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        Tuple[List[List[int]], List[Tuple[int, int]]]: List of cliques and list of edges representing the tree decomposition.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = TreeDecomposition.create_initial_cliques(mol)
        cliques = TreeDecomposition.add_ring_cliques(mol, cliques)
        nei_list = TreeDecomposition.create_neighbor_list(n_atoms, cliques)
        if merge_rings:
            cliques = TreeDecomposition.merge_cliques(cliques, nei_list)
        nei_list = TreeDecomposition.create_neighbor_list(n_atoms, cliques)
        edges, cliques = TreeDecomposition.initialize_edges(n_atoms, cliques, nei_list)
        return cliques, TreeDecomposition.compute_mst(cliques, edges)
