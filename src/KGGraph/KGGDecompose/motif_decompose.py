from rdkit.Chem import BRICS
from rdkit import Chem
import pathlib
import sys

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGChem.chemutils import get_clique_mol


class MotifDecomposition:
    @staticmethod
    def _initial_cliques(mol: Chem.Mol):
        """
        Create initial cliques based on the bonds of the molecule.

        Returns:
        list: A list of initial cliques.
        """
        cliques = [
            [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            for bond in mol.GetBonds()
        ]
        return cliques

    @staticmethod
    def _apply_brics_breaks(cliques, mol):
        """
        Apply BRICS rules to break bonds and update cliques.

        Args:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after applying BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(mol))
        res_list = [[bond[0][0], bond[0][1]] for bond in res]

        for bond in res:
            bond_indices = [bond[0][0], bond[0][1]]
            if bond_indices in cliques:
                cliques.remove(bond_indices)
            else:
                cliques.remove(
                    bond_indices[::-1]
                )  # Reverse indices if not found in order
            cliques.extend([[bond[0][0]], [bond[0][1]]])
        return cliques, res_list

    @staticmethod
    def _merge_cliques(cliques, mol):
        """
        Merge overlapping cliques.

        Args:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after merging.
        """
        n_atoms = mol.GetNumAtoms()
        for i in range(len(cliques) - 1):
            if i >= len(cliques):
                break
            for j in range(i + 1, len(cliques)):
                if j >= len(cliques):
                    break
                if set(cliques[i]) & set(cliques[j]):  # Intersection is not empty
                    cliques[i] = list(set(cliques[i]) | set(cliques[j]))  # Union
                    cliques[j] = []
            cliques = [c for c in cliques if c]
        cliques = [c for c in cliques if n_atoms > len(c) > 0]
        return cliques

    @staticmethod
    def _refine_cliques(cliques, mol):
        """
        Refine cliques to consider symmetrically equivalent substructures.

        Args:
        cliques (list): The current list of cliques.

        Returns:
        list: Refined list of cliques.
        """
        n_atoms = mol.GetNumAtoms()
        num_cli = len(cliques)
        ssr_mol = Chem.GetSymmSSSR(mol)
        for i in range(num_cli):
            c = cliques[i]
            cmol = get_clique_mol(mol, c)
            ssr = Chem.GetSymmSSSR(cmol)
            if len(ssr) > 1:
                for ring in ssr_mol:
                    if set(list(ring)) <= set(c):
                        cliques.append(list(ring))
                cliques[i] = []

        cliques = [c for c in cliques if n_atoms > len(c) > 0]
        return cliques

    @staticmethod
    def find_edges(cliques, res_list):
        """
        Find edges based on the breaks.

        Args:
        cliques (List[List[int]]): The list of cliques.
        res_list (List[Tuple]): BRICS breaks result.

        Returns:
        List[Tuple[int, int]]: List of edges representing the breaks.
        """
        edges = []
        for bond in res_list:
            c1, c2 = None, None  # Initialize c1 and c2
            for c in range(len(cliques)):
                if bond[0] in cliques[c]:
                    c1 = c
                if bond[1] in cliques[c]:
                    c2 = c
            if c1 is not None and c2 is not None:
                edges.append((c1, c2))
        for c in range(len(cliques)):
            for i in range(c + 1, len(cliques)):
                if set(cliques[c]) & set(cliques[i]):
                    c1, c2 = c, i
                    edges.append((c1, c2))
        return edges

    @staticmethod
    def defragment(mol):
        """
        Perform motif decomposition on the molecule.

        Returns:
        list: A list of atom indices representing the decomposed motifs.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = MotifDecomposition._initial_cliques(mol)
        cliques, res_list = MotifDecomposition._apply_brics_breaks(cliques, mol)
        cliques = MotifDecomposition._merge_cliques(cliques, mol)
        cliques = MotifDecomposition._refine_cliques(cliques, mol)
        edges = MotifDecomposition.find_edges(cliques, res_list)
        return cliques, edges
