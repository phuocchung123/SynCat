import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from typing import List


class BRCISDecomposition:
    @staticmethod
    def create_initial_cliques(mol: Chem.Mol) -> List[List[int]]:
        """
        Create initial cliques for each bond in the molecule.

        Args:
        mol (Chem.Mol): The RDKit molecule object.

        Returns:
        List[List[int]]: A list of cliques, where each clique is a list of atom indices.
        """
        cliques: List[List[int]] = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])
        return cliques

    @staticmethod
    def apply_brics_breaks(mol, cliques):
        """
        Apply BRICS breaks to the molecule and update cliques.

        Args:
        mol (Chem.Mol): The RDKit molecule object.
        cliques (List[List[int]]): The current list of cliques.

        Returns:
        List[List[int]]: Updated list of cliques after BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(mol))
        if len(res) == 0:
            return [list(range(mol.GetNumAtoms()))], []

        res_list = [[bond[0][0], bond[0][1]] for bond in res]

        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])
        return cliques, res_list

    @staticmethod
    def break_ring_bonds(mol, cliques):
        """
        Break bonds between ring and non-ring atoms.

        Args:
        mol (Chem.Mol): The RDKit molecule object.
        cliques (List[List[int]]): The current list of cliques.

        Returns:
        Tuple[List[List[int]], List[List[int]]]: Updated list of cliques and list of broken bonds.
        """
        break_ring_bonds = []
        for (
            c
        ) in cliques.copy():  # Use a copy to avoid modifying the list during iteration
            if len(c) > 1:
                if (
                    mol.GetAtomWithIdx(c[0]).IsInRing()
                    and not mol.GetAtomWithIdx(c[1]).IsInRing()
                ):
                    cliques.remove(c)
                    cliques.append([c[1]])
                    break_ring_bonds.append(c)
                elif (
                    mol.GetAtomWithIdx(c[1]).IsInRing()
                    and not mol.GetAtomWithIdx(c[0]).IsInRing()
                ):
                    cliques.remove(c)
                    cliques.append([c[0]])
                    break_ring_bonds.append(c)
        return cliques, break_ring_bonds

    @staticmethod
    def select_intersection_atoms(mol, cliques):
        """
        Select atoms at intersections as individual motifs.

        Parameters:
        mol (Chem.Mol): The RDKit molecule object.
        cliques (List[List[int]]): The current list of cliques.

        Returns:
        List[List[int]]: Updated list of cliques with intersection atoms selected.
        """
        break_intersections = []
        for atom in mol.GetAtoms():
            if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
                cliques.append([atom.GetIdx()])
                for nei in atom.GetNeighbors():
                    if [nei.GetIdx(), atom.GetIdx()] in cliques:
                        cliques.remove([nei.GetIdx(), atom.GetIdx()])
                        break_intersections.append([nei.GetIdx(), atom.GetIdx()])
                    elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                        cliques.remove([atom.GetIdx(), nei.GetIdx()])
                        break_intersections.append([atom.GetIdx(), nei.GetIdx()])
                    cliques.append([nei.GetIdx()])
        return cliques, break_intersections

    @staticmethod
    def merge_cliques(cliques):
        """
        Merge cliques with common elements.

        Parameters:
        cliques (List[List[int]]): The current list of cliques.

        Returns:
        List[List[int]]: Merged list of cliques.
        """
        for i in range(len(cliques) - 1):
            for j in range(i + 1, len(cliques)):
                if set(cliques[i]) & set(cliques[j]):  # Intersection check
                    cliques[i] = list(set(cliques[i]) | set(cliques[j]))  # Union
                    cliques[j] = []
        return [c for c in cliques if c]

    @staticmethod
    def find_edges(cliques, res_list, breaks_ring_bonds, break_intersections):
        """
        Find edges based on the breaks.

        Parameters:
        cliques (List[List[int]]): The list of cliques.
        res_list (List[Tuple]): BRICS breaks result.
        breaks_ring_bonds (List[List[int]]): List of broken bonds for ring and non-ring atoms.
        break_intersections (List[List[int]]): List of atoms at intersections as individual motifs.

        Returns:
        List[Tuple[int, int]]: List of edges representing the breaks.
        """
        edges = []
        for bond in res_list + breaks_ring_bonds + break_intersections:
            c1, c2 = None, None  # Initialize c1 and c2
            for c in range(len(cliques)):
                if bond[0] in cliques[c]:
                    c1 = c
                if bond[1] in cliques[c]:
                    c2 = c
            if c1 is not None and c2 is not None:
                edges.append((c1, c2))
        return edges

    @staticmethod
    def defragment(mol):
        """
        Perform BRICS decomposition on a molecule.

        Parameters:
        mol (Chem.Mol): The RDKit molecule object.

        Returns:
        Tuple[List[List[int]], List[Tuple[int, int]]]: List of cliques and list of edges representing the breaks.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = BRCISDecomposition.create_initial_cliques(mol)
        cliques, res_list = BRCISDecomposition.apply_brics_breaks(mol, cliques)
        cliques, break_ring_bonds = BRCISDecomposition.break_ring_bonds(mol, cliques)
        cliques, break_intersections = BRCISDecomposition.select_intersection_atoms(
            mol, cliques
        )
        cliques = BRCISDecomposition.merge_cliques(cliques)
        edges = BRCISDecomposition.find_edges(
            cliques, res_list, break_ring_bonds, break_intersections
        )
        return cliques, edges
