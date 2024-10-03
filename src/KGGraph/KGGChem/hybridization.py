from rdkit import Chem
from .atom_features import (
    get_degree,
    get_total_num_hs,
    get_hybridization,
)

# five features are in the order of (numbers of orbital s, numbers of orbital p,
# number of orbital d, total neighbors including hydrogens, number of lone pairs)
HYBRIDIZATION = {
    (1, 0): [1, 0, 0, 1, 0],  # AX1E0 => s => Ex: Na in NaI
    (0, 0): [1, 0, 0, 0, 0],  # AX0E0 => s => Ex: Zn2+
    (0, 1): [1, 0, 0, 1, 0],  # AX0E1 => s => Ex: H+
    (1, 1): [1, 1, 0, 1, 1],  # AX1E1 => sp => Ex: N of HCN
    (2, 0): [1, 1, 0, 2, 0],  # AX2E0 => sp => Ex: C#C
    (0, 2): [1, 1, 0, 0, 2],  # AX0E2 => sp => Ex: Cr smiles: [Cr+3]
    (2, 1): [1, 2, 0, 2, 1],  # AX2E1 => sp2 => Ex: N of Pyrimidine
    (1, 2): [1, 2, 0, 1, 2],  # AX1E2 => sp2 => Ex: O of C=O
    (3, 0): [1, 2, 0, 3, 0],  # AX3E0 => sp2 => Ex: N of pyrrole
    (0, 3): [1, 2, 0, 0, 3],  # AX0E3 => sp2 => Ex: Fe2+
    (1, 3): [1, 3, 0, 1, 3],  # AX1E3 => sp3 => Ex: R-X (X is halogen)
    (2, 2): [1, 3, 0, 2, 2],  # AX2E2 => sp3 => Ex: O of R-O-R'
    (3, 1): [1, 3, 0, 3, 1],  # AX3E1 => sp3 => Ex: N of NR3
    (4, 0): [1, 3, 0, 4, 0],  # AX4E0 => sp3 => Ex: C of CR4
    (0, 4): [1, 3, 0, 0, 4],  # AX0E4 => sp3 => Ex: X- (X is halogen) (KI)
    (6, -2): [
        1,
        3,
        0,
        6,
        0,
    ],  # AX6E0 => sp3 => Ex: Sb and hybridization: SP3 smiles: [SbH6+3]
    (2, 3): [1, 3, 1, 2, 3],  # AX2E3 => sp3d => Ex:Co
    (3, 2): [1, 3, 1, 3, 2],  # AX3E2 => sp3d
    (4, 1): [1, 3, 1, 4, 1],  # AX4E1 => sp3d
    (5, 0): [1, 3, 1, 5, 0],  # AX5E0 => sp3d => Ex: P of PCl5
    (0, 5): [
        1,
        3,
        1,
        0,
        5,
    ],  # AX0E5 => sp3d => Ex: Ag smiles: NC1=CC=C(S(=O)(=O)[N-]C2=NC=CC=N2)C=C1.[Ag+]
    (6, -1): [
        1,
        3,
        1,
        6,
        0,
    ],  # AX6E0 => sp3d => Ex: Al smiles: NC(=O)NC1N=C(O[AlH3](O)O)NC1=O
    (4, 2): [1, 3, 2, 4, 2],  # AX4E2 => sp3d2
    (2, 4): [1, 3, 2, 2, 4],  # AX2E4 => sp3d2 => Ex: Pd in PdCl2
    (3, 3): [1, 3, 2, 3, 3],  # AX3E3 => sp3d2 => Ex: Dy smiles: Cl[Dy](Cl)Cl
    (5, 1): [1, 3, 2, 5, 1],  # AX5E1 => sp3d2
    (1, 5): [1, 3, 2, 1, 5],  # AX1E5 => sp3d2 => Ex:CuI
    (6, 0): [1, 3, 2, 6, 0],  # AX6E0 => sp3d2 => Ex: S of SF6
}
# HYBRIDIZATION = {
#     (1, 0): [1, 0, 0],  # AX1E0 => s => Ex: Na in NaI
#     (0, 0): [1, 0, 0],  # AX0E0 => s => Ex: Zn2+
#     (0, 1): [1, 0, 0],  # AX0E1 => s => Ex: H+
#     (1, 1): [1, 1, 0],  # AX1E1 => sp => Ex: N of HCN
#     (2, 0): [1, 1, 0],  # AX2E0 => sp => Ex: C#C
#     (0, 2): [1, 1, 0],  # AX0E2 => sp => Ex: Cr smiles: [Cr+3]
#     (2, 1): [1, 2, 0],  # AX2E1 => sp2 => Ex: N of Pyrimidine
#     (1, 2): [1, 2, 0],  # AX1E2 => sp2 => Ex: O of C=O
#     (3, 0): [1, 2, 0],  # AX3E0 => sp2 => Ex: N of pyrrole
#     (0, 3): [1, 2, 0],  # AX0E3 => sp2 => Ex: Fe2+
#     (1, 3): [1, 3, 0],  # AX1E3 => sp3 => Ex: R-X (X is halogen)
#     (2, 2): [1, 3, 0],  # AX2E2 => sp3 => Ex: O of R-O-R'
#     (3, 1): [1, 3, 0],  # AX3E1 => sp3 => Ex: N of NR3
#     (4, 0): [1, 3, 0],  # AX4E0 => sp3 => Ex: C of CR4
#     (0, 4): [1, 3, 0],  # AX0E4 => sp3 => Ex: X- (X is halogen) (KI)
#     (6, -2): [
#         1,
#         3,
#         0,
#     ],  # AX6E0 => sp3 => Ex: Sb and hybridization: SP3 smiles: [SbH6+3]
#     (2, 3): [1, 3, 1],  # AX2E3 => sp3d => Ex:Co
#     (3, 2): [1, 3, 1],  # AX3E2 => sp3d
#     (4, 1): [1, 3, 1],  # AX4E1 => sp3d
#     (5, 0): [1, 3, 1],  # AX5E0 => sp3d => Ex: P of PCl5
#     (0, 5): [
#         1,
#         3,
#         1,
#     ],  # AX0E5 => sp3d => Ex: Ag smiles: NC1=CC=C(S(=O)(=O)[N-]C2=NC=CC=N2)C=C1.[Ag+]
#     (6, -1): [
#         1,
#         3,
#         1,
#     ],  # AX6E0 => sp3d => Ex: Al smiles: NC(=O)NC1N=C(O[AlH3](O)O)NC1=O
#     (4, 2): [1, 3, 2],  # AX4E2 => sp3d2
#     (2, 4): [1, 3, 2],  # AX2E4 => sp3d2 => Ex: Pd in PdCl2
#     (3, 3): [1, 3, 2],  # AX3E3 => sp3d2 => Ex: Dy smiles: Cl[Dy](Cl)Cl
#     (5, 1): [1, 3, 2],  # AX5E1 => sp3d2
#     (1, 5): [1, 3, 2],  # AX1E5 => sp3d2 => Ex:CuI
#     (6, 0): [1, 3, 2],  # AX6E0 => sp3d2 => Ex: S of SF6
# }

max_bond_hybridization = {
    "SP3D2": 6,
    "SP3D": 5,
    "SP3": 4,
    "SP2": 3,
    "SP": 2,
    "S": 1,
}


class HybridizationFeaturize:
    """
    Class to compute hybridization features for a given dataset of molecules.
    """

    @staticmethod
    def total_sigma_bond(atom: Chem.Atom) -> int:
        """
        Compute the total number of single bonds for a given atom, including the bonds with hydrogen atoms.

        Parameters:
        atom (Chem.Atom): The atom for which the total number of single bonds is to be computed.

        Returns:
        int: The total number of single bonds for the given atom.
        """
        total_sigma_bond = get_degree(atom) + get_total_num_hs(atom)
        return total_sigma_bond

    @staticmethod
    def num_bond_hybridization(atom: Chem.Atom) -> int:
        """
        Compute the number of bonds involved in hybridization for a given atom based on the atom's hybridization state.

        Parameters:
        atom (Chem.Atom): The atom for which the number of bonds involved in hybridization is to be computed.

        Returns:
        int: The number of bonds involved in hybridization for the given atom.
        """
        num_bonds_hybridization = max_bond_hybridization.get(get_hybridization(atom), 0)
        return num_bonds_hybridization

    @staticmethod
    def num_lone_pairs(atom: Chem.Atom) -> int:
        """
        Calculate the number of lone pairs on a given atom. This method estimates the number of lone pairs by subtracting the total number
        of single bonds (including those with hydrogens) from the atom's hybridization-based expected bonding capacity. The calculation assumes
        that each atom has a fixed bonding capacity based on its hybridization state (sp, sp2, sp3, etc.), and any valence electrons not involved
        in single bonding can be considered as part of lone pairs.

        Parameters:
        atom (Chem.Atom): The atom for which the number of lone pairs is to be computed. This atom should be part of a molecule object.

        Returns:
        int: The estimated number of lone pairs on the atom. The value is computed based on the atom's hybridization and its single bonds.

        Note:
        This method relies on the `num_bond_hybridization` and `total_sigma_bond` methods from the `HybridizationFeaturize` class. Ensure that
        these methods correctly compute the atom's expected bonding capacity based on hybridization and the actual count of single bonds,
        respectively, for accurate results.
        """
        num_lone_pairs = HybridizationFeaturize.num_bond_hybridization(
            atom
        ) - HybridizationFeaturize.total_sigma_bond(atom)
        return num_lone_pairs

    @staticmethod
    def feature(atom: Chem.Atom) -> tuple[int, int, list[int]]:
        """
        Compute a feature vector for a given atom, including the total number of single bonds, the number of lone pairs,
        and a predefined feature vector based on the atom's hybridization characteristics. This vector is intended to capture
        aspects of the atom that are relevant to its chemical behavior and properties.

        Parameters:
        atom (Chem.Atom): The atom for which the feature vector is to be computed.

        Returns:
        tuple[int, int, list[int]]: A tuple containing the total number of single bonds to the atom (including hydrogen atoms),
        the number of lone electron pairs on the atom, and a list representing the hybridization feature vector. The hybridization
        feature vector is predefined and retrieved based on the total number of single bonds and the number of lone pairs.
        """
        total_sigma_bonds = HybridizationFeaturize.total_sigma_bond(atom)
        num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
        hybri_feat = HYBRIDIZATION.get(
            (total_sigma_bonds, num_lone_pairs), [0, 0, 0, 0, 0]
        )  # features for UNSPECIFIED hybridization is [0,0,0,0,0]

        return total_sigma_bonds, num_lone_pairs, hybri_feat
