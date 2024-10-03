import rdkit
import rdkit.Chem as Chem
from typing import List, Optional
import numpy as np

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def idxfunc(a):
    return a.GetAtomMapNum() - 1


def get_mol(smiles: str) -> Optional[Chem.Mol]:
    """Generate a molecule object from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    # if mol is not None:
    #     Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


def get_smiles(mol: Chem.Mol) -> str:
    """Convert a molecule object to a SMILES string."""
    return Chem.MolToSmiles(mol, kekuleSmiles=False)


def sanitize(mol: Chem.Mol, kekulize: bool = False) -> Optional[Chem.Mol]:
    """Sanitize the given molecule and optionally kekulize it."""
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except Exception:
        if mol is None:
            mol = None
    return mol


def get_atomic_number(atom: Chem.Atom) -> int:
    """Get the atomic number of the atom."""
    return atom.GetAtomicNum()


def get_atom_types(smiles: List[str]) -> List[int]:
    """Returns a list of unique atomic numbers present in
    the molecules represented by the given SMILES strings."""
    mols = [get_mol(smile) for smile in smiles]
    atom_types = []
    for mol in mols:
        for atom in mol.GetAtoms():
            if get_atomic_number(atom) not in atom_types:
                atom_types.append(get_atomic_number(atom))
    atom_types.sort()
    return atom_types


def atomic_num_features(mol, atom_types) -> List[int]:
    """Transform a molecule into a feature vector representing the
    atomic number of each atom in the molecule."""
    # Create a feature vector with shape (num_atoms, num_atom_types)
    atomic_features = np.zeros((mol.GetNumAtoms(), len(atom_types)))

    # Populate the feature vector with the atomic number of each atom
    for idx, atom in enumerate(mol.GetAtoms()):
        atomic_features[idx] = get_atomic_number(atom)

    # Convert the feature vector to binary (1 if atom has the corresponding type, 0 otherwise)
    atomic_features = np.where(
        atomic_features == np.tile(atom_types, (mol.GetNumAtoms(), 1)), 1, 0
    )

    return atomic_features


def set_atommap(mol: Chem.Mol, num: int = 0) -> Chem.Mol:
    """Set the atom map number for all atoms
    in the molecule to the specified number."""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol


def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    """Check if two atoms are equal based on their symbol and formal charge."""
    return (
        a1.GetSymbol() == a2.GetSymbol()
        and a1.GetFormalCharge() == a2.GetFormalCharge()
    )


def copy_atom(atom: Chem.Atom, atommap: bool = True) -> Chem.Atom:
    """Create a copy of the given atom,
    optionally preserving its atom map number."""
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom
