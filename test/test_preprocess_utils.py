import os
import unittest
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from src.preprocess_utils import _DA, _chirality, _stereochemistry


class TestPreprocessUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup that runs once before all tests
        cls.chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        )

    def test_DA(self):
        # Create a simple molecule for testing
        mol = Chem.MolFromSmiles("NC(O)C=O")  # Acetamide
        D_list, A_list = _DA(mol)

        # Assertions
        self.assertEqual([0, 2], D_list)  # N and O donor
        self.assertEqual([2, 4], A_list)  # 2 oxy as acceptor

    def test_chirality(self):
        # Create a chiral molecule
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](C)Cl")
        chiral_atom = mol.GetAtomWithIdx(1)  # Second atom which is chiral

        # Add property manually to test
        chiral_atom.SetProp("Chirality", "Tet_CW")

        c_list = _chirality(chiral_atom)
        self.assertEqual(c_list, [1, 0])

    def test_stereochemistry(self):
        # Create a molecule with stereochemistry
        mol = Chem.MolFromSmiles("F/C=C/F")
        bond = mol.GetBondBetweenAtoms(1, 2)  # Bond with stereochemistry

        # Add property manually to test
        bond.SetProp("Stereochemistry", "Bond_Cis")

        s_list = _stereochemistry(bond)
        self.assertEqual(s_list, [1, 0])


if __name__ == "__main__":
    unittest.main()
