import os
import unittest
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from src.preprocess_utils import (
    _DA,
    _chirality,
    _stereochemistry,
    add_dummy,
    dict_list_to_numpy,
)


class TestPreprocessUtils(unittest.TestCase):

    def setUp(self):
        # Setup that runs once before all tests
        self.chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        )
        self.mol_dict = {
            "n_node": [],
            "n_edge": [],
            "node_attr": [],
            "edge_attr": [],
            "src": [],
            "dst": [],
        }

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

    def test_add_dummy(self):
        # Testing the addition of dummy node
        result = add_dummy(self.mol_dict)
        self.assertEqual(result["n_node"], [1])
        self.assertEqual(result["n_edge"], [0])
        self.assertTrue((result["node_attr"][0] == np.zeros((1, 155))).all())

    def test_dict_list_to_numpy_no_edges(self):
        # Test conversion of empty edge attributes
        self.mol_dict = add_dummy(self.mol_dict)  # Add dummy data
        result = dict_list_to_numpy(self.mol_dict)

        self.assertIsInstance(result["n_node"], np.ndarray)
        self.assertIsInstance(result["n_edge"], np.ndarray)
        self.assertIsInstance(result["node_attr"], np.ndarray)
        self.assertIsInstance(result["edge_attr"], np.ndarray)
        self.assertIsInstance(result["src"], np.ndarray)
        self.assertIsInstance(result["dst"], np.ndarray)

        self.assertEqual(result["n_node"].dtype, int)
        self.assertEqual(result["n_edge"].dtype, int)
        self.assertEqual(result["node_attr"].dtype, bool)
        self.assertEqual(result["edge_attr"].shape, (0, 9))  # check
        self.assertEqual(result["src"].size, 0)
        self.assertEqual(result["dst"].size, 0)

    def test_dict_list_to_numpy_with_edges(self):
        # Test conversion with actual edges
        self.mol_dict = {
            "n_node": [2],
            "n_edge": [1],
            "node_attr": [np.zeros((1, 155)), np.ones((1, 155))],
            "edge_attr": [
                np.ones((1, 9))
            ],  # Assuming length of bond_list + 4 here is 9
            "src": [[0]],
            "dst": [[1]],
        }
        result = dict_list_to_numpy(self.mol_dict)

        self.assertTrue((result["n_node"] == np.array([2])).all())
        self.assertTrue((result["n_edge"] == np.array([1])).all())
        self.assertTrue((result["node_attr"].shape == (2, 155)))
        self.assertTrue((result["edge_attr"].shape == (1, 9)))
        self.assertTrue((result["src"] == np.array([0])).all())
        self.assertTrue((result["dst"] == np.array([1])).all())


if __name__ == "__main__":
    unittest.main()
