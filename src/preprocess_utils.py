import os
import numpy as np
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

import sys
from pathlib import Path
# Get the root directory
root_dir = Path(__file__).resolve().parents[0]
# Add the root directory to the system path
print(str(root_dir))
sys.path.append(str(root_dir))
from KGGraph.KGGEncode.x_feature import x_feature
from KGGraph.KGGEncode.edge_feature import edge_feature


chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

def feature(
    mol,
    decompose_type,
    mask_node=False,
    mask_edge=False,
    mask_node_ratio=0.1,
    mask_edge_ratio=0.1,
    fix_ratio=False,
):
    x_node, x, num_part = x_feature(
        mol,
        decompose_type=decompose_type,
        mask_node=mask_node,
        mask_node_ratio=mask_node_ratio,
        fix_ratio=fix_ratio,
    )
    edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(
        mol,
        decompose_type=decompose_type,
        mask_edge=mask_edge,
        mask_edge_ratio=mask_edge_ratio,
        fix_ratio=fix_ratio,
    )

    return x, num_part, edge_index, edge_attr

def add_mol(mol_dict, mol, decompose):
    x, num_part, edge_index, edge_attr= feature(mol, decompose)
    n_node = x.shape[0]
    n_edge = edge_attr.shape[0]
    mol_dict["n_node"].append(n_node)
    mol_dict["n_edge"].append(n_edge)
    mol_dict["node_attr"].append(x.numpy())

    if n_edge > 0:
        src = edge_index[0]
        dst = edge_index[1]

        mol_dict["edge_attr"].append(edge_attr)
        mol_dict["src"].append(src)
        mol_dict["dst"].append(dst)

    return mol_dict


def add_dummy(mol_dict):
    n_node = 1
    n_edge = 0
    node_attr = np.zeros((1, 125))
    mol_dict["n_node"].append(n_node)
    mol_dict["n_edge"].append(n_edge)
    mol_dict["node_attr"].append(node_attr)

    return mol_dict


def dict_list_to_numpy(mol_dict):
    mol_dict["n_node"] = np.array(mol_dict["n_node"]).astype(int)
    mol_dict["n_edge"] = np.array(mol_dict["n_edge"]).astype(int)
    mol_dict["node_attr"] = np.vstack(mol_dict["node_attr"])
    if np.sum(mol_dict["n_edge"]) > 0:
        mol_dict["edge_attr"] = np.vstack(mol_dict["edge_attr"])
        mol_dict["src"] = np.hstack(mol_dict["src"]).astype(int)
        mol_dict["dst"] = np.hstack(mol_dict["dst"]).astype(int)
    else:
        mol_dict["edge_attr"] = np.empty((0, 5))
        mol_dict["src"] = np.empty(0).astype(int)
        mol_dict["dst"] = np.empty(0).astype(int)

    return mol_dict
