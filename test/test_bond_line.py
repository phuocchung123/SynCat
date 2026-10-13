"""Tests for the bond-line molecular graph transformation (src/bond_line.py)."""

import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rdkit import Chem  # noqa: E402

from bond_line import (  # noqa: E402
    LINE_FEATURE_VERSION,
    LINE_NODE_BOND,
    LINE_NODE_ISOLATED,
    BondLineData,
    bond_line_node_dim,
    to_bond_line,
)
from data import GraphDataset  # noqa: E402
from preprocess_utils import add_mol, dict_list_to_numpy  # noqa: E402
from reaction_data import get_graph_data  # noqa: E402

KNOWN_KEYS = ("n_node", "n_edge", "node_attr", "edge_attr", "src", "dst")


def mol_to_data(smiles: str) -> Data:
    """Build one source molecule Data graph from a SMILES (disconnected
    fragments such as "CC.[Na+]" stay in the same graph, unlike the reaction
    parser which splits on ".")."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    mol_dict = {key: [] for key in KNOWN_KEYS}
    add_mol(mol_dict, mol)
    mol_dict = dict_list_to_numpy(mol_dict)
    x = torch.tensor(mol_dict["node_attr"], dtype=torch.float32)
    edge_index = torch.tensor(
        np.array([mol_dict["src"], mol_dict["dst"]]), dtype=torch.long
    ).reshape(2, -1)
    edge_attr = torch.tensor(np.asarray(mol_dict["edge_attr"]), dtype=torch.float32)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.reshape(0, 9)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# 1. Topology of the transformation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,num_nodes,num_edges",
    [
        ("CCO", 2, 2),
        ("CC(C)C", 3, 6),
        ("C1CC1", 3, 6),
        ("CC", 1, 0),
        ("O", 1, 0),
        ("CC.[Na+]", 2, 0),
        ("CC.CC", 2, 0),
    ],
)
def test_line_topology(smiles, num_nodes, num_edges):
    line = to_bond_line(mol_to_data(smiles))
    assert line.x.shape[0] == num_nodes
    assert line.edge_index.shape == (2, num_edges)
    assert line.rev_edge.shape == (num_edges,)


def test_flag_column_distinguishes_bonds_from_isolated_atoms():
    cco = to_bond_line(mol_to_data("CCO"))
    assert cco.x[:, -1].tolist() == [LINE_NODE_BOND, LINE_NODE_BOND]

    water = to_bond_line(mol_to_data("O"))
    assert water.x[:, -1].tolist() == [LINE_NODE_ISOLATED]

    mixed = to_bond_line(mol_to_data("CC.[Na+]"))
    assert sorted(mixed.x[:, -1].tolist()) == [LINE_NODE_BOND, LINE_NODE_ISOLATED]


def test_feature_width_is_derived():
    data = mol_to_data("CCO")
    line = to_bond_line(data)
    atom_dim = data.x.shape[1]
    bond_dim = data.edge_attr.shape[1]
    assert atom_dim == 155 and bond_dim == 9
    assert line.x.shape[1] == bond_line_node_dim(atom_dim, bond_dim) == 320


# ---------------------------------------------------------------------------
# 2. Feature construction
# ---------------------------------------------------------------------------


def test_bond_node_feature_layout():
    data = mol_to_data("CCO")
    line = to_bond_line(data)
    atom_dim = data.x.shape[1]
    bond_dim = data.edge_attr.shape[1]

    # First transformed node corresponds to the first chemical bond seen.
    u, v = int(data.edge_index[0, 0]), int(data.edge_index[1, 0])
    expected = torch.cat(
        [
            data.x[u] + data.x[v],
            torch.abs(data.x[u] - data.x[v]),
            data.edge_attr[0],
            torch.zeros(1),
        ]
    )
    torch.testing.assert_close(line.x[0], expected)
    assert expected.shape[0] == 2 * atom_dim + bond_dim + 1


def test_isolated_atom_fallback_layout():
    data = mol_to_data("O")
    line = to_bond_line(data)
    atom_dim = data.x.shape[1]
    bond_dim = data.edge_attr.shape[1]
    expected = torch.cat(
        [
            data.x[0],
            torch.zeros(atom_dim),
            torch.zeros(bond_dim),
            torch.ones(1),
        ]
    )
    torch.testing.assert_close(line.x[0], expected)


def test_dummy_placeholder_is_zero_and_masked():
    real = to_bond_line(mol_to_data("O"), is_real=True)
    dummy = to_bond_line(mol_to_data("O"), is_real=False)
    assert isinstance(dummy, BondLineData)
    assert dummy.x.shape == (1, 320)
    assert torch.count_nonzero(dummy.x) == 0
    assert dummy.edge_index.shape == (2, 0)
    assert dummy.rev_edge.shape == (0,)
    # A real isolated atom and a masked placeholder are distinct.
    assert not torch.equal(real.x, dummy.x)


# ---------------------------------------------------------------------------
# 3. Robustness of the transformation
# ---------------------------------------------------------------------------


def test_endpoint_reversal_preserves_features():
    data = mol_to_data("CCO")
    line = to_bond_line(data)
    flipped = Data(
        x=data.x,
        edge_index=data.edge_index.flip(0),
        edge_attr=data.edge_attr,
    )
    line_flipped = to_bond_line(flipped)
    torch.testing.assert_close(line.x, line_flipped.x)
    assert line.edge_index.shape == line_flipped.edge_index.shape


def test_arbitrary_edge_ordering_preserves_node_feature_multiset():
    data = mol_to_data("CC(C)C")
    line = to_bond_line(data)

    perm = torch.randperm(data.edge_index.shape[1])
    shuffled = Data(
        x=data.x,
        edge_index=data.edge_index[:, perm],
        edge_attr=data.edge_attr[perm],
    )
    line_shuffled = to_bond_line(shuffled)

    assert line_shuffled.x.shape == line.x.shape
    assert line_shuffled.edge_index.shape == line.edge_index.shape
    rows_a = line.x.sort(dim=0).values
    rows_b = line_shuffled.x.sort(dim=0).values
    torch.testing.assert_close(rows_a, rows_b)


def test_disconnected_components_stay_disconnected():
    line = to_bond_line(mol_to_data("CC.[Na+]"))
    assert line.edge_index.shape[1] == 0  # no line edge crosses the components

    two_fragments = to_bond_line(mol_to_data("CC.CC"))
    assert two_fragments.x.shape[0] == 2
    assert two_fragments.edge_index.shape[1] == 0


def test_atom_renumbering_preserves_graph():
    # "CCO" and "OCC" are the same molecule with a different atom numbering.
    a = to_bond_line(mol_to_data("CCO"))
    b = to_bond_line(mol_to_data("OCC"))
    assert a.x.shape[0] == b.x.shape[0]
    assert a.edge_index.shape[1] == b.edge_index.shape[1]
    torch.testing.assert_close(
        a.x.sort(dim=0).values, b.x.sort(dim=0).values
    )


def test_reject_missing_reverse_edge():
    data = mol_to_data("CCO")
    half = Data(
        x=data.x,
        edge_index=data.edge_index[:, : data.edge_index.shape[1] // 2],
        edge_attr=data.edge_attr[: data.edge_index.shape[1] // 2],
    )
    with pytest.raises(ValueError, match="reverse directed edge"):
        to_bond_line(half)


def test_reject_mismatched_reverse_attributes():
    data = mol_to_data("CCO")
    edge_attr = data.edge_attr.clone()
    half = edge_attr.shape[0] // 2
    edge_attr[half:] = edge_attr[half:] + 1.0
    bad = Data(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr)
    with pytest.raises(ValueError, match="mismatched features"):
        to_bond_line(bad)


def test_reject_chemical_self_loop():
    data = mol_to_data("CC")
    loop = torch.tensor([[0], [0]], dtype=torch.long)
    bad = Data(
        x=data.x,
        edge_index=loop,
        edge_attr=torch.zeros(1, 9),
    )
    with pytest.raises(ValueError, match="self-loop"):
        to_bond_line(bad)


# ---------------------------------------------------------------------------
# 4. Batching of the transformed graph
# ---------------------------------------------------------------------------


def test_rev_edge_batching_with_unequal_sizes():
    branched = to_bond_line(mol_to_data("CC(C)C"))  # 3 nodes, 6 directed edges
    empty_a = to_bond_line(mol_to_data("CC"))  # 1 node, 0 edges
    empty_b = to_bond_line(mol_to_data("O"))  # 1 node, 0 edges
    batch = Batch.from_data_list([branched, empty_a, empty_b])

    assert batch.num_graphs == 3
    edge_index = batch.edge_index
    rev = batch.rev_edge
    num_edges = edge_index.shape[1]
    assert num_edges == 6
    assert rev.shape == (num_edges,)

    # rev is an edge-on-edge involution and flips every edge.
    torch.testing.assert_close(rev[rev], torch.arange(num_edges))
    torch.testing.assert_close(edge_index[:, rev], edge_index.flip(0))
    # Every line edge stays inside one molecule.
    torch.testing.assert_close(
        batch.batch[edge_index[0]], batch.batch[edge_index[1]]
    )


def test_reaction_parser_keeps_dot_separated_fragments_as_slots():
    # "[Na+].[Cl-]>>O" must become two reactant molecule slots.
    rmol, pmol, _ = get_graph_data(["[Na+].[Cl-]>>O"], 2, 1)
    assert len(rmol) == 2
    assert list(rmol[0]["dummy"]) == [True]
    assert list(rmol[1]["dummy"]) == [True]
    assert rmol[0]["n_node"].tolist() == [1]
    assert rmol[1]["n_node"].tolist() == [1]
    assert rmol[0]["n_edge"].tolist() == [0]
    assert rmol[1]["n_edge"].tolist() == [0]


def test_graph_dataset_bond_line_builds_bond_line_data():
    reaction_smiles = ["CCO>>CCO", "CC.[Na+]>>O"]
    rmol, pmol, reaction = get_graph_data(reaction_smiles, 3, 1)
    reaction["y"] = np.array([0.0, 0.0])

    atom_ds = GraphDataset(rmol=rmol, pmol=pmol, reaction=reaction)
    bond_ds = GraphDataset(
        rmol=rmol, pmol=pmol, reaction=reaction, graph_representation="bond_line"
    )
    assert atom_ds[0][0].x.shape[1] == 155
    for item in bond_ds[0][: bond_ds.rmol_max_cnt]:
        assert isinstance(item, BondLineData)
        assert item.x.shape[1] == 320
    assert bond_ds.atom_dim == 155 and bond_ds.bond_dim == 9
    assert LINE_FEATURE_VERSION == 1
