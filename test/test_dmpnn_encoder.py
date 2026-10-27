"""Tests for the D-MPNN encoder that replaces GINE on this branch (CPU)."""

import os
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dmpnn import DMPNN  # noqa: E402
from model import model  # noqa: E402

NODE_DIM = 6
EDGE_DIM = 4
HIDDEN = 8


def _make_graph(bonds, num_nodes, seed=0, node_dim=NODE_DIM, edge_dim=EDGE_DIM):
    """
    Build a graph that carries both directions of every bond, as the
    preprocessing does.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, node_dim, generator=generator)
    if not bonds:
        return Data(
            x=x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim)),
        )

    bond_feats = torch.randn(len(bonds), edge_dim, generator=generator)
    starts = [u for u, _ in bonds]
    ends = [v for _, v in bonds]
    edge_index = torch.tensor([starts + ends, ends + starts], dtype=torch.long)
    edge_attr = torch.cat((bond_feats, bond_feats), dim=0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _chain(num_nodes=4, seed=0):
    return _make_graph([(i, i + 1) for i in range(num_nodes - 1)], num_nodes, seed)


def _encoder(depth=3, dropout=0.0):
    torch.manual_seed(0)
    return DMPNN(NODE_DIM, EDGE_DIM, depth, HIDDEN, dropout)


BONDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]


def test_output_shape_matches_gin_interface():
    encoder = _encoder()
    out = encoder(_chain())
    assert out.shape == (1, HIDDEN)


def test_batched_readout_is_sum_of_single_graphs():
    encoder = _encoder().eval()
    graphs = [_chain(4, seed=1), _chain(5, seed=2), _make_graph([(0, 1)], 2, seed=3)]

    with torch.no_grad():
        batched = encoder(Batch.from_data_list(graphs))
        singles = torch.cat([encoder(graph) for graph in graphs], dim=0)

    # The readout is a sum over atoms, so it is additive over batched graphs.
    assert torch.allclose(batched, singles, atol=1e-6)


def test_every_directed_edge_has_a_reverse_partner():
    graph = _make_graph(BONDS, 5)
    reverse = DMPNN.reverse_edge_index(graph.edge_index, graph.num_nodes)
    expected = reverse[reverse]
    assert torch.equal(expected, torch.arange(graph.edge_index.size(1)))
    src, dst = graph.edge_index
    assert torch.equal(src[reverse], dst)
    assert torch.equal(dst[reverse], src)


def test_missing_reverse_edge_is_rejected():
    graph = _make_graph(BONDS, 5)
    # Drop one direction of the first bond.
    broken = Data(
        x=graph.x,
        edge_index=graph.edge_index[:, 1:],
        edge_attr=graph.edge_attr[1:],
    )
    with pytest.raises(ValueError, match="reverse edge"):
        _encoder()(broken)


def test_relabelling_atoms_does_not_change_the_embedding():
    encoder = _encoder().eval()
    graph = _make_graph(BONDS, 5, seed=4)

    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    relabelled = Data(
        x=graph.x[permutation],
        edge_index=inverse[graph.edge_index],
        edge_attr=graph.edge_attr,
    )

    with torch.no_grad():
        a = encoder(graph)
        b = encoder(relabelled)

    assert torch.allclose(a, b, atol=1e-6)


def test_depth_one_skips_updates_and_depth_uses_more_parameters():
    shallow = _encoder(depth=1)
    out = shallow(_chain())
    assert out.shape == (1, HIDDEN)

    # A D-MPNN shares one update across steps, so depth does not add parameters.
    deep = _encoder(depth=5)
    assert sum(p.numel() for p in shallow.parameters()) == sum(
        p.numel() for p in deep.parameters()
    )


def test_gradients_reach_the_message_passing_weights():
    encoder = _encoder()
    encoder(_chain()).sum().backward()

    for name in ("project_bond_feats", "update_bond_feats", "project_atom_feats"):
        grads = getattr(encoder, name).weight.grad
        assert grads is not None
        assert torch.isfinite(grads).all()
        assert grads.abs().sum() > 0


def test_single_atom_graphs_are_supported():
    encoder = _encoder().eval()
    atom = _make_graph([], 1, seed=5)
    out = encoder(Batch.from_data_list([atom, atom]))
    assert out.shape == (2, HIDDEN)
    assert torch.isfinite(out).all()


def test_model_forward_with_dmpnn_encoder():
    torch.manual_seed(0)
    net = model(
        NODE_DIM,
        EDGE_DIM,
        3,
        HIDDEN,
        0.0,
        num_heads=1,
        reaction_combine="concat",
        attention_on="reactants",
        head="linear",
    ).eval()
    assert isinstance(net.gnn, DMPNN)

    # Two reactions, two reactant slots each; the second reaction pads slot 1.
    reactant_slot0 = Batch.from_data_list([_chain(4, seed=6), _chain(5, seed=7)])
    reactant_slot1 = Batch.from_data_list([_chain(3, seed=8), _chain(2, seed=9)])
    product_slot0 = Batch.from_data_list([_chain(4, seed=10), _chain(4, seed=11)])

    r_dummy = [[True, True], [True, False]]
    p_dummy = [[True], [True]]

    with torch.no_grad():
        out, reaction_vectors = net(
            [reactant_slot0, reactant_slot1],
            [product_slot0],
            r_dummy,
            p_dummy,
            torch.device("cpu"),
        )

    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert len(reaction_vectors) == 2
    assert len(reaction_vectors[0]) == 2 * HIDDEN
