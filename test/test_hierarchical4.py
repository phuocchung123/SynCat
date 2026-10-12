"""
CPU tests for the four-level hierarchical reaction-yield model and its explicit
batch packing.

The tests build synthetic reaction graphs directly (155 node features / 9 edge
features, matching `preprocess_utils`) so that they do not depend on RDKit,
whose native libraries are blocked in some sandboxes. A single real-parser test
is included and skipped when RDKit cannot be imported; the structural string
validation is tested through `reaction_validation`, which has no RDKit import.
"""

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

from gin import propagate_gine  # noqa: E402
from hierarchical_data import (  # noqa: E402
    pack_hierarchical_batch,
    segment_mean,
)
from hierarchical_model import Hierarchical4YieldModel  # noqa: E402
from model_factory import checkpoint_config, model_from_config  # noqa: E402
from reaction_validation import (  # noqa: E402
    validate_reaction_structure,
    validate_reaction_strings,
)

NODE_DIM = 155
EDGE_DIM = 9
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------------
# synthetic graph fixtures
# ----------------------------------------------------------------------------


def _one_graph(n_atoms, edges, rng) -> Data:
    """One molecular graph with the repository's feature widths."""
    x = (rng.random((n_atoms, NODE_DIM)) > 0.5).astype(np.float32)
    if edges:
        src = np.array([u for u, v in edges] + [v for u, v in edges], dtype=np.int64)
        dst = np.array([v for u, v in edges] + [u for u, v in edges], dtype=np.int64)
    else:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = (rng.random((len(src), EDGE_DIM)) > 0.5).astype(np.float32)
    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(edge_attr),
    )


def _dummy_graph() -> Data:
    """One-atom, zero-feature padding graph, as `add_dummy` produces."""
    return Data(
        x=torch.zeros((1, NODE_DIM)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, EDGE_DIM)),
    )


def mol(n_atoms, edges=()):
    """A molecule spec: number of atoms and its (undirected) bond list."""
    return (n_atoms, list(edges))


def make_batch(reactions, seed=0):
    """
    Build a collated-style batch from explicit reaction specs.

    Parameters
    ----------
    reactions : list
        One ``(reactants, products)`` pair per reaction. Each side is a list of
        molecule specs (as returned by :func:`mol`); a ``None`` entry forces a
        padding slot.
    seed : int
        Seed of the feature RNG.

    Returns
    -------
    tuple
        (rmols, pmols, r_dummy, p_dummy) ready for the models.
    """
    rng = np.random.default_rng(seed)
    rmax = max(len(reactants) for reactants, _ in reactions)
    pmax = max(len(products) for _, products in reactions)

    rmols = []
    for j in range(rmax):
        graphs = []
        for reactants, _ in reactions:
            spec = reactants[j] if j < len(reactants) else None
            graphs.append(_one_graph(*spec, rng) if spec is not None else _dummy_graph())
        rmols.append(Batch.from_data_list(graphs))

    pmols = []
    for j in range(pmax):
        graphs = []
        for _, products in reactions:
            spec = products[j] if j < len(products) else None
            graphs.append(_one_graph(*spec, rng) if spec is not None else _dummy_graph())
        pmols.append(Batch.from_data_list(graphs))

    r_dummy = [
        [(j < len(reactants) and reactants[j] is not None) for j in range(rmax)]
        for reactants, _ in reactions
    ]
    p_dummy = [
        [(j < len(products) and products[j] is not None) for j in range(pmax)]
        for _, products in reactions
    ]
    return rmols, pmols, r_dummy, p_dummy


def _model(emb_dim=16, **kwargs):
    kwargs.setdefault("hierarchy_passes", 2)
    kwargs.setdefault("hierarchy_gine_layers", 2)
    return Hierarchical4YieldModel(
        NODE_DIM, EDGE_DIM, emb_dim, 0.0, **kwargs
    ).to(DEVICE)


# ----------------------------------------------------------------------------
# packing: counts, membership, edge containment
# ----------------------------------------------------------------------------


def test_pack_counts_match_the_real_parser_fixture():
    """Two reactions give 6 molecules, 13 atoms, 4 group nodes and 2 reactions."""
    reactions = [
        ([mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)])],
         [mol(3, [(0, 1), (1, 2)]), mol(1)]),
        ([mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    assert packed.n_molecules == 6
    assert packed.x.shape[0] == 13
    assert packed.side2rxn.numel() == 4
    assert packed.n_reactions == 2
    assert packed.atom2mol.shape == (13,)
    assert packed.mol2rxn.shape == (6,)
    assert packed.mol_side.shape == (6,)
    assert packed.mol2side.shape == (6,)


def test_pack_membership_is_consistent():
    """atom2mol, mol2rxn/mol_side, mol2side and side2rxn agree with each other."""
    reactions = [
        ([mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)])],
         [mol(3, [(0, 1), (1, 2)]), mol(1)]),
        ([mol(1)], [mol(2, [(0, 1)])]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    B, M = packed.n_reactions, packed.n_molecules
    assert torch.equal(packed.mol2side, 2 * packed.mol2rxn + packed.mol_side)
    assert torch.equal(
        packed.side2rxn, torch.arange(B).repeat_interleave(2)
    )
    assert torch.equal(packed.side_roles, torch.arange(2 * B) % 2)
    assert int(packed.mol2side.max()) == 2 * B - 1
    assert int(packed.atom2mol.min()) == 0
    assert int(packed.atom2mol.max()) == M - 1
    # every molecule has at least one atom
    counts = torch.bincount(packed.atom2mol, minlength=M)
    assert (counts > 0).all()
    # every reaction contributes a real molecule to both sides
    present = torch.zeros((B, 2), dtype=torch.bool)
    present[packed.mol2rxn, packed.mol_side] = True
    assert present.all()
    # chemical edges stay inside one molecule
    if packed.edge_index.numel():
        assert torch.equal(
            packed.atom2mol[packed.edge_index[0]],
            packed.atom2mol[packed.edge_index[1]],
        )


def test_variable_molecule_and_atom_counts():
    """Reactions with different molecule/atom counts pack independently."""
    reactions = [
        ([mol(1), mol(2, [(0, 1)]), mol(3, [(0, 1), (1, 2)])],
         [mol(4, [(0, 1), (1, 2), (2, 3)])]),
        ([mol(5, [(0, 1)])], [mol(1), mol(1)]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    # 3 + 1 reactant and 1 + 2 product molecules
    assert packed.n_molecules == 7
    # 1 + 2 + 3 + 4 + 5 + 1 + 1 atoms
    assert packed.x.shape[0] == 17
    counts = {}
    for r, s in zip(packed.mol2rxn.tolist(), packed.mol_side.tolist()):
        counts[(r, s)] = counts.get((r, s), 0) + 1
    assert counts == {(0, 0): 3, (0, 1): 1, (1, 0): 1, (1, 1): 2}


def test_multiple_products():
    """Two products in one reaction are two separate molecule nodes."""
    reactions = [([mol(2, [(0, 1)])], [mol(2, [(0, 1)]), mol(1)])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    assert packed.n_molecules == 3
    assert packed.n_reactions == 1
    assert packed.mol_side.tolist() == [0, 1, 1]
    # one group node per side
    assert packed.side2rxn.tolist() == [0, 0]


def test_salt_fragments_are_separate_molecules():
    """Dot-separated components become separate molecule graphs."""
    reactions = [([mol(3, [(0, 1), (1, 2)]), mol(1), mol(1)], [mol(3, [(0, 1), (1, 2)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    assert packed.n_molecules == 4
    assert packed.x.shape[0] == 3 + 1 + 1 + 3
    assert packed.mol_side.tolist() == [0, 0, 0, 1]


def test_repeated_identical_molecules_get_distinct_ids():
    """Two occurrences of the same molecule keep distinct molecule ids."""
    reactions = [([mol(2, [(0, 1)]), mol(2, [(0, 1)])], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    assert packed.n_molecules == 3
    mol_of_first = set(packed.atom2mol[:2].tolist())
    mol_of_second = set(packed.atom2mol[2:4].tolist())
    assert mol_of_first.isdisjoint(mol_of_second)


def test_padding_atoms_and_edges_are_excluded():
    """Non-zero padding atoms/edges never enter the packed graph."""
    reactions = [([mol(2, [(0, 1)]), None], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    # Make the padding graph loudly non-zero (features and a self-loop edge).
    rmols[1].x = rmols[1].x + 42.0
    rmols[1].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    rmols[1].edge_attr = rmols[1].edge_attr + 17.0

    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert packed.x.shape[0] == 4  # 2 reactant + 2 product real atoms only
    assert packed.n_molecules == 2
    assert not torch.equal(packed.edge_attr, torch.full_like(packed.edge_attr, 17.0))


def test_padding_noise_does_not_change_prediction():
    """A model prediction is invariant to noise in a padding slot."""
    net = _model().eval()
    reactions = [([mol(2, [(0, 1)]), None], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)

    with torch.no_grad():
        clean, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    noisy_rmols = [r for r in rmols]
    noisy_rmols[1].x = noisy_rmols[1].x + 3.0
    noisy_rmols[1].edge_attr = noisy_rmols[1].edge_attr + 1.0
    with torch.no_grad():
        noisy, _ = net(noisy_rmols, pmols, r_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(noisy, clean, atol=1e-6, rtol=1e-6)


def test_cross_molecule_edge_is_rejected():
    """An edge connecting two molecules fails packing explicitly."""
    reactions = [
        ([mol(2, [(0, 1)]), mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
        ([mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    # slot 0 holds reaction 0's and reaction 1's molecule (atoms 0,1 and 2,3);
    # connect atom 0 of reaction 0 to atom 2 of reaction 1 across graphs.
    rmols[0].edge_index = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="different molecules"):
        pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)


def test_empty_side_is_rejected():
    """A reaction with no real molecule on a side fails with a clear error."""
    reactions = [([None], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    with pytest.raises(ValueError, match="Reactants"):
        pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)


def test_segment_mean_rejects_empty_segment():
    """segment_mean raises instead of dividing by a zero count."""
    x = torch.randn(3, 4)
    index = torch.tensor([0, 0, 2])
    with pytest.raises(ValueError, match="empty segment"):
        segment_mean(x, index, 3)
    # a fully covered split is a plain mean
    out = segment_mean(x, torch.tensor([0, 1, 1]), 2)
    torch.testing.assert_close(out[0], x[0])
    torch.testing.assert_close(out[1], (x[1] + x[2]) / 2)


# ----------------------------------------------------------------------------
# model shapes, losses, invariances
# ----------------------------------------------------------------------------


def _two_reaction_batch(seed=0):
    reactions = [
        ([mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)])],
         [mol(3, [(0, 1), (1, 2)])]),
        ([mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
    ]
    return make_batch(reactions, seed=seed)


def test_forward_shapes_and_huber_loss():
    """Predictions are [B], finite, and compatible with HuberLoss."""
    net = _model().eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    with torch.no_grad():
        pred, emb = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()
    assert len(emb) == 2 and len(emb[0]) == 16

    labels = torch.tensor([0.5, 0.25], dtype=torch.float32)
    loss = torch.nn.HuberLoss()(pred, labels)
    assert torch.isfinite(loss)
    assert pred.shape == labels.shape


def test_batch_size_one():
    """A single reaction returns a [1] prediction."""
    net = _model().eval()
    reactions = [([mol(2, [(0, 1)])], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    with torch.no_grad():
        pred, emb = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert pred.shape == (1,)
    assert torch.isfinite(pred).all()
    assert len(emb) == 1


def test_isolated_atom_and_zero_bond_molecule():
    """A one-atom molecule with no bonds still takes part in every level."""
    net = _model().eval()
    reactions = [([mol(1), mol(2, [(0, 1)])], [mol(1), mol(1)])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    # the isolated atoms contribute edges only within the other molecule
    assert packed.n_molecules == 4
    with torch.no_grad():
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert pred.shape == (1,)
    assert torch.isfinite(pred).all()


def test_reactant_permutation_invariance():
    """Permuting reactant slots (with r_dummy) leaves the prediction unchanged."""
    net = _model().eval()
    reactions = [([mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)]), mol(1)],
                  [mol(3, [(0, 1), (1, 2)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    with torch.no_grad():
        base, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    perm = [1, 2, 0]
    rmols_perm = [rmols[i] for i in perm]
    r_dummy_perm = [[d[i] for i in perm] for d in r_dummy]
    with torch.no_grad():
        permuted, _ = net(rmols_perm, pmols, r_dummy_perm, p_dummy, DEVICE)
    torch.testing.assert_close(permuted, base, atol=1e-6, rtol=1e-6)


def test_product_permutation_invariance():
    """Permuting product slots (with p_dummy) leaves the prediction unchanged."""
    net = _model().eval()
    reactions = [([mol(2, [(0, 1)])],
                  [mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)]), mol(1)])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    with torch.no_grad():
        base, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    perm = [2, 0, 1]
    pmols_perm = [pmols[i] for i in perm]
    p_dummy_perm = [[d[i] for i in perm] for d in p_dummy]
    with torch.no_grad():
        permuted, _ = net(rmols, pmols_perm, r_dummy, p_dummy_perm, DEVICE)
    torch.testing.assert_close(permuted, base, atol=1e-6, rtol=1e-6)


def test_batch_independence():
    """Changing one reaction must not change another reaction's prediction."""
    net = _model().eval()
    reactions = [
        ([mol(3, [(0, 1), (1, 2)])], [mol(3, [(0, 1), (1, 2)])]),
        ([mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    with torch.no_grad():
        base, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    modified = list(rmols)
    modified[0] = Batch.from_data_list(
        [
            rmols[0][0],  # keep reaction 0 unchanged
            _one_graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)], np.random.default_rng(7)),
        ]
    )
    with torch.no_grad():
        changed, _ = net(modified, pmols, r_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(changed[0], base[0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(changed[1], base[1], atol=1e-6)


# ----------------------------------------------------------------------------
# feedback wiring
# ----------------------------------------------------------------------------


def test_second_pass_receives_feedback_updated_atoms():
    """Pass 2 starts from the feedback-updated atom states, not the bond-only ones."""
    net = _model().eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()

    captured = {}

    def hook(module, inputs):
        captured["h"] = inputs[0].detach().clone()

    handle = net.gine_stacks[1][0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    finally:
        handle.remove()

    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    h0 = net.atom_encoder(packed.x)
    e = net.bond_encoder(packed.edge_attr)
    bonds_only = propagate_gine(
        net.gine_stacks[0], h0, packed.edge_index, e, net.dropout
    )
    assert "h" in captured
    assert not torch.allclose(captured["h"], bonds_only, atol=1e-7)


def test_no_final_feedback_stage():
    """K passes use exactly K - 1 downward feedback sets."""
    for K in (1, 2, 3):
        net = _model(hierarchy_passes=K)
        expected = max(0, K - 1)
        assert len(net.reaction_to_side) == expected
        assert len(net.ln_s_down) == expected
        assert len(net.side_to_molecule) == expected
        assert len(net.ln_m_down) == expected
        assert len(net.molecule_to_atom) == expected
        assert len(net.ln_h_down) == expected


def test_no_feedback_modules_for_k1():
    """K=1 is the collect-only control and allocates no downward parameter."""
    net = _model(hierarchy_passes=1)
    assert len(net.reaction_to_side) == 0
    assert not any("_down" in name for name in net.state_dict())
    assert not any("reaction_to_side" in name for name in net.state_dict())


def test_feedback_disabled_matched_weights_control():
    """The no-feedback model shares all weights and skips the downward updates."""
    torch.manual_seed(0)
    fb = _model(hierarchy_passes=2, hierarchy_no_feedback=False)
    torch.manual_seed(0)
    nf = _model(hierarchy_passes=2, hierarchy_no_feedback=True)

    shared = {
        k: v
        for k, v in fb.state_dict().items()
        if not any(tag in k for tag in ("_down", "reaction_to_side", "side_to_molecule", "molecule_to_atom"))
    }
    for key, value in shared.items():
        assert key in nf.state_dict()
        torch.testing.assert_close(nf.state_dict()[key], value)

    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    h0 = nf.atom_encoder(packed.x)
    e = nf.bond_encoder(packed.edge_attr)
    bonds_only = propagate_gine(nf.gine_stacks[0], h0, packed.edge_index, e, nf.dropout)

    captured = {}

    def hook(module, inputs):
        captured["h"] = inputs[0].detach().clone()

    handle = nf.gine_stacks[1][0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            nf(rmols, pmols, r_dummy, p_dummy, DEVICE)
    finally:
        handle.remove()
    # without feedback, pass 2 sees exactly the bond-only propagation of pass 1
    torch.testing.assert_close(captured["h"], bonds_only, atol=1e-6, rtol=1e-6)


def test_finite_gradients_and_feedback_gradient_signal():
    """Every used feedback block gets a finite, non-zero gradient for K=2."""
    net = _model(hierarchy_passes=2)
    net.train()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    torch.nn.HuberLoss()(pred, torch.zeros_like(pred)).backward()

    blocks = {
        "reaction_to_side": net.reaction_to_side[0],
        "side_to_molecule": net.side_to_molecule[0],
        "molecule_to_atom": net.molecule_to_atom[0],
        "ln_s_down": net.ln_s_down[0],
        "ln_m_down": net.ln_m_down[0],
        "ln_h_down": net.ln_h_down[0],
    }
    for name, module in blocks.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, "%s received no gradient" % name
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(bool((g != 0).any()) for g in grads), name

    # and the upstream/forward parameters are finite too
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ----------------------------------------------------------------------------
# checkpoint reconstruction and tiny training experiment
# ----------------------------------------------------------------------------


def test_hierarchical_checkpoint_reconstruction_and_prediction():
    """model_from_config rebuilds a hierarchical checkpoint exactly."""
    torch.manual_seed(1)
    net = _model().eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    with torch.no_grad():
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    config = checkpoint_config(net)
    assert config["yield_arch"] == "hierarchical4"

    rebuilt = model_from_config(config)
    rebuilt.load_state_dict(net.state_dict())
    rebuilt.eval()
    with torch.no_grad():
        pred2, _ = rebuilt(rmols, pmols, r_dummy, p_dummy, DEVICE)
    torch.testing.assert_close(pred2, pred, atol=1e-6, rtol=1e-6)


def test_deterministic_tiny_training_reduces_loss():
    """A handful of updates on a fixed batch reduces the Huber loss."""
    torch.manual_seed(3)
    net = _model(emb_dim=16, hierarchy_passes=1, hierarchy_gine_layers=1)
    net.train()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    labels = torch.tensor([0.5, 0.25], dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    loss_fn = torch.nn.HuberLoss()

    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


# ----------------------------------------------------------------------------
# structural input validation (no RDKit import needed)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rsmi",
    [
        "CCO>>CCN>>CO",       # more than one separator
        "CCO.CCN",            # no separator
        ">>CCN",              # empty reactant side
        "CCO>>",              # empty product side
        "CCO.>>CCN",          # empty reactant component
        "CCO>>CCN.",          # empty product component
    ],
)
def test_malformed_reaction_strings_are_rejected(rsmi):
    with pytest.raises(ValueError):
        validate_reaction_structure(rsmi)


def test_valid_reaction_strings_pass():
    validate_reaction_structure("CCO.CN>>CCN.O")
    validate_reaction_structure("CC>>CO")
    validate_reaction_strings(["CCO.CN>>CCN.O", "CC>>CO"])


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown yield architecture"):
        model_from_config({"yield_arch": "does_not_exist"})


@pytest.mark.parametrize(
    "kwargs",
    [{"hierarchy_passes": 0}, {"hierarchy_gine_layers": 0}, {"hierarchy_passes": -1}],
)
def test_hierarchical_rejects_non_positive_counts(kwargs):
    """Pass and GINE-layer counts must be positive."""
    with pytest.raises(ValueError):
        Hierarchical4YieldModel(NODE_DIM, EDGE_DIM, 16, 0.0, **kwargs)


# ----------------------------------------------------------------------------
# real-parser fixture (skipped when RDKit cannot be imported)
# ----------------------------------------------------------------------------


def test_real_parser_fixture_counts():
    """The documented real-parser fixture packs 6 molecules and 13 atoms."""
    try:
        import rdkit.Chem  # noqa: F401
    except ImportError as e:  # pragma: no cover - sandbox dependent
        pytest.skip("rdkit.Chem is not importable: %s" % e)
    from data import GraphDataset
    from reaction_data import get_graph_data
    from utils import collate_reaction_graphs

    reactions = ["CCO.CN>>CCN.O", "CC>>CO"]
    rmol, pmol, reaction = get_graph_data(reactions, 2, 2, y_list=[1.0, 1.0])
    dataset = GraphDataset(rmol=rmol, pmol=pmol, reaction=reaction)
    batch = collate_reaction_graphs([dataset[i] for i in range(2)])
    rmols = list(batch[:2])
    pmols = list(batch[2:4])
    r_dummy = batch[-4]
    p_dummy = batch[-3]

    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert packed.n_molecules == 6
    assert packed.x.shape[0] == 13
    assert packed.side2rxn.numel() == 4
    assert packed.n_reactions == 2
