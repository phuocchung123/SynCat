"""
CPU tests for the three-level hierarchical reaction-yield model
(atoms -> molecules -> Reactants/Products groups) and its heads.

The synthetic graph fixtures are shared with `test_hierarchical4`; these tests
therefore do not depend on RDKit, whose native libraries are blocked in some
sandboxes.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_DIR, TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from finetune import _load_checkpoint_safely  # noqa: E402
from gin import propagate_gine  # noqa: E402
from hierarchical_data import pack_hierarchical_batch  # noqa: E402
from hierarchical_model import (  # noqa: E402
    Hierarchical3YieldModel,
    Hierarchical4YieldModel,
)
from model import HEADS  # noqa: E402
from model_factory import (  # noqa: E402
    build_yield_model,
    checkpoint_config,
    is_hierarchical_arch,
    model_from_config,
)
from test_hierarchical4 import (  # noqa: E402
    DEVICE,
    EDGE_DIM,
    NODE_DIM,
    _one_graph,
    _two_reaction_batch,
    make_batch,
    mol,
)

REACTION_LEVEL_TAGS = (
    "reaction_token",
    "side_to_reaction",
    "ln_r_up",
    "reaction_to_side",
    "ln_s_down",
)


def _model(head="linear", emb_dim=16, **kwargs):
    kwargs.setdefault("hierarchy_passes", 2)
    kwargs.setdefault("hierarchy_gine_layers", 2)
    return Hierarchical3YieldModel(
        NODE_DIM, EDGE_DIM, emb_dim, 0.0, head=head, **kwargs
    ).to(DEVICE)


def _forward(net, batch):
    rmols, pmols, r_dummy, p_dummy = batch
    with torch.no_grad():
        return net(rmols, pmols, r_dummy, p_dummy, DEVICE)


# ----------------------------------------------------------------------------
# level removal: no reaction node / reaction-level parameters
# ----------------------------------------------------------------------------


def test_no_reaction_level_parameters():
    """No reaction token or reaction-level update module remains."""
    for head in HEADS:
        net = _model(head=head)
        assert not hasattr(net, "reaction_token")
        assert not hasattr(net, "side_to_reaction")
        assert not hasattr(net, "ln_r_up")
        assert not hasattr(net, "reaction_to_side")
        assert not hasattr(net, "ln_s_down")
        keys = list(net.state_dict())
        assert not any(
            tag in key for key in keys for tag in REACTION_LEVEL_TAGS
        )


def test_two_group_states_per_reaction_and_readout_shape():
    """Exactly two group states per reaction and a [B, 2D] readout."""
    net = _model(emb_dim=16).eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    captured = {}

    def s_hook(module, inputs, output):
        captured["s"] = output.detach().clone()

    handle = net.ln_s_up[-1].register_forward_hook(s_hook)
    try:
        pred, emb = _forward(net, (rmols, pmols, r_dummy, p_dummy))
    finally:
        handle.remove()

    assert captured["s"].shape == (2 * packed.n_reactions, 16)
    assert pred.shape == (packed.n_reactions,)
    assert np.asarray(emb).shape == (packed.n_reactions, 2 * 16)


def test_concatenation_order_is_reactants_then_products():
    """The readout is exactly [Reactants group, Products group]."""
    net = _model(emb_dim=16).eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)

    captured = {}

    def s_hook(module, inputs, output):
        captured["s"] = output.detach().clone()

    def pv_hook(module, inputs):
        captured["pv"] = inputs[0].detach().clone()

    h1 = net.ln_s_up[-1].register_forward_hook(s_hook)
    h2 = net.regressor.register_forward_pre_hook(pv_hook)
    try:
        pred, emb = _forward(net, (rmols, pmols, r_dummy, p_dummy))
    finally:
        h1.remove()
        h2.remove()

    B = packed.n_reactions
    D = net.emb_dim
    pairs = captured["s"].reshape(B, 2, D)
    expected = torch.cat((pairs[:, 0], pairs[:, 1]), dim=-1)
    torch.testing.assert_close(captured["pv"], expected)
    torch.testing.assert_close(torch.tensor(emb), captured["pv"])


# ----------------------------------------------------------------------------
# heads: definitions, shapes, loss, gradients
# ----------------------------------------------------------------------------


def test_linear_head_definition():
    net = _model(head="linear", emb_dim=16)
    assert isinstance(net.regressor, nn.Linear)
    assert net.regressor.in_features == 32
    assert net.regressor.out_features == 1
    assert list(net.regressor.state_dict()) == ["weight", "bias"]
    # the MLP head was not allocated
    assert not any(k.startswith("regressor.0.") for k in net.state_dict())


def test_mlp_head_definition():
    net = _model(head="mlp", emb_dim=16)
    assert isinstance(net.regressor, nn.Sequential)
    assert len(net.regressor) == 4
    l1, act, drop, l2 = net.regressor
    assert isinstance(l1, nn.Linear) and l1.in_features == 32 and l1.out_features == 16
    assert isinstance(act, nn.ReLU)
    assert isinstance(drop, nn.Dropout)
    assert isinstance(l2, nn.Linear) and l2.in_features == 16 and l2.out_features == 1
    # the linear head was not allocated
    assert not (isinstance(net.regressor, nn.Linear))


def test_both_heads_finite_prediction_loss_and_backprop():
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    labels = torch.tensor([0.5, 0.25], dtype=torch.float32)
    for head in HEADS:
        net = _model(head=head)
        net.train()
        pred, emb = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        loss = torch.nn.HuberLoss()(pred, labels)
        assert torch.isfinite(loss)
        loss.backward()
        head_params = list(net.regressor.parameters())
        assert head_params
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head_params)


def test_head_selection_through_factory():
    import argparse

    base = dict(
        emb_dim=16,
        dropout=0.0,
        hierarchy_passes=2,
        hierarchy_gine_layers=2,
        hierarchy_no_feedback=False,
        yield_arch="hierarchical3",
    )
    linear = build_yield_model(argparse.Namespace(head="linear", **base), NODE_DIM, EDGE_DIM)
    mlp = build_yield_model(argparse.Namespace(head="mlp", **base), NODE_DIM, EDGE_DIM)
    assert linear.head == "linear" and isinstance(linear.regressor, nn.Linear)
    assert mlp.head == "mlp" and isinstance(mlp.regressor, nn.Sequential)


def test_encoder_identical_between_heads():
    """Only the head differs: both heads share an identical encoder/backbone."""
    torch.manual_seed(7)
    linear = _model(head="linear")
    torch.manual_seed(7)
    mlp = _model(head="mlp")

    linear_state = linear.state_dict()
    mlp_state = mlp.state_dict()
    backbone = [k for k in linear_state if not k.startswith("regressor.")]
    assert backbone
    for key in backbone:
        torch.testing.assert_close(mlp_state[key], linear_state[key])
    # the head itself is the only thing with different keys/weights
    assert set(k for k in linear_state if k.startswith("regressor.")) != set(
        k for k in mlp_state if k.startswith("regressor.")
    )


def test_mlp_head_parameter_count_above_linear():
    linear = sum(p.numel() for p in _model(head="linear").parameters())
    mlp = sum(p.numel() for p in _model(head="mlp").parameters())
    assert mlp > linear


# ----------------------------------------------------------------------------
# packing / invariance / exclusion
# ----------------------------------------------------------------------------


def test_variable_molecule_counts_and_multiple_products():
    net = _model().eval()
    reactions = [
        ([mol(1), mol(2, [(0, 1)]), mol(3, [(0, 1), (1, 2)])],
         [mol(4, [(0, 1), (1, 2), (2, 3)]), mol(1)]),
        ([mol(5, [(0, 1)])], [mol(1), mol(1)]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert packed.side2rxn.numel() == 2 * packed.n_reactions
    pred, emb = _forward(net, (rmols, pmols, r_dummy, p_dummy))
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()
    assert np.asarray(emb).shape == (2, 32)


def test_nonzero_padding_graphs_are_excluded():
    net = _model().eval()
    reactions = [([mol(2, [(0, 1)]), None], [mol(2, [(0, 1)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    clean, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))

    rmols[1].x = rmols[1].x + 42.0
    rmols[1].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    rmols[1].edge_attr = rmols[1].edge_attr + 17.0
    noisy, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))
    torch.testing.assert_close(noisy, clean, atol=1e-6, rtol=1e-6)


def test_reactant_permutation_invariance():
    net = _model().eval()
    reactions = [([mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)]), mol(1)],
                  [mol(3, [(0, 1), (1, 2)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    base, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))

    perm = [1, 2, 0]
    rmols_perm = [rmols[i] for i in perm]
    r_dummy_perm = [[d[i] for i in perm] for d in r_dummy]
    permuted, _ = _forward(net, (rmols_perm, pmols, r_dummy_perm, p_dummy))
    torch.testing.assert_close(permuted, base, atol=1e-6, rtol=1e-6)


def test_product_permutation_invariance():
    net = _model().eval()
    reactions = [([mol(2, [(0, 1)])],
                  [mol(3, [(0, 1), (1, 2)]), mol(2, [(0, 1)]), mol(1)])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    base, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))

    perm = [2, 0, 1]
    pmols_perm = [pmols[i] for i in perm]
    p_dummy_perm = [[d[i] for i in perm] for d in p_dummy]
    permuted, _ = _forward(net, (rmols, pmols_perm, r_dummy, p_dummy_perm))
    torch.testing.assert_close(permuted, base, atol=1e-6, rtol=1e-6)


def test_batch_independence():
    """Changing one reaction must not change another reaction's prediction."""
    net = _model().eval()
    reactions = [
        ([mol(3, [(0, 1), (1, 2)])], [mol(3, [(0, 1), (1, 2)])]),
        ([mol(2, [(0, 1)])], [mol(2, [(0, 1)])]),
    ]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    base, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))

    modified = list(rmols)
    modified[0] = Batch.from_data_list(
        [
            rmols[0][0],
            _one_graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)], np.random.default_rng(7)),
        ]
    )
    changed, _ = _forward(net, (modified, pmols, r_dummy, p_dummy))
    torch.testing.assert_close(changed[0], base[0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(changed[1], base[1], atol=1e-6)


def test_no_cross_side_messages():
    """Reactants and Products reach each other only through the head."""
    net = _model(emb_dim=16).eval()
    reactions = [([mol(2, [(0, 1)])], [mol(3, [(0, 1), (1, 2)])])]
    rmols, pmols, r_dummy, p_dummy = make_batch(reactions)
    base, base_emb = _forward(net, (rmols, pmols, r_dummy, p_dummy))
    base_emb = torch.tensor(base_emb)

    D = net.emb_dim
    # change only the products: the Reactants half must not move
    pmols2 = list(pmols)
    pmols2[0] = Batch.from_data_list(
        [_one_graph(4, [(0, 1), (1, 2), (2, 3)], np.random.default_rng(11))]
    )
    _, prod_emb = _forward(net, (rmols, pmols2, r_dummy, p_dummy))
    torch.testing.assert_close(torch.tensor(prod_emb)[:, :D], base_emb[:, :D], atol=1e-6, rtol=1e-6)

    # change only the reactants: the Products half must not move
    rmols2 = list(rmols)
    rmols2[0] = Batch.from_data_list(
        [_one_graph(4, [(0, 1), (1, 2), (2, 3)], np.random.default_rng(13))]
    )
    _, react_emb = _forward(net, (rmols2, pmols, r_dummy, p_dummy))
    torch.testing.assert_close(torch.tensor(react_emb)[:, D:], base_emb[:, D:], atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------
# feedback wiring
# ----------------------------------------------------------------------------


def test_pass2_consumes_feedback_updated_atoms():
    net = _model().eval()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()

    captured = {}

    def hook(module, inputs):
        captured["h"] = inputs[0].detach().clone()

    handle = net.gine_stacks[1][0].register_forward_pre_hook(hook)
    try:
        _forward(net, (rmols, pmols, r_dummy, p_dummy))
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
    """K passes use exactly K - 1 feedback sets of the two remaining steps."""
    for K in (1, 2, 3):
        net = _model(hierarchy_passes=K)
        expected = max(0, K - 1)
        assert len(net.side_to_molecule) == expected
        assert len(net.ln_m_down) == expected
        assert len(net.molecule_to_atom) == expected
        assert len(net.ln_h_down) == expected


def test_k1_has_no_feedback_modules():
    net = _model(hierarchy_passes=1)
    assert len(net.side_to_molecule) == 0
    assert len(net.molecule_to_atom) == 0
    keys = list(net.state_dict())
    assert not any("_down" in k for k in keys)
    assert not any("side_to_molecule" in k for k in keys)
    assert not any("molecule_to_atom" in k for k in keys)


def test_used_feedback_blocks_receive_gradient():
    net = _model().train()
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    torch.nn.HuberLoss()(pred, torch.zeros_like(pred)).backward()

    blocks = {
        "side_to_molecule": net.side_to_molecule[0],
        "molecule_to_atom": net.molecule_to_atom[0],
        "ln_m_down": net.ln_m_down[0],
        "ln_h_down": net.ln_h_down[0],
    }
    for name, module in blocks.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, "%s received no gradient" % name
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(bool((g != 0).any()) for g in grads), name
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_feedback_disabled_matched_weights_control():
    """With matched backbone weights, feedback-disabled pass 2 equals the
    bond-only propagation of pass 1, while the feedback model differs."""
    torch.manual_seed(0)
    fb = _model(hierarchy_passes=2, hierarchy_no_feedback=False).eval()
    torch.manual_seed(0)
    nf = _model(hierarchy_passes=2, hierarchy_no_feedback=True).eval()

    # copy every shared weight so the two models have identical encoders/head
    shared = {k: v for k, v in fb.state_dict().items() if k in nf.state_dict()}
    nf.load_state_dict(shared, strict=False)

    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    packed = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, DEVICE)
    h0 = nf.atom_encoder(packed.x)
    e = nf.bond_encoder(packed.edge_attr)
    bonds_only = propagate_gine(nf.gine_stacks[0], h0, packed.edge_index, e, nf.dropout)

    def capture_h2(net):
        captured = {}

        def hook(module, inputs):
            captured["h"] = inputs[0].detach().clone()

        handle = net.gine_stacks[1][0].register_forward_pre_hook(hook)
        try:
            _forward(net, (rmols, pmols, r_dummy, p_dummy))
        finally:
            handle.remove()
        return captured["h"]

    torch.testing.assert_close(capture_h2(nf), bonds_only, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(capture_h2(fb), bonds_only, atol=1e-7)


# ----------------------------------------------------------------------------
# checkpoint reconstruction and tiny training experiment
# ----------------------------------------------------------------------------


def test_checkpoint_round_trip_both_heads():
    rmols, pmols, r_dummy, p_dummy = _two_reaction_batch()
    for head in HEADS:
        torch.manual_seed(1)
        net = _model(head=head).eval()
        pred, _ = _forward(net, (rmols, pmols, r_dummy, p_dummy))

        config = checkpoint_config(net)
        assert config["yield_arch"] == "hierarchical3"
        assert config["head"] == head
        assert config["emb_dim"] == 16
        assert config["hierarchy_passes"] == 2
        assert config["hierarchy_gine_layers"] == 2
        assert config["hierarchy_no_feedback"] is False

        rebuilt = model_from_config(config)
        assert isinstance(rebuilt, Hierarchical3YieldModel)
        assert rebuilt.head == head
        rebuilt.load_state_dict(net.state_dict())
        rebuilt.eval()
        pred2, _ = _forward(rebuilt, (rmols, pmols, r_dummy, p_dummy))
        torch.testing.assert_close(pred2, pred, atol=1e-6, rtol=1e-6)


def test_hierarchical4_is_not_reinterpreted_as_hierarchical3():
    """hierarchical4 checkpoints keep selecting the four-level model."""
    import argparse

    h4 = build_yield_model(
        argparse.Namespace(
            emb_dim=16, dropout=0.0, hierarchy_passes=2, hierarchy_gine_layers=2,
            hierarchy_no_feedback=False, yield_arch="hierarchical4",
        ),
        NODE_DIM,
        EDGE_DIM,
    )
    rebuilt = model_from_config(checkpoint_config(h4))
    assert isinstance(rebuilt, Hierarchical4YieldModel)
    assert not isinstance(rebuilt, Hierarchical3YieldModel)

    # cross-architecture loading is rejected explicitly
    logger = __import__("logging").getLogger("test_h3")
    h3 = _model()
    h4_ckpt = {"model_config": checkpoint_config(h4), "model_state_dict": h4.state_dict()}
    h3_ckpt = {"model_config": checkpoint_config(h3), "model_state_dict": h3.state_dict()}
    with pytest.raises(RuntimeError, match="architecture"):
        _load_checkpoint_safely(h3, h4_ckpt, logger)
    with pytest.raises(RuntimeError, match="architecture"):
        _load_checkpoint_safely(h4, h3_ckpt, logger)


def test_head_mismatch_is_a_warm_start_not_a_hard_error():
    """A different head reloads as an explicit warm-start, as for legacy."""
    logger = __import__("logging").getLogger("test_h3")
    linear = _model(head="linear")
    ckpt = {"model_config": checkpoint_config(linear), "model_state_dict": linear.state_dict()}
    mlp = _model(head="mlp")
    # must not raise: only regressor.* keys differ
    _load_checkpoint_safely(mlp, ckpt, logger)


def test_hierarchical3_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Hierarchical3YieldModel(NODE_DIM, EDGE_DIM, 16, 0.0, hierarchy_passes=0)
    with pytest.raises(ValueError):
        Hierarchical3YieldModel(NODE_DIM, EDGE_DIM, 16, 0.0, hierarchy_gine_layers=0)
    with pytest.raises(ValueError, match="head"):
        Hierarchical3YieldModel(NODE_DIM, EDGE_DIM, 16, 0.0, head="nope")


def _training_loss(head):
    torch.manual_seed(3)
    net = _model(head=head, emb_dim=16, hierarchy_passes=1, hierarchy_gine_layers=1)
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
    return losses


@pytest.mark.parametrize("head", HEADS)
def test_deterministic_tiny_training_reduces_loss(head):
    losses = _training_loss(head)
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_is_hierarchical_arch_helper():
    assert is_hierarchical_arch("hierarchical3")
    assert is_hierarchical_arch("hierarchical4")
    assert not is_hierarchical_arch("legacy")
    assert not is_hierarchical_arch(None)
