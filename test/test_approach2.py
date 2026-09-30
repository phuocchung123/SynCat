"""
Tests for SynCat approach 2: the optional pairwise reactant tokens
(`reactant_tokens="comb"`) and the optional MLP regression head
(`head="mlp"`), both disabled by default.

This machine's Windows Application Control policy blocks several native
libraries (rdkit's ``rdBase``, scikit-learn's ``_sorting``), so ``rdkit.rdBase``
-- the only rdkit symbol ``utils`` imports, for log control -- is stubbed when
the real module cannot be loaded. Chemistry is never stubbed: the
real-featurization test needs genuine rdkit and skips with an explicit message,
and any test whose import chain hits a blocked library skips the same way.
"""

import logging
import os
import sys
import types

import numpy as np
import pytest
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# rdkit: real if possible, log-only stub otherwise
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment dependent
    from rdkit import rdBase as _real_rdBase  # noqa: F401

    HAS_RDKIT = True
except Exception:  # pragma: no cover - environment dependent
    HAS_RDKIT = False

from data import GraphDataset  # noqa: E402
from model import HEADS, REACTANT_TOKENS, model  # noqa: E402

torch.manual_seed(0)
DEVICE = torch.device("cpu")


def _install_rdbase_stub():
    """
    Make ``utils`` importable when only rdkit's native library is blocked.

    ``utils`` imports ``rdkit.rdBase`` solely for log control; the stub provides
    that API and nothing else, so chemistry still requires genuine rdkit. It is
    installed lazily (when a test actually needs ``utils``) so that collection
    of the other test modules is not affected.
    """
    if HAS_RDKIT:
        return
    if "rdkit.rdBase" not in sys.modules:
        stub = types.ModuleType("rdkit")
        rd_base = types.ModuleType("rdkit.rdBase")
        rd_base.DisableLog = lambda *args, **kwargs: None
        rd_base.EnableLog = lambda *args, **kwargs: None
        stub.rdBase = rd_base
        sys.modules.setdefault("rdkit", stub)
        sys.modules.setdefault("rdkit.rdBase", rd_base)


def _collate_reaction_graphs():
    """The repository's collate function, or a skip if the environment blocks it."""
    _install_rdbase_stub()
    try:
        from utils import collate_reaction_graphs

        return collate_reaction_graphs
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip("utils import blocked by the environment: %s" % exc)


# 3 real Suzuki reactions are enough; the prepared graphs are the output of the
# repository's own featurizer, so no rdkit chemistry is needed to load them.
SUZUKI_TEST_NPZ = os.path.join(
    ROOT_DIR, "Data", "processed", "suzuki", "npz", "split_0", "test.npz"
)


def _require_utils():
    _collate_reaction_graphs()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiny_model(emb_dim=8, node_in_feats=4, edge_in_feats=3, **kwargs):
    """A one-layer GIN model; only the architecture under test matters here."""
    return model(
        node_in_feats=node_in_feats,
        edge_in_feats=edge_in_feats,
        num_layer=1,
        emb_dim=emb_dim,
        drop_ratio=0.0,
        **kwargs,
    )


def _subset_mol(mol, positions):
    """Slice a mol dict to `positions` (mirrors suzuki_splits._subset_mol_dict)."""
    n_csum = np.concatenate([[0], np.cumsum(mol["n_node"])])
    e_csum = np.concatenate([[0], np.cumsum(mol["n_edge"])])
    node_rows = np.concatenate(
        [np.arange(n_csum[p], n_csum[p + 1]) for p in positions]
    ).astype(int)
    edge_rows = np.concatenate(
        [np.arange(e_csum[p], e_csum[p + 1]) for p in positions]
    ).astype(int)
    return {
        "n_node": mol["n_node"][positions],
        "n_edge": mol["n_edge"][positions],
        "dummy": [mol["dummy"][p] for p in positions],
        "node_attr": mol["node_attr"][node_rows],
        "edge_attr": mol["edge_attr"][edge_rows],
        "src": mol["src"][edge_rows],
        "dst": mol["dst"][edge_rows],
    }


def _tiny_prepared_dataset(n=3):
    """A GraphDataset of `n` real featurized reactions from the prepared npz."""
    with np.load(SUZUKI_TEST_NPZ, allow_pickle=True) as npz:
        rmol = list(npz["rmol"])
        pmol = list(npz["pmol"])
        reaction = npz["reaction"].item()
    positions = np.arange(n)
    tiny_rmol = [_subset_mol(m, positions) for m in rmol]
    tiny_pmol = [_subset_mol(m, positions) for m in pmol]
    tiny_reaction = {
        "y": np.asarray(reaction["y"])[positions],
        "rsmi": [reaction["rsmi"][p] for p in positions],
    }
    return GraphDataset(rmol=tiny_rmol, pmol=tiny_pmol, reaction=tiny_reaction)


def _batch(dataset, batch_size=2):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_reaction_graphs(),
        num_workers=0,
    )
    return next(iter(loader))


def _split_batch(batch, rmol_max_cnt, pmol_max_cnt):
    return (
        batch[:rmol_max_cnt],
        batch[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt],
        batch[-4],
        batch[-3],
        batch[-2],
    )


def _dummy_slot_batch(batch_size, node_dim, edge_dim, noisy=False):
    """One extra reactant slot per sample: a dummy graph (optionally noisy)."""
    from torch_geometric.data import Batch, Data

    if noisy:
        x = torch.randn(1, node_dim)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.randn(1, edge_dim)
    else:
        x = torch.zeros(1, node_dim)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, edge_dim)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return Batch.from_data_list([data.clone() for _ in range(batch_size)])


# ---------------------------------------------------------------------------
# _add_pair_tokens
# ---------------------------------------------------------------------------


def test_pair_tokens_shape_and_order():
    net = _tiny_model()
    tokens = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)

    expanded, expanded_mask, slots = net._add_pair_tokens(tokens, mask)

    assert expanded.shape == (2, 6, 8)
    assert expanded_mask.shape == (2, 6)
    assert slots == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


def test_pair_tokens_values_are_slot_sums():
    net = _tiny_model()
    tokens = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    expanded, _, slots = net._add_pair_tokens(tokens, mask)

    # individual slots first
    for k in range(4):
        assert slots[k] == (k,)
        assert torch.equal(expanded[:, k], tokens[:, k])
    # pair slots equal tokens[:, i] + tokens[:, j], in triu_indices order
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for offset, (i, j) in enumerate(expected_pairs):
        assert slots[4 + offset] == (i, j)
        assert torch.equal(expanded[:, 4 + offset], tokens[:, i] + tokens[:, j])
    assert expanded.shape[1] == 4 + 6


def test_pair_tokens_mask_with_padding():
    net = _tiny_model()
    tokens = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, True], [True, True, True]])

    _, expanded_mask, _ = net._add_pair_tokens(tokens, mask)

    expected = torch.tensor(
        [
            [True, False, True, False, True, False],  # (0,1) F, (0,2) T, (1,2) F
            [True, True, True, True, True, True],
        ]
    )
    assert torch.equal(expanded_mask, expected)


def test_pair_tokens_single_slot():
    net = _tiny_model()
    tokens = torch.randn(2, 1, 8)
    mask = torch.tensor([[True], [False]])

    expanded, expanded_mask, slots = net._add_pair_tokens(tokens, mask)

    assert expanded.shape == (2, 1, 8)
    assert torch.equal(expanded, tokens)
    assert torch.equal(expanded_mask, mask)
    assert slots == [(0,)]


def test_pair_tokens_one_real_of_many():
    net = _tiny_model()
    tokens = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, False, False]])

    expanded, expanded_mask, slots = net._add_pair_tokens(tokens, mask)

    assert expanded.shape == (1, 10, 8)  # 4 + 6
    assert len(slots) == 10
    assert int(expanded_mask.sum()) == 1
    assert bool(expanded_mask[0, 0])
    assert not bool(expanded_mask[0, 1:].any())


def test_pair_tokens_permutation_invariance():
    """Sums are symmetric, so permuting slots only permutes the token set."""
    net = _tiny_model()
    torch.manual_seed(1)
    tokens = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)

    expanded, _, _ = net._add_pair_tokens(tokens, mask)
    perm = [3, 0, 4, 1, 2]
    permuted, _, _ = net._add_pair_tokens(tokens[:, perm], mask[:, perm])

    # pair part of the expanded tensors must be the same set of sums
    assert torch.allclose(
        expanded[:, 5:].sort(dim=1).values, permuted[:, 5:].sort(dim=1).values
    )


# ---------------------------------------------------------------------------
# regression heads
# ---------------------------------------------------------------------------


def test_linear_head_is_plain_linear():
    net = _tiny_model(emb_dim=8, head="linear")
    assert type(net.regressor) is torch.nn.Linear
    keys = {k for k in net.state_dict() if k.startswith("regressor.")}
    assert keys == {"regressor.weight", "regressor.bias"}


def test_mlp_head_exact_sequence():
    net = _tiny_model(emb_dim=8, head="mlp", reaction_combine="concat")
    layers = list(net.regressor)
    assert isinstance(layers[0], torch.nn.Linear)
    assert (layers[0].in_features, layers[0].out_features) == (16, 8)
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Dropout)
    assert isinstance(layers[3], torch.nn.Linear)
    assert (layers[3].in_features, layers[3].out_features) == (8, 1)
    keys = {k for k in net.state_dict() if k.startswith("regressor.")}
    assert keys == {
        "regressor.0.weight",
        "regressor.0.bias",
        "regressor.3.weight",
        "regressor.3.bias",
    }


def test_mlp_head_output_shape():
    net = _tiny_model(emb_dim=8, head="mlp").eval()
    out = net.regressor(torch.randn(3, 16)).squeeze(-1)
    assert out.shape == (3,)


def test_invalid_reactant_tokens_and_head_raise():
    with pytest.raises(ValueError):
        _tiny_model(reactant_tokens="nope")
    with pytest.raises(ValueError):
        _tiny_model(head="nope")


# ---------------------------------------------------------------------------
# full model: comb + mlp + attention_on="reactants"
# ---------------------------------------------------------------------------


def _comb_mlp_model(dataset):
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    return _tiny_model(
        emb_dim=16,
        attention_on="reactants",
        reactant_tokens="comb",
        head="mlp",
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
    ).eval()


def test_comb_mlp_forward_and_token_slots():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    torch.manual_seed(0)
    dataset = _tiny_prepared_dataset(3)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.store_attention = True
    with torch.no_grad():
        pred, vectors = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    s = dataset.rmol_max_cnt
    assert pred.shape == (2,)
    assert len(vectors) == 2 and len(vectors[0]) == 32  # concat: 2 * emb_dim
    assert len(net.last_token_slots) == s + s * (s - 1) // 2
    assert net.last_token_slots[:s] == [(slot,) for slot in range(s)]
    assert net.last_token_slots[s] == (0, 1)


def test_comb_mlp_permutation_invariance():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    torch.manual_seed(0)
    dataset = _tiny_prepared_dataset(3)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    with torch.no_grad():
        pred, vectors = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    perm = list(range(dataset.rmol_max_cnt))[::-1]
    perm_rmols = [rmols[p] for p in perm]
    perm_r_dummy = [[row[p] for p in perm] for row in r_dummy]
    with torch.no_grad():
        pred_perm, vectors_perm = net(perm_rmols, pmols, perm_r_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(pred, pred_perm, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        torch.tensor(vectors), torch.tensor(vectors_perm), rtol=1e-4, atol=1e-5
    )


def test_comb_mlp_extra_padded_slot_and_noise_invariance():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    torch.manual_seed(0)
    dataset = _tiny_prepared_dataset(3)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    with torch.no_grad():
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    padded_rmols = list(rmols) + [_dummy_slot_batch(2, node_dim, edge_dim)]
    padded_dummy = [row + [False] for row in r_dummy]
    noisy_rmols = list(rmols) + [_dummy_slot_batch(2, node_dim, edge_dim, noisy=True)]
    with torch.no_grad():
        pred_padded, _ = net(padded_rmols, pmols, padded_dummy, p_dummy, DEVICE)
        pred_noisy, _ = net(noisy_rmols, pmols, padded_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(pred, pred_padded, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(pred, pred_noisy, rtol=1e-4, atol=1e-5)


def test_comb_mlp_backprop_reaches_gin():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    torch.manual_seed(0)
    dataset = _tiny_prepared_dataset(3)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.zero_grad()
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    loss = (pred**2).sum()
    loss.backward()

    grads = [p.grad for p in net.gnn.parameters() if p.grad is not None]
    assert grads
    assert any(
        torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in grads
    ), "no GIN parameter received a finite, nonzero gradient"


def test_last_token_slots_cleared_when_not_storing():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    dataset = _tiny_prepared_dataset(2)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.store_attention = True
    with torch.no_grad():
        net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert net.last_token_slots is not None

    net.store_attention = False
    with torch.no_grad():
        net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert net.last_token_slots is None


def test_ind_token_slots_are_individual_only():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    dataset = _tiny_prepared_dataset(2)
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    net = _tiny_model(
        emb_dim=16,
        attention_on="reactants",
        reactant_tokens="ind",
        head="linear",
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
    ).eval()
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.store_attention = True
    with torch.no_grad():
        net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    s = dataset.rmol_max_cnt
    assert net.last_token_slots == [(slot,) for slot in range(s)]


# ---------------------------------------------------------------------------
# real featurization (needs genuine rdkit)
# ---------------------------------------------------------------------------

_TINY_REACTIONS = [
    "CCO.CC(=O)O>>CCOC(C)=O",
    "CO.OC(=O)C>>COC(C)=O",
    "CCBr.[OH-]>>CCO",
]


def test_real_featurization_comb_mlp_invariance():
    if not HAS_RDKIT:
        pytest.skip(
            "rdkit native library is blocked by the Windows Application Control "
            "policy; run this test where rdkit loads"
        )
    from reaction_data import get_graph_data

    y = np.array([0.2, 0.5, 0.8])
    rmol, pmol, reaction = get_graph_data(_TINY_REACTIONS, 2, 1, None, None, y)
    dataset = GraphDataset(rmol=rmol, pmol=pmol, reaction=reaction)
    net = _comb_mlp_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    with torch.no_grad():
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    padded_rmols = list(rmols) + [_dummy_slot_batch(2, node_dim, edge_dim)]
    padded_dummy = [row + [False] for row in r_dummy]
    with torch.no_grad():
        pred_padded, _ = net(padded_rmols, pmols, padded_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(pred, pred_padded, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# backward compatibility: ind/linear must equal the original model
# ---------------------------------------------------------------------------


def _reference_forward(net, rmols, pmols, r_dummy, p_dummy, device):
    """Frozen copy of the original (pre-approach-2) forward pass."""
    r_tokens = torch.stack([net.gnn(rmol) for rmol in rmols], dim=1).to(device)
    p_tokens = torch.stack([net.gnn(pmol) for pmol in pmols], dim=1).to(device)
    r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
    p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)

    if net.attention_on == "all":
        n_reactant_slots = r_tokens.shape[1]
        attended, _ = net._attend(
            torch.cat((r_tokens, p_tokens), dim=1),
            torch.cat((r_mask, p_mask), dim=1),
        )
        r_tokens = attended[:, :n_reactant_slots]
        p_tokens = attended[:, n_reactant_slots:]
    else:
        if net.attention_on in ("reactants", "both"):
            r_tokens, _ = net._attend(r_tokens, r_mask)
        if net.attention_on in ("products", "both"):
            p_tokens, _ = net._attend(p_tokens, p_mask)

    reactant_vectors = net._masked_mean(r_tokens, r_mask)
    product_vectors = net._masked_mean(p_tokens, p_mask)
    reaction_vectors = net._combine(reactant_vectors, product_vectors)
    out = net.regressor(reaction_vectors).squeeze(-1)
    return out, reaction_vectors.tolist()


def _tiny_inputs():
    _require_utils()
    if not os.path.isfile(SUZUKI_TEST_NPZ):
        pytest.skip("prepared Suzuki npz not available")
    dataset = _tiny_prepared_dataset(3)
    batch = _batch(dataset, 2)
    return dataset, _split_batch(batch, dataset.rmol_max_cnt, dataset.pmol_max_cnt)


def test_default_model_matches_frozen_reference():
    torch.manual_seed(0)
    dataset, (rmols, pmols, r_dummy, p_dummy, _) = _tiny_inputs()
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    net = _tiny_model(
        emb_dim=16, node_in_feats=node_dim, edge_in_feats=edge_dim
    ).eval()

    with torch.no_grad():
        pred_new, vectors_new = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
        pred_ref, vectors_ref = _reference_forward(
            net, rmols, pmols, r_dummy, p_dummy, DEVICE
        )

    torch.testing.assert_close(pred_new, pred_ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.tensor(vectors_new), torch.tensor(vectors_ref), rtol=1e-6, atol=1e-7
    )


def test_default_config_is_ind_linear():
    net = _tiny_model()
    assert net.reactant_tokens == "ind"
    assert net.head == "linear"
    assert net.config["reactant_tokens"] == "ind"
    assert net.config["head"] == "linear"
    assert isinstance(net.regressor, torch.nn.Linear)
    assert REACTANT_TOKENS == ("ind", "comb")
    assert HEADS == ("linear", "mlp")


def test_old_linear_state_dict_loads_strict():
    torch.manual_seed(0)
    dataset, (rmols, pmols, r_dummy, p_dummy, _) = _tiny_inputs()
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    old = _tiny_model(emb_dim=16, node_in_feats=node_dim, edge_in_feats=edge_dim)
    old_state = {k: v.clone() for k, v in old.state_dict().items()}
    old_keys = {
        "gnn.project_node_feats.0.weight",
        "gnn.project_node_feats.0.bias",
        "regressor.weight",
        "regressor.bias",
    }
    assert old_keys <= set(old_state)

    new = _tiny_model(emb_dim=16, node_in_feats=node_dim, edge_in_feats=edge_dim)
    new.load_state_dict(old_state, strict=True)  # must not raise

    with torch.no_grad():
        pred, _ = new(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert pred.shape == (2,)


def test_old_config_loads_through_from_config():
    config = {
        "node_in_feats": 155,
        "edge_in_feats": 9,
        "num_layer": 1,
        "emb_dim": 16,
        "drop_ratio": 0.0,
    }
    net = model.from_config(config)
    assert net.reactant_tokens == "ind"
    assert net.head == "linear"
    assert isinstance(net.regressor, torch.nn.Linear)
    assert net.config["reactant_tokens"] == "ind"
    assert net.config["head"] == "linear"


# ---------------------------------------------------------------------------
# CLI + checkpoint compatibility
# ---------------------------------------------------------------------------


def test_cli_defaults_and_explicit_choices():
    if not HAS_RDKIT:
        pytest.skip(
            "main_finetune imports rdkit chemistry (prepare_data/reaction_data), "
            "blocked by the Windows Application Control policy"
        )
    from main_finetune import build_parser

    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.reactant_tokens == "ind"
    assert defaults.head == "linear"

    explicit = parser.parse_args(["--reactant_tokens", "comb", "--head", "mlp"])
    assert explicit.reactant_tokens == "comb"
    assert explicit.head == "mlp"

    with pytest.raises(SystemExit):
        parser.parse_args(["--head", "nope"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--reactant_tokens", "nope"])


def _import_finetune():
    try:
        from finetune import _build_model, _load_checkpoint_safely

        return _build_model, _load_checkpoint_safely
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip("finetune import blocked by the environment: %s" % exc)


def test_build_model_getattr_defaults():
    _build_model, _ = _import_finetune()
    import argparse

    # an older Namespace without the new options must resolve to ind/linear
    old_args = argparse.Namespace(layer=1, emb_dim=16, dropout=0.0)
    net = _build_model(old_args, 155, 9)
    assert net.reactant_tokens == "ind"
    assert net.head == "linear"


def test_checkpoint_safely_old_and_mismatches(tmp_path):
    _, _load_checkpoint_safely = _import_finetune()
    logger = logging.getLogger("test_approach2")
    torch.manual_seed(0)

    net = _tiny_model(emb_dim=16)
    old_config = {
        k: v for k, v in net.config.items() if k not in ("reactant_tokens", "head")
    }

    # 1. an old checkpoint (no new keys) loads and resolves to ind/linear
    target = _tiny_model(emb_dim=16)
    _load_checkpoint_safely(
        target,
        {"model_state_dict": net.state_dict(), "model_config": old_config},
        logger,
    )
    assert target.reactant_tokens == "ind" and target.head == "linear"

    # 2. a reactant_tokens mismatch is an incompatible resumed architecture
    comb_ckpt = {
        "model_state_dict": net.state_dict(),
        "model_config": dict(net.config, reactant_tokens="comb"),
    }
    with pytest.raises(RuntimeError):
        _load_checkpoint_safely(_tiny_model(emb_dim=16), comb_ckpt, logger)

    # 3. a head mismatch is tolerated as a warm-start head replacement
    mlp_net = _tiny_model(emb_dim=16, head="mlp")
    linear_ckpt = {
        "model_state_dict": net.state_dict(),
        "model_config": dict(net.config, head="linear"),
    }
    _load_checkpoint_safely(mlp_net, linear_ckpt, logger)  # must not raise

    # 4. an encoder mismatch still fails
    bad_encoder = {
        "model_state_dict": net.state_dict(),
        "model_config": dict(net.config, num_layer=2),
    }
    with pytest.raises(RuntimeError):
        _load_checkpoint_safely(_tiny_model(emb_dim=16), bad_encoder, logger)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
