"""
Tests for SynCat approach 1: relation-network pooling of the attended reactant
tokens (``reactant_pooling="rn"``), which combines with the already-existing
pairwise reactant tokens (``reactant_tokens="comb"``) and the switchable head.

All tests are deliberately tiny so they finish in seconds on a CPU: small
embeddings, one GIN layer, at most a couple of batches. Only one test uses the
real Suzuki token width (14 slots, T=105, 5460 pairs) and it does a single
forward plus backward.

``rdkit.Chem`` cannot load on this machine (Windows Application Control blocks
``rdchem``), so the graph tensors come from the synthetic builders in
``test_regression_smoke`` instead of featurizing SMILES.
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_DIR, TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import REACTANT_POOLINGS, RelationPooling, model  # noqa: E402

# Reuse the synthetic builders written for the rdkit block; do not copy them.
try:
    from test_regression_smoke import (  # noqa: E402
        _build_synthetic_dataset,
        _split_batch,
    )
except ImportError:  # pragma: no cover - depends on how pytest imports the package
    from .test_regression_smoke import (  # noqa: E402
        _build_synthetic_dataset,
        _split_batch,
    )

DEVICE = torch.device("cpu")


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


def _rp(emb_dim=8):
    return RelationPooling(emb_dim, 0.0).eval()


def _dataset_with_slots(rmol_max_cnt):
    y_values = np.linspace(0.1, 0.9, 4)
    return _build_synthetic_dataset(
        y_values, rmol_max_cnt=rmol_max_cnt, pmol_max_cnt=1
    )


def _batch(dataset, batch_size=2):
    from torch.utils.data import DataLoader

    from utils import collate_reaction_graphs

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=0,
    )
    return next(iter(loader))


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


def test_pair_tokens_are_slot_sums():
    net = _tiny_model()
    tokens = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    expanded, _, slots = net._add_pair_tokens(tokens, mask)

    for k in range(4):
        assert slots[k] == (k,)
        assert torch.equal(expanded[:, k], tokens[:, k])
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for offset, (i, j) in enumerate(expected_pairs):
        assert slots[4 + offset] == (i, j)
        assert torch.equal(expanded[:, 4 + offset], tokens[:, i] + tokens[:, j])


def test_pair_tokens_mask_with_padding():
    net = _tiny_model()
    tokens = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, True], [True, True, True]])

    _, expanded_mask, _ = net._add_pair_tokens(tokens, mask)

    expected = torch.tensor(
        [
            [True, False, True, False, True, False],
            [True, True, True, True, True, True],
        ]
    )
    assert torch.equal(expanded_mask, expected)


def test_pair_tokens_single_real_reactant():
    net = _tiny_model()
    tokens = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, False, False]])

    expanded, expanded_mask, slots = net._add_pair_tokens(tokens, mask)

    assert expanded.shape == (1, 10, 8)
    assert len(slots) == 10
    assert int(expanded_mask.sum()) == 1
    assert bool(expanded_mask[0, 0])
    assert not bool(expanded_mask[0, 1:].any())


# ---------------------------------------------------------------------------
# RelationPooling
# ---------------------------------------------------------------------------


def test_relation_pooling_output_shape():
    rp = _rp()
    out = rp(torch.randn(3, 4, 8), torch.ones(3, 4, dtype=torch.bool))
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_relation_pooling_permutation_invariance():
    torch.manual_seed(1)
    rp = _rp()
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    out = rp(x, mask)
    perm = [2, 0, 3, 1]
    out_perm = rp(x[:, perm], mask[:, perm])

    torch.testing.assert_close(out, out_perm, rtol=1e-5, atol=1e-5)


def test_relation_pooling_ignores_masked_token_noise():
    torch.manual_seed(2)
    rp = _rp()
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])

    out = rp(x, mask)
    x_noisy = x.clone()
    x_noisy[~mask] = 1e4 * torch.randn_like(x_noisy[~mask])
    out_noisy = rp(x_noisy, mask)

    torch.testing.assert_close(out, out_noisy, rtol=1e-5, atol=1e-5)


def test_relation_pooling_extra_masked_token_is_ignored():
    torch.manual_seed(3)
    rp = _rp()
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    out = rp(x, mask)
    x_extra = torch.cat((x, torch.randn(2, 1, 8)), dim=1)
    mask_extra = torch.cat((mask, torch.zeros(2, 1, dtype=torch.bool)), dim=1)
    out_extra = rp(x_extra, mask_extra)

    torch.testing.assert_close(out, out_extra, rtol=1e-5, atol=1e-5)


def test_relation_pooling_single_token():
    torch.manual_seed(4)
    rp = _rp()
    x = torch.randn(2, 1, 8)
    mask = torch.ones(2, 1, dtype=torch.bool)

    out = rp(x, mask)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()

    _, pair_out, (a, b) = rp(x, mask, return_pairs=True)
    assert pair_out.shape == (2, 0, 8)
    assert a.numel() == 0 and b.numel() == 0


def test_relation_pooling_gradients_reach_phi_and_g():
    torch.manual_seed(5)
    rp = RelationPooling(8, 0.0)
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    rp(x, mask).sum().backward()

    for module in (rp.phi, rp.g):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)


# ---------------------------------------------------------------------------
# full model: comb + rn + attention_on="reactants"
# ---------------------------------------------------------------------------


def _comb_rn_model(dataset):
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    return _tiny_model(
        emb_dim=16,
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
        attention_on="reactants",
        reactant_tokens="comb",
        reactant_pooling="rn",
    ).eval()


def test_full_comb_rn_permutation_invariance():
    torch.manual_seed(0)
    dataset = _dataset_with_slots(3)
    net = _comb_rn_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )
    # One padding slot, so the permutation also has to move the mask.
    padded_dummy = [[True, True, False] for _ in r_dummy]

    with torch.no_grad():
        pred, _ = net(rmols, pmols, padded_dummy, p_dummy, DEVICE)

    perm = [2, 0, 1]
    perm_rmols = [rmols[p] for p in perm]
    perm_dummy = [[row[p] for p in perm] for row in padded_dummy]
    with torch.no_grad():
        pred_perm, _ = net(perm_rmols, pmols, perm_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(pred, pred_perm, rtol=1e-5, atol=1e-5)


def test_full_comb_rn_extra_and_noisy_padded_slots_are_ignored():
    torch.manual_seed(0)
    dataset = _dataset_with_slots(4)
    net = _comb_rn_model(dataset)
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )
    base_rmols = list(rmols[:3])
    base_dummy = [[True, True, True] for _ in r_dummy]
    extra_dummy = [[True, True, True, False] for _ in r_dummy]

    with torch.no_grad():
        pred_base, _ = net(base_rmols, pmols, base_dummy, p_dummy, DEVICE)
        pred_extra, _ = net(rmols, pmols, extra_dummy, p_dummy, DEVICE)

    noisy_rmols = list(rmols[:3]) + [
        _dummy_slot_batch(2, node_dim, edge_dim, noisy=True)
    ]
    with torch.no_grad():
        pred_noisy, _ = net(noisy_rmols, pmols, extra_dummy, p_dummy, DEVICE)

    torch.testing.assert_close(pred_base, pred_extra, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pred_base, pred_noisy, rtol=1e-5, atol=1e-5)


def test_suzuki_width_comb_rn_forward_backward():
    """The real Suzuki width: 14 slots -> T=105 tokens, 5460 pairs."""
    torch.manual_seed(0)
    dataset = _dataset_with_slots(14)
    assert dataset.rmol_max_cnt == 14
    net = _comb_rn_model(dataset)
    net.store_attention = True
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.zero_grad()
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()
    assert len(net.last_token_slots) == 105  # 14 + 14 * 13 // 2
    assert len(net.last_pair_terms) == 5460  # 105 * 104 // 2

    pred.sum().backward()
    grads = [p.grad for p in net.reactant_pool.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_last_pair_terms_cleared_when_not_storing():
    torch.manual_seed(0)
    dataset = _dataset_with_slots(3)
    net = _comb_rn_model(dataset)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    net.store_attention = True
    with torch.no_grad():
        net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert net.last_pair_terms is not None

    net.store_attention = False
    with torch.no_grad():
        net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert net.last_pair_terms is None


# ---------------------------------------------------------------------------
# backward compatibility: the default model is unchanged
# ---------------------------------------------------------------------------


def _reference_forward_default(net, rmols, pmols, r_dummy, p_dummy, device):
    """Frozen copy of the pre-approach-1 (masked-mean) forward pass."""
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


def test_default_model_matches_frozen_reference():
    torch.manual_seed(0)
    dataset = _dataset_with_slots(3)
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    net = _tiny_model(
        emb_dim=16, node_in_feats=node_dim, edge_in_feats=edge_dim
    ).eval()
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(
        _batch(dataset, 2), dataset.rmol_max_cnt, dataset.pmol_max_cnt
    )

    with torch.no_grad():
        pred, vectors = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
        pred_ref, vectors_ref = _reference_forward_default(
            net, rmols, pmols, r_dummy, p_dummy, DEVICE
        )

    torch.testing.assert_close(pred, pred_ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.tensor(vectors), torch.tensor(vectors_ref), rtol=1e-6, atol=1e-7
    )


def test_default_config_is_mean_and_has_no_pool_module():
    net = _tiny_model()
    assert net.reactant_pooling == "mean"
    assert net.config["reactant_pooling"] == "mean"
    assert not hasattr(net, "reactant_pool")
    assert REACTANT_POOLINGS == ("mean", "rn")


def test_old_state_dict_loads_strict():
    torch.manual_seed(0)
    old = _tiny_model(emb_dim=16)
    state = {k: v.clone() for k, v in old.state_dict().items()}

    new = _tiny_model(emb_dim=16)
    new.load_state_dict(state, strict=True)  # must not raise
    assert not hasattr(new, "reactant_pool")


def test_old_config_loads_through_from_config():
    config = {
        "node_in_feats": 155,
        "edge_in_feats": 9,
        "num_layer": 1,
        "emb_dim": 16,
        "drop_ratio": 0.0,
    }
    net = model.from_config(config)
    assert net.reactant_pooling == "mean"
    assert net.config["reactant_pooling"] == "mean"
    assert not hasattr(net, "reactant_pool")


def test_invalid_reactant_pooling_raises():
    with pytest.raises(ValueError):
        _tiny_model(reactant_pooling="nope")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
