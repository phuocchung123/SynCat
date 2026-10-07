"""
Unit tests for the 16-cell factorial ablation grid runner and model switches.

CPU-only, no rdkit.Chem imports, finishes in under 60 seconds.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SRC_DIR, TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import model  # noqa: E402
from run_grid import (  # noqa: E402
    CELLS,
    _get_peak_gpu_memory_mb,
    _reset_peak_gpu_memory,
    summarize,
    task_id_to_dataset_and_cell,
    tasks,
)
from test_model_approach1 import (  # noqa: E402
    _batch,
    _dataset_with_slots,
    _dummy_slot_batch,
    _split_batch,
)

DEVICE = torch.device("cpu")

EXPECTED_CELL_LABELS = (
    "baseline",
    "baseline_mlp",
    "B_linear",
    "B_mlp",
    "control_linear",
    "control_mlp",
    "A_linear",
    "A_mlp",
    "approach3",
    "approach2",
    "approach1_linear",
    "approach1_mlp",
    "comb_none_mean_linear",
    "comb_none_mean_mlp",
    "comb_none_rn_linear",
    "comb_none_rn_mlp",
)


# ---------------------------------------------------------------------------
# 1. Parametrize over all 16 cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS, ids=[c["label"] for c in CELLS])
def test_all_16_cells_forward_backward_invariance(cell):
    torch.manual_seed(42)
    dataset = _dataset_with_slots(rmol_max_cnt=3)
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    net = model(
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        attention_on=cell["attention_on"],
        reactant_tokens=cell["reactant_tokens"],
        reactant_pooling=cell["reactant_pooling"],
        head=cell["head"],
    )

    batch = _batch(dataset, batch_size=2)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(batch, 3, 1)

    # 1. Forward + backward
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()

    loss = pred.sum()
    loss.backward()
    grads = [p.grad for p in net.gnn.parameters() if p.grad is not None]
    assert grads, "no GNN parameter gradient found"
    assert any(
        torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in grads
    ), "no finite, nonzero gradient reached GNN"

    # In eval mode for invariance tests
    net.eval()
    with torch.no_grad():
        ref_pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    # 2. Permutation invariance: permuting reactant slots + dummy flags
    perm = [2, 0, 1]
    perm_rmols = [rmols[i] for i in perm]
    perm_r_dummy = [[row[i] for i in perm] for row in r_dummy]
    with torch.no_grad():
        pred_perm, _ = net(perm_rmols, pmols, perm_r_dummy, p_dummy, DEVICE)
    torch.testing.assert_close(ref_pred, pred_perm, rtol=1e-4, atol=1e-5)

    # 3. Padded noisy slot invariance
    noisy_slot = _dummy_slot_batch(batch_size=2, node_dim=node_dim, edge_dim=edge_dim, noisy=True)
    padded_rmols = list(rmols) + [noisy_slot]
    padded_r_dummy = [row + [False] for row in r_dummy]
    with torch.no_grad():
        pred_noisy, _ = net(padded_rmols, pmols, padded_r_dummy, p_dummy, DEVICE)
    torch.testing.assert_close(ref_pred, pred_noisy, rtol=1e-4, atol=1e-5)

    # 4. S == 1 (rmol_max_cnt = 1)
    dataset_s1 = _dataset_with_slots(rmol_max_cnt=1)
    batch_s1 = _batch(dataset_s1, batch_size=2)
    rmols_s1, pmols_s1, r_dummy_s1, p_dummy_s1, _ = _split_batch(batch_s1, 1, 1)

    net_s1 = model(
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        attention_on=cell["attention_on"],
        reactant_tokens=cell["reactant_tokens"],
        reactant_pooling=cell["reactant_pooling"],
        head=cell["head"],
    )
    pred_s1, _ = net_s1(rmols_s1, pmols_s1, r_dummy_s1, p_dummy_s1, DEVICE)
    assert pred_s1.shape == (2,)
    assert torch.isfinite(pred_s1).all()

    loss_s1 = pred_s1.sum()
    loss_s1.backward()
    grads_s1 = [p.grad for p in net_s1.gnn.parameters() if p.grad is not None]
    assert grads_s1 and any(
        torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in grads_s1
    )

    if cell["reactant_tokens"] == "comb":
        net_s1.store_attention = True
        with torch.no_grad():
            net_s1(rmols_s1, pmols_s1, r_dummy_s1, p_dummy_s1, DEVICE)
        assert len(net_s1.last_token_slots) == 1

    if cell["reactant_pooling"] == "rn":
        with torch.no_grad():
            _, pair_out, _ = net_s1.reactant_pool(
                torch.randn(2, 1, 16),
                torch.ones(2, 1, dtype=torch.bool),
                return_pairs=True,
            )
        assert pair_out.shape[1] == 0


# ---------------------------------------------------------------------------
# 2. attention_on="all" + comb
# ---------------------------------------------------------------------------


def test_attention_on_all_with_comb():
    torch.manual_seed(42)
    dataset = _dataset_with_slots(rmol_max_cnt=3)
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    net = model(
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        attention_on="all",
        reactant_tokens="comb",
        reactant_pooling="rn",
        head="mlp",
    ).eval()
    net.store_attention = True

    batch = _batch(dataset, batch_size=2)
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(batch, 3, 1)

    with torch.no_grad():
        pred, _ = net(rmols, pmols, r_dummy, p_dummy, DEVICE)

    # 1. Attention matrix shape is (S + S*(S-1)/2 + P) square: (3 + 3 + 1) = 7 square
    assert net.last_attention is not None
    att_arr = np.asarray(net.last_attention)
    assert att_arr.shape[-2:] == (7, 7)

    # 2. Manual reference calculation
    r_tokens = torch.stack([net.gnn(rmol) for rmol in rmols], dim=1)
    p_tokens = torch.stack([net.gnn(pmol) for pmol in pmols], dim=1)
    r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool))
    p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool))

    r_tokens_exp, r_mask_exp, _ = net._add_pair_tokens(r_tokens, r_mask)
    all_tokens = torch.cat((r_tokens_exp, p_tokens), dim=1)
    all_mask = torch.cat((r_mask_exp, p_mask), dim=1)
    attended, _ = net._attend(all_tokens, all_mask)

    n_reactant_slots = r_tokens_exp.shape[1]
    r_attended = attended[:, :n_reactant_slots]
    p_attended = attended[:, n_reactant_slots:]

    r_vec = net._pool_reactants(r_attended, r_mask_exp)
    p_vec = net._masked_mean(p_attended, p_mask)
    rxn_vec = net._combine(r_vec, p_vec)
    ref_pred = net.regressor(rxn_vec).squeeze(-1)

    torch.testing.assert_close(pred, ref_pred, rtol=1e-4, atol=1e-5)

    # 3. Padded noise invariance
    noisy_slot = _dummy_slot_batch(batch_size=2, node_dim=node_dim, edge_dim=edge_dim, noisy=True)
    padded_rmols = list(rmols) + [noisy_slot]
    padded_r_dummy = [row + [False] for row in r_dummy]
    with torch.no_grad():
        pred_noisy, _ = net(padded_rmols, pmols, padded_r_dummy, p_dummy, DEVICE)
    torch.testing.assert_close(pred, pred_noisy, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Grid definition & helpers
# ---------------------------------------------------------------------------


def test_grid_definitions_and_labels():
    assert len(CELLS) == 16
    seen_combos = set()
    for idx, cell in enumerate(CELLS):
        combo = (
            cell["reactant_tokens"],
            cell["attention_on"],
            cell["reactant_pooling"],
            cell["head"],
        )
        assert combo not in seen_combos, "duplicate cell combination: %s" % (combo,)
        seen_combos.add(combo)
        assert cell["label"] == EXPECTED_CELL_LABELS[idx]


def test_task_id_mapping():
    pairs = [task_id_to_dataset_and_cell(i) for i in range(32)]
    unique_pairs = {(d, c["label"]) for d, c in pairs}
    assert len(unique_pairs) == 32

    for i in range(16):
        assert pairs[i][0] == "Buchwald-Hartwig"
        assert pairs[i][1] == CELLS[i]
    for i in range(16, 32):
        assert pairs[i][0] == "Suzuki-Miyaura"
        assert pairs[i][1] == CELLS[i - 16]


def test_tasks_splits_and_metadata_reading(tmp_path):
    # Setup tmp Suzuki split_0 metadata
    suzuki_dir = tmp_path / "processed" / "suzuki" / "npz" / "split_0"
    suzuki_dir.mkdir(parents=True)
    meta_path = suzuki_dir / "split_metadata.json"
    meta_path.write_text(json.dumps({"seed": 777}))

    args = argparse.Namespace(
        profile="full",
        bh_cv_ids=list(range(1, 11)),
        bh_test_ids=[1, 2, 3, 4],
        suzuki_split_ids=[0],
        seed=42,
        Data_folder=str(tmp_path) + os.sep,
    )
    task_list = tasks(args, profile="full")

    bh_cv = [t for t in task_list if t["dataset"] == "Buchwald-Hartwig" and t["kind"] == "cv"]
    bh_test = [t for t in task_list if t["dataset"] == "Buchwald-Hartwig" and t["kind"] == "test"]
    suzuki = [t for t in task_list if t["dataset"] == "Suzuki-Miyaura"]

    assert len(bh_cv) == 10
    assert all(t["seed"] == 42 for t in bh_cv)
    assert len(bh_test) == 4
    assert all(t["seed"] == 42 for t in bh_test)
    assert len(suzuki) == 1
    assert suzuki[0]["seed"] == 777


def test_summarize_toy_frame():
    toy_rows = [
        # Group 1: n = 2 successful runs
        {
            "dataset": "Buchwald-Hartwig",
            "kind": "cv",
            "label": "baseline",
            "cell": "ind/reactants/mean/linear",
            "status": "success",
            "trainable_parameters": 1000,
            "train_r2": 0.8,
            "train_mae": 0.1,
            "train_rmse": 0.15,
            "test_r2": 0.7,
            "test_mae": 0.12,
            "test_rmse": 0.16,
            "seconds_per_epoch": 2.0,
            "peak_gpu_mem_mb": 50.0,
        },
        {
            "dataset": "Buchwald-Hartwig",
            "kind": "cv",
            "label": "baseline",
            "cell": "ind/reactants/mean/linear",
            "status": "success",
            "trainable_parameters": 1000,
            "train_r2": 0.9,
            "train_mae": 0.08,
            "train_rmse": 0.11,
            "test_r2": 0.8,
            "test_mae": 0.10,
            "test_rmse": 0.14,
            "seconds_per_epoch": 3.0,
            "peak_gpu_mem_mb": 60.0,
        },
        # Group 2: n = 1 successful, 1 failed
        {
            "dataset": "Buchwald-Hartwig",
            "kind": "test",
            "label": "baseline",
            "cell": "ind/reactants/mean/linear",
            "status": "success",
            "trainable_parameters": 1000,
            "train_r2": 0.85,
            "train_mae": 0.09,
            "train_rmse": 0.12,
            "test_r2": 0.75,
            "test_mae": 0.11,
            "test_rmse": 0.15,
            "seconds_per_epoch": 2.5,
            "peak_gpu_mem_mb": 55.0,
        },
        {
            "dataset": "Buchwald-Hartwig",
            "kind": "test",
            "label": "baseline",
            "cell": "ind/reactants/mean/linear",
            "status": "failed",
            "trainable_parameters": 1000,
            "train_r2": None,
            "train_mae": None,
            "train_rmse": None,
            "test_r2": None,
            "test_mae": None,
            "test_rmse": None,
            "seconds_per_epoch": None,
            "peak_gpu_mem_mb": None,
        },
    ]

    df = pd.DataFrame(toy_rows)
    summary = summarize(df)

    assert len(summary) == 2
    row1 = summary.iloc[0]
    assert row1["dataset"] == "Buchwald-Hartwig"
    assert row1["kind"] == "cv"
    assert row1["label"] == "baseline"
    assert row1["n_runs"] == 2
    assert row1["n_failed"] == 0
    assert row1["test_r2_mean"] == pytest.approx(0.75)
    expected_std = np.std([0.7, 0.8], ddof=1)
    assert row1["test_r2_std"] == pytest.approx(expected_std)
    assert row1["test_r2_pm"] == "%.4f ± %.4f" % (0.75, expected_std)

    row2 = summary.iloc[1]
    assert row2["dataset"] == "Buchwald-Hartwig"
    assert row2["kind"] == "test"
    assert row2["label"] == "baseline"
    assert row2["n_runs"] == 1
    assert row2["n_failed"] == 1
    assert row2["test_r2_mean"] == pytest.approx(0.75)
    assert np.isnan(row2["test_r2_std"])
    assert row2["test_r2_pm"] == "0.7500 ± nan"


def test_gpu_memory_helpers_safe_on_cpu_and_cuda_handling(monkeypatch):
    cpu_device = torch.device("cpu")
    # Safe no-op on CPU
    _reset_peak_gpu_memory(cpu_device)
    assert _get_peak_gpu_memory_mb(cpu_device) is None

    # Test error resilience when cuda raises invalid device argument
    cuda_device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)

    def mock_empty(*args, **kwargs):
        raise RuntimeError("Invalid device argument")

    monkeypatch.setattr(torch, "empty", mock_empty)
    # Should not raise exception
    _reset_peak_gpu_memory(cuda_device)

    # Test safe get_peak_gpu_memory_mb fallback
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 1024 * 1024 * 100)
    assert _get_peak_gpu_memory_mb(cuda_device) == 100.0

