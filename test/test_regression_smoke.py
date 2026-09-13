"""
Regression smoke test for the reaction-yield model.

NOTE: this environment's Windows Smart App Control policy blocks loading
rdkit's native `cDataStructs.pyd` (unsigned/unrecognized binary), so this test
builds synthetic reaction graphs with the same tensor shapes that
`reaction_data.get_graph_data` (which depends on rdkit) would normally
produce, instead of featurizing real SMILES. This still exercises the real
`GraphDataset`, `collate_reaction_graphs`, and `model` code paths end to end.
The real TSV file is still read with pandas to validate the data pipeline's
loading/target handling.
"""

import os
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import collate_reaction_graphs, read_reaction_table  # noqa: E402
from data import GraphDataset  # noqa: E402
from model import model  # noqa: E402

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "raw",
    "random_split_0.tsv",
)

NODE_DIM = 155  # matches preprocess_utils.add_mol/add_dummy node feature width
EDGE_DIM = 9  # matches preprocess_utils bond_fea1(5) + bond_fea2(2) + bond_fea3(2)


def _synthetic_mol_dict(n_samples: int, rng: np.random.Generator) -> dict:
    n_node = rng.integers(2, 4, size=n_samples)
    n_edge = n_node  # one bidirectional bond per node, arbitrary but consistent

    node_attr = rng.random((int(n_node.sum()), NODE_DIM)) > 0.5
    edge_attr = rng.random((int(n_edge.sum()), EDGE_DIM)) > 0.5

    src, dst = [], []
    for n in n_node:
        idx = np.arange(n)
        src.append(idx)
        dst.append(np.roll(idx, 1))
    src = np.concatenate(src).astype(int)
    dst = np.concatenate(dst).astype(int)

    return {
        "n_node": n_node.astype(int),
        "n_edge": n_edge.astype(int),
        "dummy": np.ones(n_samples, dtype=bool),
        "node_attr": node_attr.astype(bool),
        "edge_attr": edge_attr.astype(bool),
        "src": src,
        "dst": dst,
    }


def _build_synthetic_dataset(y_values: np.ndarray, rmol_max_cnt=2, pmol_max_cnt=1):
    """Builds a GraphDataset with synthetic graphs but real yield targets."""
    n_samples = len(y_values)
    rng = np.random.default_rng(0)

    rmol_dict = [_synthetic_mol_dict(n_samples, rng) for _ in range(rmol_max_cnt)]
    pmol_dict = [_synthetic_mol_dict(n_samples, rng) for _ in range(pmol_max_cnt)]
    reaction_dict = {
        "y": np.asarray(y_values, dtype=float),
        "rsmi": ["synthetic_rxn_%d" % i for i in range(n_samples)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "smoke.npz")
        np.savez_compressed(
            npz_path, rmol=rmol_dict, pmol=pmol_dict, reaction=reaction_dict
        )
        dataset = GraphDataset(save_path=npz_path)

    return dataset


def _split_batch(batch, rmol_max_cnt, pmol_max_cnt):
    rmols = batch[:rmol_max_cnt]
    pmols = batch[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
    r_dummy = batch[-4]
    p_dummy = batch[-3]
    labels = batch[-2]
    return rmols, pmols, r_dummy, p_dummy, labels


def test_tsv_loads_and_targets_are_valid():
    """--- The TSV can be loaded, and yield targets are valid floats. ---"""
    df = read_reaction_table(DATA_PATH)
    assert "rxn" in df.columns
    assert "y" in df.columns
    assert len(df) > 0

    y = df["y"].astype(float)
    assert y.notna().all()
    assert (y >= 0).all() and (y <= 1).all()


def test_regression_smoke():
    df = read_reaction_table(DATA_PATH).head(8)
    y_values = df["y"].values.astype(float)

    dataset = _build_synthetic_dataset(y_values, rmol_max_cnt=2, pmol_max_cnt=1)
    assert len(dataset) == 8

    rmol_max_cnt = dataset.rmol_max_cnt
    pmol_max_cnt = dataset.pmol_max_cnt
    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    assert node_dim == NODE_DIM
    assert edge_dim == EDGE_DIM

    # --- a small batch can be created ---
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_reaction_graphs
    )
    batch = next(iter(loader))
    rmols, pmols, r_dummy, p_dummy, labels = _split_batch(
        batch, rmol_max_cnt, pmol_max_cnt
    )

    # --- reaction inputs have the expected shape ---
    for r in rmols:
        assert r.x.shape[1] == node_dim
        assert r.edge_attr.shape[1] == edge_dim
    for p in pmols:
        assert p.x.shape[1] == node_dim

    # --- targets are floating point and shaped [batch_size] ---
    assert labels.dtype == torch.float32
    assert labels.shape == (4,)

    device = torch.device("cpu")
    net = model(node_dim, edge_dim, num_layer=2, emb_dim=16, drop_ratio=0.0).to(device)

    # --- forward pass succeeds, output shape is [4] ---
    pred, _, _, _ = net(rmols, pmols, r_dummy, p_dummy, device)
    assert pred.shape == (4,)
    assert torch.isfinite(pred).all()

    # --- final layer outputs one value per reaction ---
    assert net.regressor.out_features == 1

    # --- loss is finite ---
    loss_fn = torch.nn.HuberLoss()
    loss = loss_fn(pred, labels)
    assert torch.isfinite(loss)

    # --- backpropagation succeeds with finite, non-null gradients on the head ---
    net.zero_grad()
    loss.backward()
    assert net.regressor.weight.grad is not None
    assert torch.isfinite(net.regressor.weight.grad).all()

    # --- batch of 1: output shape is [1] ---
    loader1 = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_reaction_graphs
    )
    batch1 = next(iter(loader1))
    rmols1, pmols1, r_dummy1, p_dummy1, labels1 = _split_batch(
        batch1, rmol_max_cnt, pmol_max_cnt
    )
    assert labels1.shape == (1,)

    pred1, _, _, _ = net(rmols1, pmols1, r_dummy1, p_dummy1, device)
    assert pred1.shape == (1,)
    assert torch.isfinite(pred1).all()


if __name__ == "__main__":
    test_tsv_loads_and_targets_are_valid()
    test_regression_smoke()
    print("Smoke test passed.")
