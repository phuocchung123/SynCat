"""Tests for the bond-line encoders and the end-to-end bond-line model."""

import logging
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from bond_line import to_bond_line  # noqa: E402
from bond_line_encoders import BondLineDMPNN, BondLineGAT, BondLineGIN  # noqa: E402
from data import GraphDataset  # noqa: E402
from finetune import _load_checkpoint_safely  # noqa: E402
from model import model, validate_encoder  # noqa: E402
from reaction_data import get_graph_data  # noqa: E402
from run_approach1_grid import (  # noqa: E402
    _build_cells,
    _cell_fingerprint,
    build_grid_parser,
)
from utils import collate_reaction_graphs, set_seed  # noqa: E402
from test_bond_line import mol_to_data  # noqa: E402

DEVICE = torch.device("cpu")
ENC_DIM = 16
LINE_DIM = 320
ENCODER_CLASSES = {"gin": BondLineGIN, "gat": BondLineGAT, "dmpnn": BondLineDMPNN}


def build_encoder(gnn, depth=2, dim=ENC_DIM):
    set_seed(0)
    if gnn == "gat":
        return ENCODER_CLASSES[gnn](LINE_DIM, depth, dim, 0.1, gat_heads=4)
    return ENCODER_CLASSES[gnn](LINE_DIM, depth, dim, 0.1)


def bond_line_batch(smiles_list):
    return Batch.from_data_list([to_bond_line(mol_to_data(s)) for s in smiles_list])


def make_model(gnn, representation, dim=ENC_DIM, seed=0, **kwargs):
    set_seed(seed)
    return model(
        155,
        9,
        2,
        dim,
        0.0,
        num_attention_layer=2,
        num_heads=2,
        reaction_combine="concat",
        attention_on="reactants",
        reactant_tokens="comb",
        reactant_pooling="mean",
        head="linear",
        gnn=gnn,
        graph_representation=representation,
        gat_heads=4,
        **kwargs,
    )


def reaction_batch(rsmis, rmol_max, pmol_max, representation):
    rmol, pmol, reaction = get_graph_data(rsmis, rmol_max, pmol_max)
    reaction["y"] = np.zeros(len(rsmis))
    dataset = GraphDataset(
        rmol=rmol,
        pmol=pmol,
        reaction=reaction,
        graph_representation=representation,
    )
    loader = DataLoader(
        dataset, batch_size=len(rsmis), collate_fn=collate_reaction_graphs
    )
    return next(iter(loader)), rmol_max, pmol_max


def run_net(net, batch, rmol_max, pmol_max):
    rmol = list(batch[:rmol_max])
    pmol = list(batch[rmol_max: rmol_max + pmol_max])
    return net(rmol, pmol, batch[-4], batch[-3], DEVICE)


# ---------------------------------------------------------------------------
# 1. Encoder shapes, gradients, batching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_encoder_output_shape(gnn):
    encoder = build_encoder(gnn).eval()
    data = bond_line_batch(["CC(C)C", "CCO", "O"])
    with torch.no_grad():
        out = encoder(data)
    assert out.shape == (3, ENC_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_encoder_forward_backward_reaches_layers(gnn):
    encoder = build_encoder(gnn)
    data = bond_line_batch(["CC(C)C", "CCO"])
    out = encoder(data)
    out.sum().backward()

    projection = [p for n, p in encoder.named_parameters() if "project_node_feats" in n]
    assert projection and all(p.grad is not None for p in projection)
    assert any(float(p.grad.abs().sum()) > 0 for p in projection)

    layer_grads = [
        p
        for n, p in encoder.named_parameters()
        if "gnn_layers" in n or "W_in" in n
    ]
    assert layer_grads
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer_grads)
    assert any(float(p.grad.abs().sum()) > 0 for p in layer_grads)

    if gnn == "dmpnn":
        assert encoder.W_message is not None
        assert encoder.W_message.weight.grad is not None
        assert float(encoder.W_message.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_individual_matches_batched(gnn):
    encoder = build_encoder(gnn).eval()
    smiles = ["CC(C)C", "CCO", "O"]
    with torch.no_grad():
        batched = encoder(bond_line_batch(smiles))
        individual = torch.cat(
            [encoder(bond_line_batch([s])) for s in smiles], dim=0
        )
    torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_no_messages_cross_molecules(gnn):
    encoder = build_encoder(gnn).eval()
    with torch.no_grad():
        alone = encoder(bond_line_batch(["CC(C)C"]))
        together = encoder(bond_line_batch(["CC(C)C", "C1CC1.c1ccccc1"]))
    torch.testing.assert_close(alone[0], together[0], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Adapted D-MPNN reverse exclusion (hand-calculated states)
# ---------------------------------------------------------------------------


def _dmpnn_message(encoder, data):
    z = encoder.project_node_feats(data.x)
    src, dst = data.edge_index
    s0 = torch.relu(encoder.W_in(z[src]))
    incoming = z.new_zeros((z.shape[0], z.shape[1])).index_add(0, dst, s0)
    return s0, incoming[src] - s0[data.rev_edge]


def test_dmpnn_two_node_line_has_zero_nonbacktracking_message():
    encoder = build_encoder("dmpnn").eval()
    data = bond_line_batch(["CCO"])  # 2 transformed nodes, 2 directed edges
    assert data.edge_index.shape[1] == 2
    s0, message = _dmpnn_message(encoder, data)
    assert torch.count_nonzero(message) == 0
    # An update therefore leaves the initial state unchanged.
    state = torch.relu(s0 + encoder.W_message(message))
    torch.testing.assert_close(state, s0)


def test_dmpnn_chain_has_nonzero_propagation():
    encoder = build_encoder("dmpnn").eval()
    chain = bond_line_batch(["CC(C)C"])  # 3 nodes, 6 directed edges
    assert chain.edge_index.shape[1] == 6
    _, message = _dmpnn_message(encoder, chain)
    assert float(message.detach().abs().sum()) > 0

    triangle = bond_line_batch(["C1CC1"])  # 3 nodes, 6 directed edges
    _, tri_message = _dmpnn_message(encoder, triangle)
    assert float(tri_message.detach().abs().sum()) > 0


# ---------------------------------------------------------------------------
# 3. End-to-end model on the bond-line representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_model_prediction_shape(gnn):
    net = make_model(gnn, "bond_line").eval()
    batch, rmol_max, pmol_max = reaction_batch(
        ["CCO>>CCO", "CC(C)C.[Na+].O>>CC(=O)O"], 3, 1, "bond_line"
    )
    with torch.no_grad():
        pred, embeddings = run_net(net, batch, rmol_max, pmol_max)
    assert pred.shape == (2,)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 2 * ENC_DIM


def test_model_single_sample_batch():
    net = make_model("dmpnn", "bond_line").eval()
    batch, rmol_max, pmol_max = reaction_batch(["CCO>>CCO"], 1, 1, "bond_line")
    with torch.no_grad():
        pred, embeddings = run_net(net, batch, rmol_max, pmol_max)
    assert pred.shape == (1,)
    assert len(embeddings) == 1


def test_padding_does_not_affect_predictions():
    net = make_model("gin", "bond_line", seed=3).eval()
    base, r1, p1 = reaction_batch(["CCO>>CCO"], 1, 1, "bond_line")
    padded, r3, p3 = reaction_batch(["CCO.>>CCO"], 3, 1, "bond_line")
    with torch.no_grad():
        pred_base, _ = run_net(net, base, r1, p1)
        pred_padded, _ = run_net(net, padded, r3, p3)
    torch.testing.assert_close(pred_base, pred_padded, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_reactant_permutation_invariance(gnn):
    net = make_model(gnn, "bond_line", seed=5).eval()
    a, ra, pa = reaction_batch(
        ["CCO.[Na+].O>>CC(=O)O"], 3, 1, "bond_line"
    )
    b, rb, pb = reaction_batch(
        ["O.CCO.[Na+]>>CC(=O)O"], 3, 1, "bond_line"
    )
    with torch.no_grad():
        pred_a, _ = run_net(net, a, ra, pa)
        pred_b, _ = run_net(net, b, rb, pb)
    torch.testing.assert_close(pred_a, pred_b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_product_permutation_invariance(gnn):
    net = make_model(gnn, "bond_line", seed=6).eval()
    a, ra, pa = reaction_batch(["CCO>>CC=O.O"], 1, 2, "bond_line")
    b, rb, pb = reaction_batch(["CCO>>O.CC=O"], 1, 2, "bond_line")
    with torch.no_grad():
        pred_a, _ = run_net(net, a, ra, pa)
        pred_b, _ = run_net(net, b, rb, pb)
    torch.testing.assert_close(pred_a, pred_b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("gnn", ["gin", "gat", "dmpnn"])
def test_atom_renumbering_preserves_embeddings(gnn):
    # "CCO" and "OCC" are the same molecule with a different atom numbering.
    encoder = build_encoder(gnn).eval()
    with torch.no_grad():
        out_a = encoder(bond_line_batch(["CCO"]))
        out_b = encoder(bond_line_batch(["OCC"]))
    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Checkpoint reconstruction and compatibility
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip():
    net = make_model("gat", "bond_line", seed=1)
    rebuilt = model.from_config(dict(net.config))
    assert rebuilt.graph_representation == "bond_line"
    assert rebuilt.gnn_type == "gat"
    assert rebuilt.config["gat_heads"] == 4
    assert list(rebuilt.state_dict()) == list(net.state_dict())
    rebuilt.load_state_dict(net.state_dict())  # strict


def test_load_checkpoint_accepts_matching_config():
    net = make_model("dmpnn", "bond_line")
    _load_checkpoint_safely(
        net,
        {"model_config": dict(net.config), "model_state_dict": net.state_dict()},
        logging.getLogger("test"),
    )


def test_load_checkpoint_rejects_representation_mismatch():
    net = make_model("gat", "bond_line")
    mismatched = dict(net.config)
    mismatched["graph_representation"] = "atom"
    with pytest.raises(RuntimeError, match="graph representation"):
        _load_checkpoint_safely(
            net,
            {
                "model_config": mismatched,
                "model_state_dict": net.state_dict(),
            },
            logging.getLogger("test"),
        )


def test_load_checkpoint_rejects_feature_version():
    net = make_model("gin", "bond_line")
    stale = dict(net.config)
    stale["bond_line_feature_version"] = 999
    with pytest.raises(RuntimeError, match="bond_line_feature_version"):
        _load_checkpoint_safely(
            net,
            {"model_config": stale, "model_state_dict": net.state_dict()},
            logging.getLogger("test"),
        )


def test_legacy_checkpoint_config_defaults_to_atom():
    net = make_model("gine", "atom")
    legacy = {
        "node_in_feats": 155,
        "edge_in_feats": 9,
        "num_layer": 2,
        "emb_dim": ENC_DIM,
        "drop_ratio": 0.0,
        "num_attention_layer": 2,
        "num_heads": 2,
        "reaction_combine": "concat",
        "attention_on": "reactants",
        "reactant_tokens": "comb",
        "reactant_pooling": "mean",
        "head": "linear",
        "gnn": "gine",
    }
    _load_checkpoint_safely(
        net,
        {"model_config": legacy, "model_state_dict": net.state_dict()},
        logging.getLogger("test"),
    )


def test_from_config_legacy_defaults_to_atom():
    legacy = {
        "node_in_feats": 155,
        "edge_in_feats": 9,
        "num_layer": 2,
        "emb_dim": ENC_DIM,
        "drop_ratio": 0.0,
    }
    net = model.from_config(legacy)
    assert net.graph_representation == "atom"
    assert net.gnn_type == "gine"
    assert net.config["gat_heads"] == 4


# ---------------------------------------------------------------------------
# 5. Representation / encoder validation and the default atom path
# ---------------------------------------------------------------------------


def test_unsupported_combinations_rejected():
    with pytest.raises(ValueError, match="not valid"):
        model(155, 9, 2, ENC_DIM, 0.0, gnn="gin", graph_representation="atom")
    with pytest.raises(ValueError, match="not valid"):
        model(155, 9, 2, ENC_DIM, 0.0, gnn="gine", graph_representation="bond_line")
    with pytest.raises(ValueError, match="not valid"):
        validate_encoder("atom", "gin")
    validate_encoder("atom", "gine")
    validate_encoder("bond_line", "gin")


def test_default_atom_model_is_unchanged():
    net = model(155, 9, 2, ENC_DIM, 0.0, reactant_tokens="comb")
    assert net.graph_representation == "atom"
    assert net.gnn_type == "gine"
    assert net.config["node_in_feats"] == 155
    assert net.config["edge_in_feats"] == 9
    assert "gnn.project_node_feats.0.weight" in net.state_dict()


def test_bond_line_uses_source_widths_in_config():
    net = make_model("gin", "bond_line")
    assert net.config["node_in_feats"] == 155
    assert net.config["edge_in_feats"] == 9
    assert net.config["bond_line_feature_version"] == 1


# ---------------------------------------------------------------------------
# 6. Grid integration
# ---------------------------------------------------------------------------


def test_bond_line_cell_labels():
    labels = {
        c["label"] for c in _build_cells(("gin", "gat", "dmpnn"), "bond_line")
    }
    assert "comb-reactants-mean-linear-bond_line-gin" in labels
    assert "comb-reactants-mean-linear-bond_line-gat" in labels
    assert "comb-reactants-mean-linear-bond_line-dmpnn" in labels
    assert len(_build_cells(("gin", "gat", "dmpnn"), "bond_line")) == 72
    # Atom labels are preserved.
    atom_labels = {c["label"] for c in _build_cells(("gine",), "atom")}
    assert "comb-reactants-mean-linear" in atom_labels


def test_build_cells_rejects_unsupported_pairs():
    with pytest.raises(ValueError):
        _build_cells(("gin",), "atom")
    with pytest.raises(ValueError):
        _build_cells(("gine",), "bond_line")


def test_fingerprints_distinguish_configuration():
    cells = _build_cells(("gin", "gat"), "bond_line")
    gin_cell = next(c for c in cells if c["gnn"] == "gin")
    gat_cell = next(c for c in cells if c["gnn"] == "gat")
    args = build_grid_parser().parse_args(
        ["--graph_representation", "bond_line", "--gnn_types", "gin", "gat"]
    )
    assert _cell_fingerprint(args, gin_cell) != _cell_fingerprint(args, gat_cell)

    atom_cell = _build_cells(("gine",), "atom")[0]
    atom_args = build_grid_parser().parse_args(["--graph_representation", "atom"])
    assert _cell_fingerprint(atom_args, atom_cell) != _cell_fingerprint(
        args, gin_cell
    )

    seed_args = build_grid_parser().parse_args(
        ["--graph_representation", "bond_line", "--gnn_types", "gin", "--seed", "7"]
    )
    assert _cell_fingerprint(seed_args, gin_cell) != _cell_fingerprint(
        args, gin_cell
    )


# ---------------------------------------------------------------------------
# 7. Prediction selects the representation from the checkpoint
# ---------------------------------------------------------------------------


def _save_checkpoint(tmp_path, net, name="model.pt"):
    torch.save(
        {"model_state_dict": net.state_dict(), "model_config": dict(net.config)},
        os.path.join(str(tmp_path), name),
    )


def test_predict_selects_representation_from_checkpoint(tmp_path):
    from predict import predict

    net = make_model("gin", "bond_line", seed=2)
    _save_checkpoint(tmp_path, net)

    predictions = predict(
        ["CCO>>CC=O", "CC(=O)O.OCC>>CC(=O)OCC"],
        model_path=str(tmp_path) + os.sep,
        model_name="model.pt",
        device=0,
    )
    assert len(predictions) >= 1
    for pred, embeddings in predictions:
        assert pred.shape[0] == len(embeddings)


def test_predict_rejects_stale_feature_version(tmp_path):
    from predict import predict

    net = make_model("gin", "bond_line")
    config = dict(net.config)
    config["bond_line_feature_version"] = 999
    torch.save(
        {"model_state_dict": net.state_dict(), "model_config": config},
        os.path.join(str(tmp_path), "model.pt"),
    )
    with pytest.raises(ValueError, match="bond-line feature version"):
        predict(
            ["CCO>>CC=O"],
            model_path=str(tmp_path) + os.sep,
            model_name="model.pt",
            device=0,
        )
