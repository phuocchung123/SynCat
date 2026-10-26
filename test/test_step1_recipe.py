"""Tests for the step-1 training-recipe flags (CPU)."""

import argparse
import logging
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from data import ConcatGraphDataset  # noqa: E402
from finetune import _make_train_loader  # noqa: E402
from main_finetune import build_parser  # noqa: E402
from model import model  # noqa: E402
from recipe import (  # noqa: E402
    build_scheduler,
    step_scheduler,
    validate_recipe_args,
)
from run_approach1_grid import (  # noqa: E402
    LIGHT_CELLS,
    METRICS,
    _light_smoke_cells,
    _new_row,
    _select_cells,
    _write_csv_atomic,
    summarize_grid,
)
from training import train  # noqa: E402
from utils import collate_reaction_graphs  # noqa: E402
from validation import (  # noqa: E402
    evaluate_test_predictions,
    predict_mc,
    validation,
)
from test_regression_smoke import (  # noqa: E402
    EDGE_DIM,
    NODE_DIM,
    _build_synthetic_dataset,
    _synthetic_mol_dict,
)


@pytest.fixture(autouse=True)
def _close_logging():
    yield
    # Release log-file handles so pytest can remove the temporary directories.
    logging.shutdown()


def _args(tmp_path, **overrides):
    values = dict(
        monitor_folder=os.path.join(str(tmp_path), "monitor") + os.sep,
        image_folder=os.path.join(str(tmp_path), "images") + os.sep,
        patience=0,
        lr_scheduler="plateau",
        lr_milestones=[400, 450],
        lr_gamma=0.1,
        loss="huber",
        target_scaling="none",
        checkpoint_select="best_val",
        merge_valid_into_train=False,
        shuffle_train=False,
        mc_dropout_passes=0,
        clip_predictions=False,
        epochs=2,
        weight_decay=1e-4,
        batch_size=4,
        num_workers=0,
        seed=42,
    )
    values.update(overrides)
    args = argparse.Namespace(**values)
    os.makedirs(args.monitor_folder, exist_ok=True)
    os.makedirs(args.image_folder, exist_ok=True)
    return args


def _loader(dataset, batch_size=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
    )


def _tiny_net(drop_ratio=0.0):
    return model(NODE_DIM, EDGE_DIM, num_layer=1, emb_dim=8, drop_ratio=drop_ratio)


def _run_train(tmp_path, **overrides):
    args = _args(tmp_path, **overrides)
    dataset = _build_synthetic_dataset(np.linspace(10.0, 80.0, 8))
    loader = _loader(dataset)
    net = _tiny_net()
    model_path = os.path.join(str(tmp_path), "model.pt")
    val_loader = None if args.merge_valid_into_train else loader
    history = train(
        args,
        net,
        loader,
        val_loader,
        model_path,
        torch.device("cpu"),
        int(overrides.get("epochs", args.epochs)),
        1e-3,
        args.weight_decay,
    )
    return args, dataset, net, model_path, history


class _ConstantNet(torch.nn.Module):
    """Returns one constant standardized prediction for every sample."""

    def __init__(self, value):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.value = float(value)

    def forward(self, inputs_rmol, inputs_pmol, r_dummy, p_dummy, device):
        # r_dummy holds one entry per sample, each listing its reactant slots.
        batch_size = len(r_dummy)
        return torch.full((batch_size,), self.value), None


class _CountingNet(torch.nn.Module):
    """Returns the forward-pass number, so the MC mean is easy to predict."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.calls = 0

    def forward(self, inputs_rmol, inputs_pmol, r_dummy, p_dummy, device):
        self.calls += 1
        # r_dummy holds one entry per sample, each listing its reactant slots.
        batch_size = len(r_dummy)
        return torch.full((batch_size,), float(self.calls)), None


def test_parser_defaults_match_the_old_recipe():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.loss == "huber"
    assert args.lr_scheduler == "plateau"
    assert args.lr_milestones == [400, 450]
    assert args.lr_gamma == 0.1
    assert args.checkpoint_select == "best_val"
    assert args.target_scaling == "none"
    assert args.shuffle_train is False
    assert args.merge_valid_into_train is False
    assert args.mc_dropout_passes == 0
    assert args.clip_predictions is False
    assert args.weight_decay == 1e-4

    with pytest.raises(SystemExit):
        parser.parse_args(["--loss", "nll"])


def test_default_train_builds_plateau_and_huber(tmp_path):
    with mock.patch.object(
        torch.optim.lr_scheduler,
        "ReduceLROnPlateau",
        wraps=torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) as plateau, mock.patch.object(
        torch.nn, "HuberLoss", wraps=torch.nn.HuberLoss
    ) as huber:
        _args_obj, _dataset, _net, model_path, _history = _run_train(tmp_path, epochs=1)

    plateau.assert_called_once()
    huber.assert_called()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_type"] == "plateau"


def test_multistep_learning_rates():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    args = argparse.Namespace(
        lr_scheduler="multistep", lr_milestones=[1, 2], lr_gamma=0.1
    )
    scheduler = build_scheduler(args, optimizer)

    learning_rates = []
    for _ in range(3):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        step_scheduler(scheduler, "multistep")

    assert learning_rates == pytest.approx([1e-3, 1e-4, 1e-5])


def test_zscore_loss_uses_standardized_labels_metrics_original_units(tmp_path):
    args = _args(tmp_path, target_scaling="zscore", loss="mse")
    dataset = _build_synthetic_dataset(np.array([10.0, 20.0]))
    loader = _loader(dataset, batch_size=2)
    net = _ConstantNet(0.0)

    metrics, loss = validation(
        args,
        net,
        loader,
        torch.device("cpu"),
        torch.nn.MSELoss(),
        y_mean=10.0,
        y_std=2.0,
    )

    # labels standardized: [0, 5] -> MSE 12.5 (raw labels would give 250)
    assert loss == pytest.approx(12.5)
    # predictions de-standardized back: 0 * 2 + 10 = 10
    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(50.0))


def test_zscore_checkpoint_stores_scaler(tmp_path):
    _args_obj, dataset, _net, model_path, history = _run_train(
        tmp_path,
        target_scaling="zscore",
        loss="mse",
        checkpoint_select="last",
        lr_scheduler="none",
        epochs=2,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    assert checkpoint["target_scaler"]["mean"] == pytest.approx(
        float(np.mean(dataset.y))
    )
    assert checkpoint["target_scaler"]["std"] == pytest.approx(float(np.std(dataset.y)))
    # The logged MAE is in original units (tens of yield points), not z-scores.
    assert history["train"]["mae"][-1] > 1.0


def test_predict_destandardizes_checkpoint_predictions(tmp_path, monkeypatch):
    import predict as predict_module

    net = _tiny_net()
    with torch.no_grad():
        net.regressor.weight.zero_()
        net.regressor.bias.zero_()

    def save_checkpoint(name, scaler):
        path = os.path.join(str(tmp_path), name)
        payload = {
            "model_state_dict": net.state_dict(),
            "model_config": net.config,
        }
        if scaler is not None:
            payload["target_scaler"] = scaler
        torch.save(payload, path)

    def fake_graph_data(rsmi_lst, rmol_max_cnt, pmol_max_cnt):
        rng = np.random.default_rng(0)
        rmol = [_synthetic_mol_dict(len(rsmi_lst), rng) for _ in range(rmol_max_cnt)]
        pmol = [_synthetic_mol_dict(len(rsmi_lst), rng) for _ in range(pmol_max_cnt)]
        reaction = {"y": np.zeros(len(rsmi_lst)), "rsmi": list(rsmi_lst)}
        return rmol, pmol, reaction

    monkeypatch.setattr(predict_module, "get_graph_data", fake_graph_data)
    save_checkpoint("scaled.pt", {"mean": 45.0, "std": 20.0})
    scaled = predict_module.predict(
        ["C>>C"], model_path=str(tmp_path) + os.sep, model_name="scaled.pt"
    )
    assert float(scaled[0][0][0]) == pytest.approx(45.0)

    # A checkpoint without target_scaler keeps the old raw predictions.
    save_checkpoint("legacy.pt", None)
    legacy = predict_module.predict(
        ["C>>C"], model_path=str(tmp_path) + os.sep, model_name="legacy.pt"
    )
    assert float(legacy[0][0][0]) == pytest.approx(0.0)


def test_shuffle_train_uses_a_random_sampler(tmp_path):
    args = _args(tmp_path, shuffle_train=False, batch_size=5)
    dataset = _build_synthetic_dataset(np.arange(20.0))

    ordered = _make_train_loader(dataset, args)
    assert isinstance(ordered.sampler, torch.utils.data.SequentialSampler)
    ordered_labels = [v for batch in ordered for v in batch[-2].tolist()]
    assert ordered_labels == sorted(ordered_labels)

    args.shuffle_train = True
    shuffled = _make_train_loader(dataset, args)
    assert isinstance(shuffled.sampler, torch.utils.data.RandomSampler)
    torch.manual_seed(0)
    shuffled_labels = [v for batch in shuffled for v in batch[-2].tolist()]
    assert shuffled_labels != ordered_labels
    assert sorted(shuffled_labels) == sorted(ordered_labels)


def test_concat_dataset_merges_targets_and_metadata():
    train_part = _build_synthetic_dataset(np.arange(10.0))
    valid_part = _build_synthetic_dataset(np.arange(10.0, 20.0))
    merged = ConcatGraphDataset([train_part, valid_part])

    assert len(merged) == len(train_part) + len(valid_part)
    assert np.allclose(merged.y, np.arange(20.0))
    assert merged.rmol_max_cnt == train_part.rmol_max_cnt
    assert merged.pmol_max_cnt == train_part.pmol_max_cnt
    assert merged[15][-2] == pytest.approx(15.0)


def test_merge_trains_without_validation_loader(tmp_path):
    train_part = _build_synthetic_dataset(np.arange(10.0))
    valid_part = _build_synthetic_dataset(np.arange(10.0, 20.0))
    merged = ConcatGraphDataset([train_part, valid_part])
    args = _args(
        tmp_path,
        merge_valid_into_train=True,
        checkpoint_select="last",
        lr_scheduler="multistep",
        lr_milestones=[1, 2],
    )
    model_path = os.path.join(str(tmp_path), "model.pt")

    history = train(
        args,
        _tiny_net(),
        _loader(merged),
        None,
        model_path,
        torch.device("cpu"),
        2,
        1e-3,
        1e-4,
    )

    assert os.path.isfile(model_path)
    assert history["val"]["loss"] == []
    assert len(history["train"]["loss"]) == 2
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_type"] == "multistep"


def test_merge_requires_last_checkpoint_and_non_plateau_scheduler():
    with pytest.raises(ValueError):
        validate_recipe_args(
            argparse.Namespace(
                merge_valid_into_train=True,
                checkpoint_select="best_val",
                lr_scheduler="multistep",
                patience=0,
            )
        )
    with pytest.raises(ValueError):
        validate_recipe_args(
            argparse.Namespace(
                merge_valid_into_train=True,
                checkpoint_select="last",
                lr_scheduler="plateau",
                patience=0,
            )
        )
    with pytest.raises(ValueError):
        validate_recipe_args(
            argparse.Namespace(
                merge_valid_into_train=False,
                checkpoint_select="last",
                lr_scheduler="none",
                patience=2,
            )
        )


def test_last_checkpoint_holds_the_final_epoch_weights(tmp_path):
    _args_obj, _dataset, net, model_path, _history = _run_train(
        tmp_path, checkpoint_select="last", lr_scheduler="none", epochs=3
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    assert checkpoint["epoch"] == 2
    for key, value in net.state_dict().items():
        assert torch.equal(checkpoint["model_state_dict"][key], value)


def test_last_checkpoint_does_not_step_plateau_on_validation(tmp_path):
    """With last-epoch selection, validation must not change the learning rate."""
    with mock.patch.object(
        torch.optim.lr_scheduler.ReduceLROnPlateau, "step"
    ) as plateau_step:
        _run_train(tmp_path, checkpoint_select="last", epochs=2)

    plateau_step.assert_not_called()


def test_mc_dropout_is_the_mean_of_passes_and_ends_in_eval_mode(tmp_path):
    args = _args(tmp_path)
    dataset = _build_synthetic_dataset(np.arange(8.0))
    loader = _loader(dataset, batch_size=8)
    net = _CountingNet()

    metrics, labels, preds = predict_mc(
        args, net, loader, torch.device("cpu"), 4, y_mean=0.0, y_std=1.0
    )

    # Passes returned 1, 2, 3, 4, so their mean is 2.5 for every sample.
    assert preds == pytest.approx([2.5] * 8)
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(np.arange(8.0) - 2.5)))
    assert labels == pytest.approx(list(np.arange(8.0)))
    assert net.calls == 4
    assert net.training is False
    assert net.dropout.training is False


def test_mc_off_and_no_clip_gives_exactly_the_deterministic_predictions(tmp_path):
    args = _args(tmp_path, mc_dropout_passes=0, clip_predictions=False)
    dataset = _build_synthetic_dataset(np.arange(8.0))
    loader = _loader(dataset, batch_size=8)
    net = _ConstantNet(0.0)

    result = evaluate_test_predictions(
        args, net, loader, torch.device("cpu"), y_mean=10.0, y_std=2.0
    )

    assert result["preds"] == result["det_preds"]
    for metric, value in result["det_metrics"].items():
        assert result["metrics"][metric] == pytest.approx(value, nan_ok=True)
    assert result["preds"] == pytest.approx([10.0] * 8)


def test_evaluate_test_predictions_combines_mc_and_clipping(tmp_path):
    args = _args(tmp_path, mc_dropout_passes=3, clip_predictions=True)
    dataset = _build_synthetic_dataset(np.arange(8.0))
    loader = _loader(dataset, batch_size=8)
    net = _ConstantNet(2.0)

    result = evaluate_test_predictions(
        args,
        net,
        loader,
        torch.device("cpu"),
        y_mean=10.0,
        y_std=20.0,
        y_max=1.0,
    )

    # Deterministic 50 is clipped to the fraction range; MC of a constant is 50.
    assert result["det_preds"] == pytest.approx([50.0] * 8)
    assert result["preds"] == pytest.approx([1.0] * 8)
    assert result["metrics"]["mae"] == pytest.approx(
        np.mean(np.abs(np.arange(8.0) - 1.0))
    )


def test_clipping_uses_the_yield_range(tmp_path):
    args = _args(tmp_path, mc_dropout_passes=0, clip_predictions=True)
    dataset = _build_synthetic_dataset(np.arange(8.0))
    loader = _loader(dataset, batch_size=8)
    net = _ConstantNet(2.0)

    result = evaluate_test_predictions(
        args,
        net,
        loader,
        torch.device("cpu"),
        y_mean=10.0,
        y_std=20.0,
        y_max=1.0,
    )

    # standardized 2 -> de-standardized 50, clipped to the fraction range
    assert result["preds"] == pytest.approx([1.0] * 8)
    assert result["det_preds"] == pytest.approx([50.0] * 8)


def test_report16_selects_the_sixteen_reported_cells():
    cells = _select_cells(["report16"])
    labels = {cell["label"] for cell in cells}

    assert len(cells) == 16
    assert all(cell["reactant_pooling"] in ("mean", "rn") for cell in cells)
    assert "ind-reactants-mean-linear" in labels  # G1
    assert "ind-none-rn-mlp" in labels  # G6
    assert "comb-reactants-rn-mlp" in labels  # G12
    assert "comb-none-rn-mlp" in labels  # G16
    assert "comb-reactants-sum-linear" not in labels


def test_light_smoke_profile_keeps_four_representative_cells():
    cells = _light_smoke_cells(_select_cells(["report16"]))

    assert len(cells) == 4
    assert {cell["label"] for cell in cells} == set(LIGHT_CELLS)
    assert all(cell["reactant_pooling"] in ("mean", "rn") for cell in cells)


def test_progress_csv_is_written_atomically(tmp_path):
    path = str(tmp_path / "progress.csv")
    _write_csv_atomic(pd.DataFrame({"cell": ["G1", "G6"]}), path)

    assert os.path.isfile(path)
    assert not os.path.exists(path + ".tmp")
    assert pd.read_csv(path)["cell"].tolist() == ["G1", "G6"]


def test_grid_rows_and_summary_have_the_step1_columns(tmp_path):
    args = _args(tmp_path, loss="mse", target_scaling="zscore", epochs=7)
    row = _new_row(args, "bh", "fullcv01_split70", "cv", _select_cells(["report16"])[0])

    for column in (
        "n_train",
        "epochs",
        "loss",
        "target_scaling",
        "lr_scheduler",
        "lr_milestones",
        "weight_decay",
        "shuffle_train",
        "merge_valid_into_train",
        "checkpoint_select",
        "mc_dropout_passes",
        "clip_predictions",
        "test_r2_det",
        "test_mae_det",
        "test_rmse_det",
    ):
        assert column in row

    assert row["epochs"] == 7
    assert row["loss"] == "mse"
    assert row["target_scaling"] == "zscore"

    row.update(
        {
            "status": "success",
            "train_r2": 0.9,
            "train_mae": 1.0,
            "train_rmse": 2.0,
            "test_r2": 0.8,
            "test_mae": 1.5,
            "test_rmse": 3.0,
            "test_r2_det": 0.75,
            "test_mae_det": 1.6,
            "test_rmse_det": 3.1,
        }
    )
    summary = summarize_grid(pd.DataFrame([row]))

    for metric in ("test_r2_det", "test_mae_det", "test_rmse_det"):
        assert metric in METRICS
        assert metric + "_mean" in summary.columns
        assert metric + "_std" in summary.columns
    assert summary.loc[0, "test_r2_det_mean"] == pytest.approx(0.75)
