"""Tests for the fixed-learning-rate option (--lr_scheduler)."""

import argparse
import logging
import os
import sys
from unittest import mock

import numpy as np
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

import training as training_module  # noqa: E402
from main_finetune import build_parser  # noqa: E402
from model import model  # noqa: E402
from utils import collate_reaction_graphs  # noqa: E402
from test_regression_smoke import (  # noqa: E402
    EDGE_DIM,
    NODE_DIM,
    _build_synthetic_dataset,
)


def test_lr_scheduler_parser_defaults_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).lr_scheduler == "plateau"
    assert parser.parse_args(["--lr_scheduler", "none"]).lr_scheduler == "none"
    assert parser.parse_args(["--lr", "1e-4"]).lr == 1e-4
    with pytest.raises(SystemExit):
        parser.parse_args(["--lr_scheduler", "cosine"])


def _train_args(tmp_dir, lr_scheduler):
    monitor = os.path.join(str(tmp_dir), "monitor") + os.sep
    images = os.path.join(str(tmp_dir), "images") + os.sep
    os.makedirs(monitor, exist_ok=True)
    os.makedirs(images, exist_ok=True)
    return argparse.Namespace(
        monitor_folder=monitor,
        image_folder=images,
        patience=0,
        lr_scheduler=lr_scheduler,
    )


def _tiny_loader():
    dataset = _build_synthetic_dataset(
        np.linspace(10.0, 80.0, 8), rmol_max_cnt=2, pmol_max_cnt=1
    )
    return DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_reaction_graphs
    )


def _run_train(tmp_dir, lr_scheduler):
    loader = _tiny_loader()
    net = model(NODE_DIM, EDGE_DIM, num_layer=1, emb_dim=8, drop_ratio=0.0)
    model_path = os.path.join(str(tmp_dir), "model.pt")
    try:
        training_module.train(
            _train_args(tmp_dir, lr_scheduler),
            net,
            loader,
            loader,
            model_path,
            torch.device("cpu"),
            2,
            1e-4,
            1e-4,
        )
        return torch.load(model_path, map_location="cpu", weights_only=False)
    finally:
        # Release the log file handle so the temp directory can be removed.
        logging.shutdown()


def test_constant_lr_does_not_build_a_scheduler(tmp_path):
    """--lr_scheduler none must not construct ReduceLROnPlateau at all."""
    with mock.patch.object(
        torch.optim.lr_scheduler, "ReduceLROnPlateau"
    ) as scheduler_cls:
        checkpoint = _run_train(tmp_path, "none")
    scheduler_cls.assert_not_called()
    # No schedule means no scheduler state is stored either.
    assert checkpoint["scheduler_state_dict"] is None


def test_plateau_still_builds_a_scheduler(tmp_path):
    with mock.patch.object(
        torch.optim.lr_scheduler,
        "ReduceLROnPlateau",
        wraps=torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) as scheduler_cls:
        checkpoint = _run_train(tmp_path, "plateau")
    scheduler_cls.assert_called_once()
    assert checkpoint["scheduler_state_dict"] is not None
