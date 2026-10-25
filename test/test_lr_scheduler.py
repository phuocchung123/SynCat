"""Tests for the --lr_scheduler option (ported from confirm_app3).

The original test covered plateau/none; it is extended here with multistep.
"""

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


@pytest.fixture(autouse=True)
def _close_logging():
    yield
    # Release log-file handles so pytest can remove the temporary directories.
    logging.shutdown()


def test_lr_scheduler_parser_defaults_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).lr_scheduler == "plateau"
    assert parser.parse_args(["--lr_scheduler", "none"]).lr_scheduler == "none"
    assert parser.parse_args(["--lr_scheduler", "multistep"]).lr_scheduler == (
        "multistep"
    )
    assert parser.parse_args([]).lr_milestones == [400, 450]
    assert parser.parse_args([]).lr_gamma == 0.1
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
        lr_milestones=[400, 450],
        lr_gamma=0.1,
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


def test_constant_lr_does_not_build_a_scheduler(tmp_path):
    """--lr_scheduler none must not construct ReduceLROnPlateau at all."""
    with mock.patch.object(
        torch.optim.lr_scheduler, "ReduceLROnPlateau"
    ) as scheduler_cls:
        checkpoint = _run_train(tmp_path, "none")
    scheduler_cls.assert_not_called()
    # No schedule means no scheduler state is stored either.
    assert checkpoint["scheduler_state_dict"] is None
    assert checkpoint["scheduler_type"] == "none"


def test_plateau_still_builds_a_scheduler(tmp_path):
    with mock.patch.object(
        torch.optim.lr_scheduler,
        "ReduceLROnPlateau",
        wraps=torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) as scheduler_cls:
        checkpoint = _run_train(tmp_path, "plateau")
    scheduler_cls.assert_called_once()
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["scheduler_type"] == "plateau"


def test_multistep_builds_and_steps_without_arguments(tmp_path):
    with mock.patch.object(
        torch.optim.lr_scheduler,
        "MultiStepLR",
        wraps=torch.optim.lr_scheduler.MultiStepLR,
    ) as scheduler_cls:
        checkpoint = _run_train(tmp_path, "multistep")
    scheduler_cls.assert_called_once()
    assert scheduler_cls.call_args.kwargs["milestones"] == [400, 450]
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["scheduler_type"] == "multistep"
