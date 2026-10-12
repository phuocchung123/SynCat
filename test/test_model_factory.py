"""
Focused tests for the shared model factory, checkpoint metadata and the
preservation of the legacy architecture.

These tests only need torch; they do not featurize real reactions, so they run
even when RDKit's native libraries cannot be loaded.
"""

import argparse
import copy
import logging
import os
import sys

import torch

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from finetune import _load_checkpoint_safely  # noqa: E402
from gin import GIN  # noqa: E402
from hierarchical_model import Hierarchical4YieldModel  # noqa: E402
from model import model  # noqa: E402
from model_factory import (  # noqa: E402
    build_yield_model,
    checkpoint_config,
    describe_model,
    model_from_config,
)
from utils import set_seed  # noqa: E402

NODE_DIM = 155
EDGE_DIM = 9


def legacy_args(**overrides):
    """Namespace with the legacy defaults plus an explicit yield_arch."""
    values = dict(
        layer=2,
        emb_dim=16,
        dropout=0.0,
        attention_layer=1,
        num_heads=1,
        reaction_combine="concat",
        attention_on="reactants",
        reactant_tokens="ind",
        reactant_pooling="mean",
        head="linear",
        yield_arch="legacy",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def hierarchical_args(**overrides):
    values = dict(
        emb_dim=16,
        dropout=0.0,
        hierarchy_passes=2,
        hierarchy_gine_layers=2,
        hierarchy_no_feedback=False,
        yield_arch="hierarchical4",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


# ----------------------------------------------------------------------------
# legacy construction preservation
# ----------------------------------------------------------------------------


def test_legacy_path_matches_direct_model_and_uses_no_extra_randomness():
    """build_yield_model(legacy) is exactly a direct model() with the same seed."""
    set_seed(123)
    via_factory = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    set_seed(123)
    direct = model(
        NODE_DIM,
        EDGE_DIM,
        2,
        16,
        0.0,
        num_attention_layer=1,
        num_heads=1,
        reaction_combine="concat",
        attention_on="reactants",
        reactant_tokens="ind",
        reactant_pooling="mean",
        head="linear",
    )
    assert isinstance(via_factory, model)
    assert not isinstance(via_factory, Hierarchical4YieldModel)
    assert not hasattr(via_factory, "gine_stacks")
    factory_state = via_factory.state_dict()
    direct_state = direct.state_dict()
    assert list(factory_state) == list(direct_state)
    for key in factory_state:
        torch.testing.assert_close(factory_state[key], direct_state[key])


def test_legacy_config_schema_is_unchanged():
    """The legacy model's own config keeps its schema; the selector is added at save."""
    net = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    assert "yield_arch" not in net.config
    saved = checkpoint_config(net)
    assert saved["yield_arch"] == "legacy"
    # the original config object is not mutated
    assert "yield_arch" not in net.config


def test_legacy_checkpoint_round_trip():
    """A legacy config with a selector rebuilds the original legacy model."""
    net = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    rebuilt = model_from_config(checkpoint_config(net))
    assert type(rebuilt) is model
    rebuilt.load_state_dict(net.state_dict(), strict=True)


def test_missing_arch_defaults_to_legacy():
    """A checkpoint written before the selector existed reloads as legacy."""
    net = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    config = dict(net.config)  # no yield_arch at all
    rebuilt = model_from_config(config)
    assert type(rebuilt) is model


def test_hierarchical_selection_builds_hierarchical_model():
    net = build_yield_model(hierarchical_args(), NODE_DIM, EDGE_DIM)
    assert isinstance(net, Hierarchical4YieldModel)
    assert net.hierarchy_passes == 2
    assert net.hierarchy_gine_layers == 2
    assert net.feedback_enabled is True


def test_unknown_architecture_is_rejected_both_ways():
    import pytest

    with pytest.raises(ValueError, match="unknown yield architecture"):
        build_yield_model(legacy_args(yield_arch="nope"), NODE_DIM, EDGE_DIM)
    with pytest.raises(ValueError, match="unknown yield architecture"):
        model_from_config({"yield_arch": "nope"})


def test_describe_model_reports_effective_settings():
    legacy = describe_model(build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM))
    assert "yield_arch=legacy" in legacy
    hier = describe_model(build_yield_model(hierarchical_args(), NODE_DIM, EDGE_DIM))
    assert "yield_arch=hierarchical4" in hier
    assert "passes=2" in hier


# ----------------------------------------------------------------------------
# GIN state-dict layout is preserved
# ----------------------------------------------------------------------------


def test_gin_state_dict_layout_preserved():
    gin = GIN(NODE_DIM, EDGE_DIM, 3, 16, 0.0)
    keys = set(gin.state_dict())
    assert "project_node_feats.0.weight" in keys
    assert "project_node_feats.0.bias" in keys
    assert "project_edge_feats.0.weight" in keys
    assert {
        "gnn_layers.%d.nn.0.weight" % i for i in range(3)
    }.issubset(keys)
    assert {
        "gnn_layers.%d.nn.2.weight" % i for i in range(3)
    }.issubset(keys)
    # exactly the expected shapes (no extra normalisation/bias parameters)
    assert gin.gnn_layers[0].nn[0].in_features == 16
    assert gin.gnn_layers[0].nn[2].out_features == 16


# ----------------------------------------------------------------------------
# checkpoint dispatch in _load_checkpoint_safely
# ----------------------------------------------------------------------------


def test_cross_architecture_load_is_rejected():
    logger = logging.getLogger("test_factory")
    legacy_net = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    hier_net = build_yield_model(hierarchical_args(), NODE_DIM, EDGE_DIM)

    legacy_ckpt = {
        "model_config": checkpoint_config(legacy_net),
        "model_state_dict": legacy_net.state_dict(),
    }
    hier_ckpt = {
        "model_config": checkpoint_config(hier_net),
        "model_state_dict": hier_net.state_dict(),
    }

    import pytest

    with pytest.raises(RuntimeError, match="architecture"):
        _load_checkpoint_safely(hier_net, legacy_ckpt, logger)
    with pytest.raises(RuntimeError, match="architecture"):
        _load_checkpoint_safely(legacy_net, hier_ckpt, logger)


def test_hierarchical_checkpoint_pass_mismatch_is_rejected():
    logger = logging.getLogger("test_factory")
    net_k2 = build_yield_model(hierarchical_args(hierarchy_passes=2), NODE_DIM, EDGE_DIM)
    net_k3 = build_yield_model(hierarchical_args(hierarchy_passes=3), NODE_DIM, EDGE_DIM)
    ckpt = {
        "model_config": checkpoint_config(net_k3),
        "model_state_dict": net_k3.state_dict(),
    }
    import pytest

    with pytest.raises(RuntimeError, match="different architecture"):
        _load_checkpoint_safely(net_k2, ckpt, logger)


def test_matching_hierarchical_checkpoint_loads():
    logger = logging.getLogger("test_factory")
    net = build_yield_model(hierarchical_args(), NODE_DIM, EDGE_DIM)
    ckpt = {
        "model_config": checkpoint_config(net),
        "model_state_dict": net.state_dict(),
    }
    target = build_yield_model(hierarchical_args(), NODE_DIM, EDGE_DIM)
    _load_checkpoint_safely(target, ckpt, logger)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_legacy_head_warm_start_still_allowed():
    """The legacy head warm-start behaviour (linear -> mlp) is preserved."""
    logger = logging.getLogger("test_factory")
    linear = build_yield_model(legacy_args(head="linear"), NODE_DIM, EDGE_DIM)
    ckpt = {
        "model_config": checkpoint_config(linear),
        "model_state_dict": linear.state_dict(),
    }
    mlp = build_yield_model(legacy_args(head="mlp"), NODE_DIM, EDGE_DIM)
    # must not raise: only the regressor keys differ
    _load_checkpoint_safely(mlp, copy.deepcopy(ckpt), logger)


def test_missing_arch_legacy_checkpoint_loads_into_legacy_model():
    logger = logging.getLogger("test_factory")
    net = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    config = dict(net.config)  # old checkpoint: no selector
    ckpt = {"model_config": config, "model_state_dict": net.state_dict()}
    target = build_yield_model(legacy_args(), NODE_DIM, EDGE_DIM)
    _load_checkpoint_safely(target, ckpt, logger)
