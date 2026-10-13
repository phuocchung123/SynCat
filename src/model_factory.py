"""
Shared construction, reconstruction and checkpoint metadata for yield models.

Every place that builds or reloads a reaction-yield network goes through this
factory so that the architecture recorded in a checkpoint ("yield_arch") always
selects the same class on reload:

    legacy         the existing attention/GINE model (default), returned
                   directly and untouched
    hierarchical4  the four-level atoms -> molecules -> side groups -> reaction
                   model
    hierarchical3  the three-level atoms -> molecules -> side groups model with
                   a concatenated [Reactants, Products] readout

The selector is stored in the checkpoint's "model_config" but is stripped before
the legacy `model.from_config()` is called, so the legacy model's config schema
and constructor stay exactly as they were.
"""

from typing import Union

from hierarchical_model import Hierarchical3YieldModel, Hierarchical4YieldModel
from model import model

# Architectures that share the hierarchical packing/inputs and therefore need
# the strict reaction-string validation.
HIERARCHICAL_ARCHS = ("hierarchical3", "hierarchical4")
YIELD_ARCHS = ("legacy",) + HIERARCHICAL_ARCHS

# Key added to a legacy checkpoint's "model_config"; older checkpoints do not
# carry it and therefore default to the legacy architecture.
ARCH_KEY = "yield_arch"


def is_hierarchical_arch(arch: str) -> bool:
    """
    Whether `arch` (as stored in a checkpoint or given on the CLI) is one of the
    hierarchical architectures.

    Parameters
    ----------
    arch : str
        Architecture identifier; may be None.

    Returns
    -------
    bool
        True for "hierarchical3" and "hierarchical4".
    """
    return (arch or "legacy") in HIERARCHICAL_ARCHS


def build_yield_model(
    args, node_dim: int, edge_dim: int
) -> Union[model, Hierarchical3YieldModel, Hierarchical4YieldModel]:
    """
    Build the yield model selected by `args.yield_arch`.

    The legacy branch is byte-for-byte the previous `finetune._build_model`, so
    the legacy initialisation randomness and parameter names are preserved and
    no hierarchical module is instantiated.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options; needs `layer`, `emb_dim`, `dropout` and the legacy
        attention/head options for the legacy branch, and the hierarchy options
        for the hierarchical branches.
    node_dim : int
        Node feature dimension.
    edge_dim : int
        Edge feature dimension.

    Returns
    -------
    torch.nn.Module
        The freshly initialised model.
    """
    arch = getattr(args, "yield_arch", "legacy") or "legacy"
    if arch == "legacy":
        return model(
            node_dim,
            edge_dim,
            args.layer,
            args.emb_dim,
            args.dropout,
            num_attention_layer=getattr(args, "attention_layer", 1),
            num_heads=getattr(args, "num_heads", 1),
            reaction_combine=getattr(args, "reaction_combine", "concat"),
            attention_on=getattr(args, "attention_on", "reactants"),
            reactant_tokens=getattr(args, "reactant_tokens", "ind"),
            reactant_pooling=getattr(args, "reactant_pooling", "mean"),
            head=getattr(args, "head", "linear"),
        )
    if arch == "hierarchical4":
        return Hierarchical4YieldModel(
            node_dim,
            edge_dim,
            args.emb_dim,
            args.dropout,
            hierarchy_passes=getattr(args, "hierarchy_passes", 2),
            hierarchy_gine_layers=getattr(args, "hierarchy_gine_layers", 2),
            hierarchy_no_feedback=getattr(args, "hierarchy_no_feedback", False),
        )
    if arch == "hierarchical3":
        return Hierarchical3YieldModel(
            node_dim,
            edge_dim,
            args.emb_dim,
            args.dropout,
            hierarchy_passes=getattr(args, "hierarchy_passes", 2),
            hierarchy_gine_layers=getattr(args, "hierarchy_gine_layers", 2),
            hierarchy_no_feedback=getattr(args, "hierarchy_no_feedback", False),
            head=getattr(args, "head", "linear"),
        )
    raise ValueError(
        "unknown yield architecture %r; expected one of %s"
        % (arch, list(YIELD_ARCHS))
    )


def model_from_config(
    config: dict,
) -> Union[model, Hierarchical3YieldModel, Hierarchical4YieldModel]:
    """
    Rebuild a model from a checkpoint's "model_config".

    A missing `yield_arch` means a checkpoint written before the selector
    existed and therefore selects the legacy model; the selector is removed
    before the legacy `model.from_config()` is called.

    Parameters
    ----------
    config : dict
        The stored architecture config.

    Returns
    -------
    torch.nn.Module
        A freshly initialised model with that architecture.
    """
    config = dict(config)
    arch = config.pop(ARCH_KEY, "legacy") or "legacy"
    if arch == "legacy":
        return model.from_config(config)
    if arch == "hierarchical4":
        return Hierarchical4YieldModel.from_config(config)
    if arch == "hierarchical3":
        return Hierarchical3YieldModel.from_config(config)
    raise ValueError(
        "unknown yield architecture %r; expected one of %s"
        % (arch, list(YIELD_ARCHS))
    )


def checkpoint_config(net) -> dict:
    """
    Architecture config to store in a checkpoint.

    The legacy `net.config` schema is left untouched; the architecture selector
    is only added to the returned copy.

    Parameters
    ----------
    net : torch.nn.Module
        The model whose `config` is recorded.

    Returns
    -------
    dict
        The config with "yield_arch" guaranteed to be present.
    """
    config = dict(net.config)
    config.setdefault(ARCH_KEY, getattr(net, "yield_arch", "legacy"))
    return config


def describe_model(net) -> str:
    """
    Short, log-friendly description of the effective architecture.

    Parameters
    ----------
    net : torch.nn.Module
        The model to describe.

    Returns
    -------
    str
        One line naming the architecture and its effective settings.
    """
    arch = getattr(net, "yield_arch", "legacy")
    if arch == "hierarchical3":
        return (
            "yield_arch=hierarchical3, passes=%d, gine_layers_per_pass=%d, "
            "feedback=%s, emb_dim=%d, dropout=%s, head=%s"
            % (
                net.hierarchy_passes,
                net.hierarchy_gine_layers,
                net.feedback_enabled,
                net.emb_dim,
                net.config["drop_ratio"],
                net.head,
            )
        )
    if arch == "hierarchical4":
        return (
            "yield_arch=hierarchical4, passes=%d, gine_layers_per_pass=%d, "
            "feedback=%s, emb_dim=%d, dropout=%s"
            % (
                net.hierarchy_passes,
                net.hierarchy_gine_layers,
                net.feedback_enabled,
                net.emb_dim,
                net.config["drop_ratio"],
            )
        )
    return (
        "yield_arch=legacy, num_layer=%d, emb_dim=%d, dropout=%s, "
        "attention_on=%s, attention_layer=%d, num_heads=%d, "
        "reaction_combine=%s, reactant_tokens=%s, reactant_pooling=%s, head=%s"
        % (
            net.config["num_layer"],
            net.config["emb_dim"],
            net.config["drop_ratio"],
            net.config["attention_on"],
            net.config["num_attention_layer"],
            net.config["num_heads"],
            net.config["reaction_combine"],
            net.config["reactant_tokens"],
            net.config["reactant_pooling"],
            net.config["head"],
        )
    )
