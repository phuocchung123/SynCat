"""
Training-recipe options shared by the entry points, `training.py` and
`validation.py`.

The defaults of every function here reproduce the recipe used before step 1:
Huber loss on raw targets, no batch shuffling, `ReduceLROnPlateau(0.3, patience
10, min 1e-6)`, and checkpoint selection on the best validation loss.
"""

import numpy as np
import torch

LOSSES = ("huber", "mse")
TARGET_SCALINGS = ("none", "zscore")
LR_SCHEDULERS = ("plateau", "multistep", "none")
CHECKPOINT_SELECTIONS = ("best_val", "last")
MERGED_SCHEDULERS = ("multistep", "none")


def validate_recipe_args(args) -> None:
    """
    Rejects combinations of recipe flags that cannot be trained meaningfully.

    Raises
    ------
    ValueError
        - ``--checkpoint_select last`` together with ``--patience > 0``.
        - ``--merge_valid_into_train`` without a last-epoch checkpoint or with
          the plateau scheduler (there is no validation loss to monitor).
    """
    checkpoint_select = getattr(args, "checkpoint_select", "best_val")
    patience = int(getattr(args, "patience", 0))
    if checkpoint_select == "last" and patience > 0:
        raise ValueError(
            "--checkpoint_select last cannot be combined with --patience > 0: "
            "there is no best-validation checkpoint to keep"
        )

    if not getattr(args, "merge_valid_into_train", False):
        return
    if checkpoint_select != "last":
        raise ValueError(
            "--merge_valid_into_train requires --checkpoint_select last "
            "(merging removes the validation set used for model selection)"
        )
    scheduler = getattr(args, "lr_scheduler", "plateau")
    if scheduler not in MERGED_SCHEDULERS:
        raise ValueError(
            "--merge_valid_into_train requires --lr_scheduler %s (got %r): "
            "ReduceLROnPlateau needs a validation loss"
            % (" or ".join(MERGED_SCHEDULERS), scheduler)
        )


def target_scaler(args, dataset) -> tuple:
    """
    Mean and population standard deviation of the targets actually trained on.

    Returns ``(0.0, 1.0)`` unless ``--target_scaling zscore`` is set. A constant
    target column (std 0) keeps a unit scale so the loss stays finite.
    """
    if getattr(args, "target_scaling", "none") != "zscore":
        return 0.0, 1.0
    y = np.asarray(dataset.y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return mean, std


def target_scaler_from_checkpoint(checkpoint: dict) -> tuple:
    """
    ``(mean, std)`` stored in a checkpoint, or the identity for old checkpoints.
    """
    scaler = checkpoint.get("target_scaler") or {}
    return float(scaler.get("mean", 0.0)), float(scaler.get("std", 1.0))


def scale_targets(labels, y_mean: float, y_std: float):
    """Standardizes raw labels for the loss."""
    return (labels - y_mean) / y_std


def unscale_predictions(preds, y_mean: float, y_std: float):
    """Maps standardized model outputs back to the original target units."""
    return preds * y_std + y_mean


def build_loss_fn(args):
    """``torch.nn.HuberLoss()`` (default) or ``torch.nn.MSELoss()``."""
    if getattr(args, "loss", "huber") == "mse":
        return torch.nn.MSELoss()
    return torch.nn.HuberLoss()


def build_scheduler(args, optimizer, scheduler_state=None, checkpoint_type=None):
    """
    Builds the learning-rate scheduler selected by ``--lr_scheduler``.

    ``scheduler_state`` is restored only when the checkpoint it came from used
    the same scheduler type (``checkpoint_type``, plateau for old checkpoints).
    """
    kind = getattr(args, "lr_scheduler", "plateau")
    if kind == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(getattr(args, "lr_milestones", [400, 450])),
            gamma=float(getattr(args, "lr_gamma", 0.1)),
        )
    elif kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=10, min_lr=1e-6
        )
    else:
        return None

    if scheduler_state is not None and (checkpoint_type or "plateau") == kind:
        scheduler.load_state_dict(scheduler_state)
    return scheduler


def step_scheduler(scheduler, kind: str, val_loss=None) -> None:
    """Steps the scheduler once per epoch, as each scheduler type expects."""
    if scheduler is None:
        return
    if kind == "plateau":
        if val_loss is not None:
            scheduler.step(val_loss)
    else:
        scheduler.step()


def target_upper_bound(y) -> float:
    """100 for percent targets, 1.0 for fraction targets."""
    max_abs = float(np.nanmax(np.abs(np.asarray(y, dtype=float))))
    return 100.0 if max_abs > 1.5 else 1.0
