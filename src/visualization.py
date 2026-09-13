import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e6e5e1"
AXIS = "#c9c8c3"

# Line style + marker give each split a second, non-color encoding.
SPLIT_STYLES = {
    "train": {"label": "Train", "color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "val": {"label": "Validation", "color": "#eb6834", "linestyle": "--", "marker": "s"},
    "test": {"label": "Test", "color": "#1baf7a", "linestyle": ":", "marker": "^"},
}

METRIC_LABELS = {
    "loss": "Huber loss",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "pearson": "Pearson r",
}


def init_history() -> dict:
    """
    Creates an empty per-epoch history for train/val/test loss and metrics.

    Returns
    -------
    dict
        {"epoch": [], "train": {metric: []}, "val": {...}, "test": {...}}
    """
    history = {"epoch": []}
    for split in SPLIT_STYLES:
        history[split] = {metric: [] for metric in METRIC_LABELS}
    return history


def record_epoch(history: dict, split: str, loss: float, metrics: dict) -> None:
    """
    Appends one epoch's loss and regression metrics for a split to the history.

    Parameters
    ----------
    history : dict
        History created by `init_history`.
    split : str
        One of "train", "val", "test".
    loss : float
        Mean loss over the epoch.
    metrics : dict
        Output of `validation.compute_regression_metrics`.
    """
    history[split]["loss"].append(float(loss))
    for metric in ("mae", "rmse", "r2", "pearson"):
        value = metrics[metric]
        history[split][metric].append(None if value is None else float(value))


def save_history(history: dict, path: str) -> None:
    """Writes the history to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _style_axes(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=12)


def _plot_metric(ax, history: dict, metric: str) -> None:
    epochs = history["epoch"]
    markevery = max(1, len(epochs) // 15)
    for split, style in SPLIT_STYLES.items():
        values = history[split][metric]
        if not values or len(values) != len(epochs):
            continue
        y = np.array([np.nan if v is None else v for v in values], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(
            epochs,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            markevery=markevery,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            linewidth=2,
            label=style["label"],
        )
    _style_axes(ax, METRIC_LABELS[metric], METRIC_LABELS[metric])


def _save(fig, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history: dict, save_path: str) -> None:
    """
    Plots train/validation/test loss per epoch and marks the best validation epoch.

    Parameters
    ----------
    history : dict
        History created by `init_history` and filled by `record_epoch`.
    save_path : str
        Output image path.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _plot_metric(ax, history, "loss")

    val_loss = history["val"]["loss"]
    if val_loss:
        best = int(np.argmin(val_loss))
        best_epoch = history["epoch"][best]
        ax.axvline(best_epoch, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=0)
        ax.text(
            best_epoch,
            0.98,
            " best val %.4f (epoch %d)" % (val_loss[best], best_epoch),
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        frameon=False,
        labelcolor=INK_SECONDARY,
        fontsize=9,
    )
    _save(fig, save_path)


def plot_metric_curves(history: dict, save_path: str) -> None:
    """
    Plots MAE, RMSE, R² and Pearson r per epoch for each split in a 2x2 grid.

    Parameters
    ----------
    history : dict
        History created by `init_history` and filled by `record_epoch`.
    save_path : str
        Output image path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, metric in zip(axes.flat, ("mae", "rmse", "r2", "pearson")):
        _plot_metric(ax, history, metric)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels) or 1,
        frameon=False,
        labelcolor=INK_SECONDARY,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, save_path)


def plot_training_history(history: dict, image_folder: str) -> None:
    """
    Saves the loss curve and metric curves for the current history.

    Parameters
    ----------
    history : dict
        History created by `init_history` and filled by `record_epoch`.
    image_folder : str
        Folder where `loss_curve.png` and `metric_curves.png` are written.
    """
    plot_loss_curve(history, os.path.join(image_folder, "loss_curve.png"))
    plot_metric_curves(history, os.path.join(image_folder, "metric_curves.png"))


def plot_parity(labels: list, preds: list, metrics: dict, save_path: str) -> None:
    """
    Scatter plot of predicted vs observed yield on an evaluation set.

    Parameters
    ----------
    labels : list
        Observed yields.
    preds : list
        Predicted yields.
    metrics : dict
        Output of `validation.compute_regression_metrics` for the same data.
    save_path : str
        Output image path.
    """
    labels_arr = np.asarray(labels, dtype=float)
    preds_arr = np.asarray(preds, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=SURFACE)
    lo = min(labels_arr.min(), preds_arr.min())
    hi = max(labels_arr.max(), preds_arr.max())
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(
        labels_arr,
        preds_arr,
        s=16,
        color=SPLIT_STYLES["test"]["color"],
        alpha=0.6,
        edgecolors=SURFACE,
        linewidths=0.5,
        zorder=2,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    def fmt(value):
        return "n/a" if value is None else "%.4f" % value

    ax.text(
        0.03,
        0.97,
        "n = %d\nMAE = %s\nRMSE = %s\nR² = %s\nPearson r = %s"
        % (
            len(labels_arr),
            fmt(metrics["mae"]),
            fmt(metrics["rmse"]),
            fmt(metrics["r2"]),
            fmt(metrics["pearson"]),
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
        bbox={"facecolor": SURFACE, "edgecolor": GRID, "alpha": 0.9, "boxstyle": "round,pad=0.4"},
        zorder=3,
    )

    _style_axes(ax, "Predicted yield", "Test set: predicted vs observed")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_xlabel("Observed yield", color=INK_SECONDARY, fontsize=10)
    _save(fig, save_path)
