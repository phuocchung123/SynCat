"""
Simple figures for the step-1 grid (reaction_yield_nn training recipe).

The script reads the merged ``step1_per_split.csv`` (from
``merge_grid_results.py``) and, optionally, the report's per-split CSV. It
writes easy-to-read figures to ``--output_dir``:

    step1_leaderboard_bh_cv.png     configurations sorted by test R² (BH CV)
    step1_leaderboard_bh_tests.png  same for the four named BH tests
    step1_leaderboard_suzuki.png    same for Suzuki
    step1_report_vs_step1.png       report test R² versus step-1 test R²
    step1_train_test_gap.png        train R² versus test R² (underfitting)
    step1_delta_heatmap.png         step-1 minus report, per cell and dataset
    step1_vs_literature.png         best step-1 cell vs report/Yield-BERT/DRFP
    step1_baselines.csv             best step-1 cell next to literature values

Green bars beat the report, orange bars fall behind it. Literature values for
Yield-BERT and DRFP + gradient boosting are drawn as dashed reference lines;
they come from other splits (see the note in each figure) and are not a
like-for-like re-evaluation.

Usage (from src/):
    python plot_step1_results.py \
        --step1_csv ../logs/step1/step1_per_split.csv \
        --report_csv D:/workspace/syncat_folder/20261009_analyse_syncat_/approach1_per_split.csv \
        --output_dir ../logs/step1/figures/
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from compare_with_report import REPORT_ID_BY_CELL  # noqa: E402
from visualization import (  # noqa: E402
    AXIS,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

DATASET_LABELS = {"bh": "Buchwald-Hartwig", "suzuki": "Suzuki-Miyaura"}
REPORT_ORDER = {
    report_id: i for i, report_id in enumerate(REPORT_ID_BY_CELL.values())
}

# Published means on random 70/30 splits (report plan, table on p. 61):
# Yield-BERT: Schwaller et al., MLST 2021, BH 0.95 ± 0.005, SM 0.81 ± 0.01.
# DRFP + gradient boosting: Probst et al., Digital Discovery 2022,
# BH 0.95 ± 0.005, SM 0.85 ± 0.01. Kwon et al. (2022) is an approximate range.
LITERATURE_BASELINES = {
    "bh": [
        {"label": "Yield-BERT (2021)", "mean": 0.95, "std": 0.005},
        {"label": "DRFP + GB (2022)", "mean": 0.95, "std": 0.005},
        {"label": "Kwon et al. (2022)", "mean": 0.96, "lo": 0.95, "hi": 0.97},
    ],
    "suzuki": [
        {"label": "Yield-BERT (2021)", "mean": 0.81, "std": 0.01},
        {"label": "DRFP + GB (2022)", "mean": 0.85, "std": 0.01},
        {"label": "Kwon et al. (2022)", "mean": 0.87, "lo": 0.85, "hi": 0.89},
    ],
}
BASELINE_COLORS = {
    "Yield-BERT (2021)": "#7b3294",
    "DRFP + GB (2022)": "#d95f02",
    "Kwon et al. (2022)": "#1b9e77",
}
BASELINE_NOTE = (
    "Literature reference lines are from other splits and are shown for context, "
    "not as a like-for-like re-evaluation."
)


def _read_success(path: str) -> pd.DataFrame:
    """Reads a per-split CSV and keeps successful rows."""
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "success"].copy()
    return frame


def _require_columns(frame: pd.DataFrame, columns, path: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("%s is missing columns %s" % (path, missing))


def load_step1(path: str) -> pd.DataFrame:
    """Loads the merged step-1 table and keeps the 16 report cells."""
    frame = _read_success(path)
    _require_columns(
        frame, ["dataset", "split", "cell", "train_r2", "test_r2"], path
    )
    if "test_r2_det" not in frame.columns:
        frame["test_r2_det"] = np.nan
    frame["report_id"] = frame["cell"].map(REPORT_ID_BY_CELL)
    frame = frame[frame["report_id"].notna()].copy()
    if frame.empty:
        raise ValueError("%s has no rows for the 16 report cells" % path)
    return frame


def load_report(path: str) -> pd.DataFrame:
    """Loads the report table and renames its metrics to ``report_*``."""
    frame = _read_success(path)
    _require_columns(frame, ["dataset", "split", "cell", "train_r2", "test_r2"], path)
    frame = frame.rename(
        columns={"train_r2": "report_train_r2", "test_r2": "report_test_r2"}
    )
    frame["report_id"] = frame["cell"].map(REPORT_ID_BY_CELL)
    return frame[frame["report_id"].notna()].copy()


def pair_with_report(step1: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, split, cell), with step-1 minus report deltas."""
    keys = ["dataset", "split", "cell"]
    report_columns = keys + ["report_train_r2", "report_test_r2"]
    paired = step1.merge(
        report[report_columns], on=keys, how="inner", validate="one_to_one"
    )
    paired["delta_mc"] = paired["test_r2"] - paired["report_test_r2"]
    paired["delta_det"] = paired["test_r2_det"] - paired["report_test_r2"]
    return paired


def without_report(step1: pd.DataFrame) -> pd.DataFrame:
    """Adds empty report columns so the plotting code has one shape."""
    paired = step1.copy()
    for column in ("report_train_r2", "report_test_r2", "delta_mc", "delta_det"):
        paired[column] = np.nan
    return paired


def _subset(paired: pd.DataFrame, kind: str) -> pd.DataFrame:
    """BH CV, BH Tests 1-4, or Suzuki rows."""
    if kind == "bh_cv":
        mask = (paired["dataset"] == "bh") & paired["split"].str.startswith("fullcv")
    elif kind == "bh_test":
        mask = (paired["dataset"] == "bh") & paired["split"].str.startswith("test")
    elif kind == "suzuki":
        mask = paired["dataset"] == "suzuki"
    else:
        raise ValueError("unknown subset %r" % kind)
    return paired[mask].copy()


def _mean_std(values) -> tuple:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    return float(clean.mean()), std


def aggregate(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over splits, one row per report cell."""
    metrics = (
        "train_r2",
        "test_r2",
        "test_r2_det",
        "report_train_r2",
        "report_test_r2",
        "delta_mc",
        "delta_det",
    )
    rows = []
    for (report_id, cell), group in paired.groupby(["report_id", "cell"]):
        row = {"report_id": report_id, "cell": cell, "n_splits": len(group)}
        for metric in metrics:
            mean, std = _mean_std(group[metric]) if metric in group else (np.nan, np.nan)
            row[metric + "_mean"] = mean
            row[metric + "_std"] = std
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        columns = ["report_id", "cell", "n_splits"]
        columns += [metric + suffix for metric in metrics for suffix in ("_mean", "_std")]
        return pd.DataFrame(columns=columns)
    table["_order"] = table["report_id"].map(REPORT_ORDER)
    table = table.sort_values(["_order"]).drop(columns="_order")
    return table.reset_index(drop=True)


def _style_axes(ax, xlabel: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def _save(fig, path: str, dpi: int) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_colors(deltas) -> list:
    """Green for a clear win, orange for a clear loss, blue otherwise."""
    colors = []
    for delta in deltas:
        if pd.isna(delta):
            colors.append("#9ecae1")
        elif delta > 0.005:
            colors.append("#1baf7a")
        elif delta < -0.005:
            colors.append("#d95f02")
        else:
            colors.append("#9ecae1")
    return colors


def _baseline_handles(baselines) -> list:
    handles = []
    for baseline in baselines:
        color = BASELINE_COLORS.get(baseline["label"], INK_MUTED)
        handles.append(
            Line2D([0], [0], color=color, linestyle="--", linewidth=1.2,
                   label=baseline["label"] + " (literature)")
        )
    return handles


def plot_leaderboard(
    paired: pd.DataFrame,
    kind: str,
    title: str,
    baselines,
    path: str,
    with_report: bool,
    dpi: int = 200,
) -> str:
    """Horizontal bars sorted by step-1 test R², with report and baselines."""
    table = aggregate(_subset(paired, kind))
    table = table.dropna(subset=["test_r2_mean"]).sort_values("test_r2_mean")
    if table.empty:
        # Partial result files (e.g. before the BH tests finish) are common.
        return None

    labels = [
        "%s  %s" % (row.report_id, row.cell) for row in table.itertuples()
    ]
    y = np.arange(len(table))
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.42 * len(table) + 1.5)), facecolor=SURFACE
    )
    ax.barh(
        y,
        table["test_r2_mean"],
        color=_bar_colors(table["delta_mc_mean"]),
        edgecolor=SURFACE,
        height=0.68,
    )
    ax.errorbar(
        table["test_r2_mean"],
        y,
        xerr=np.nan_to_num(table["test_r2_std"].to_numpy(), nan=0.0),
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=1,
        capsize=2,
        zorder=3,
    )
    ax.scatter(
        table["test_r2_det_mean"],
        y,
        marker="o",
        facecolors="none",
        edgecolors=INK_PRIMARY,
        s=26,
        linewidths=1,
        zorder=4,
        label="step 1 deterministic",
    )
    if with_report:
        ax.scatter(
            table["report_test_r2_mean"],
            y,
            marker="x",
            color=INK_PRIMARY,
            s=34,
            linewidths=1.2,
            zorder=5,
            label="report",
        )
    for baseline in baselines:
        color = BASELINE_COLORS.get(baseline["label"], INK_MUTED)
        ax.axvline(
            baseline["mean"], color=color, linestyle="--", linewidth=1.2, zorder=1
        )
        if "lo" in baseline and "hi" in baseline:
            ax.axvspan(
                baseline["lo"], baseline["hi"], color=color, alpha=0.08,
                linewidth=0, zorder=0,
            )

    for yi, value in zip(y, table["test_r2_mean"]):
        ax.text(
            value + (0.006 if value >= 0 else -0.006),
            yi,
            "%.3f" % value,
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=INK_SECONDARY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK_PRIMARY)
    low = min(0.0, float(np.nanmin(table["test_r2_mean"])) - 0.03)
    ax.set_xlim(low, 1.0)
    note = "Green = better than report; orange = worse; dashed = literature (other splits)."
    _style_axes(ax, "Test R² (mean ± SD over splits)")
    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=12)

    handles = [
        Patch(facecolor="#9ecae1", label="step 1 MC (bar)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=INK_PRIMARY, markersize=6,
               label="step 1 deterministic"),
    ]
    if with_report:
        handles.append(
            Line2D([0], [0], marker="x", color=INK_PRIMARY, linestyle="none",
                   markersize=6, label="report")
        )
    handles.extend(_baseline_handles(baselines))
    fig.subplots_adjust(bottom=0.30)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    fig.text(0.01, 0.005, note, fontsize=7, color=INK_MUTED)
    return _save(fig, path, dpi)


def plot_scatter(paired: pd.DataFrame, path: str, dpi: int = 200) -> str:
    """Report test R² versus step-1 test R² on the same splits."""
    specs = [
        ("BH CV (10 splits)", "bh_cv"),
        ("BH Tests 1-4", "bh_test"),
        ("Suzuki (10 splits)", "suzuki"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, (title, kind) in zip(axes, specs):
        data = _subset(paired, kind).dropna(subset=["report_test_r2", "test_r2"])
        if data.empty:
            ax.text(0.5, 0.5, "no paired rows", ha="center", va="center",
                    transform=ax.transAxes, color=INK_MUTED)
            _style_axes(ax, "Report test R²", "Step 1 test R² (MC)")
            ax.grid(axis="both", color=GRID, linewidth=0.8)
            ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
            continue
        improved = int((data["test_r2"] > data["report_test_r2"]).sum())
        ax.scatter(
            data["report_test_r2"],
            data["test_r2"],
            s=22,
            color="#2a78d6",
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=0.4,
            zorder=3,
        )
        low = float(min(data["report_test_r2"].min(), data["test_r2"].min())) - 0.03
        high = float(max(data["report_test_r2"].max(), data["test_r2"].max())) + 0.03
        low, high = max(-0.05, low), min(1.05, high)
        ax.plot([low, high], [low, high], color=INK_MUTED, linestyle=":",
                linewidth=1, zorder=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        _style_axes(ax, "Report test R²", "Step 1 test R² (MC)")
        ax.grid(axis="both", color=GRID, linewidth=0.8)
        ax.set_title(
            "%s — step 1 wins %d/%d" % (title, improved, len(data)),
            loc="left", color=INK_PRIMARY, fontsize=11,
        )
    fig.text(
        0.01, -0.02,
        "Points above the dotted diagonal are better after the recipe change; "
        "each point is one split × configuration.",
        fontsize=7, color=INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_train_test_gap(paired: pd.DataFrame, path: str, dpi: int = 200) -> str:
    """Train R² versus test R²; closer to the diagonal means less underfitting."""
    specs = [("BH CV", "bh_cv"), ("Suzuki", "suzuki")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax, (title, kind) in zip(axes, specs):
        table = aggregate(_subset(paired, kind))
        data = table.dropna(subset=["train_r2_mean", "test_r2_mean"])
        if data.empty:
            ax.text(0.5, 0.5, "no successful rows", ha="center", va="center",
                    transform=ax.transAxes, color=INK_MUTED)
            ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
            continue

        has_report = data["report_test_r2_mean"].notna().any()
        if has_report:
            ax.scatter(
                data["report_test_r2_mean"],
                data["report_train_r2_mean"],
                facecolors="none",
                edgecolors=INK_MUTED,
                s=34,
                linewidths=1,
                label="report",
                zorder=3,
            )
            for row in data.itertuples():
                if pd.isna(row.report_test_r2_mean) or pd.isna(row.report_train_r2_mean):
                    continue
                ax.annotate(
                    "",
                    xy=(row.test_r2_mean, row.train_r2_mean),
                    xytext=(row.report_test_r2_mean, row.report_train_r2_mean),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": GRID,
                        "linewidth": 0.8,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                    zorder=1,
                )
        ax.scatter(
            data["test_r2_mean"],
            data["train_r2_mean"],
            color="#2a78d6",
            s=30,
            edgecolors=SURFACE,
            linewidths=0.4,
            label="step 1",
            zorder=4,
        )
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _style_axes(ax, "Test R²", "Train R²")
        ax.grid(axis="both", color=GRID, linewidth=0.8)
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    fig.text(
        0.01, -0.02,
        "The vertical distance above the dotted diagonal is underfitting "
        "(train R² minus test R²). Arrows go from the report to step 1.",
        fontsize=7, color=INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, path, dpi)


def _delta_matrix(paired: pd.DataFrame) -> tuple:
    """Rows G1-G16, columns BH CV / BH Tests / Suzuki, values mean Δ test R²."""
    specs = [("BH CV", "bh_cv"), ("BH Tests", "bh_test"), ("Suzuki", "suzuki")]
    cells = sorted(REPORT_ID_BY_CELL.items(), key=lambda item: REPORT_ORDER[item[1]])
    matrix = np.full((len(cells), len(specs)), np.nan)
    for col, (_, kind) in enumerate(specs):
        table = aggregate(_subset(paired, kind)).set_index("cell")
        for row, (cell, _) in enumerate(cells):
            if cell in table.index:
                matrix[row, col] = table.loc[cell, "delta_mc_mean"]
    return cells, [title for title, _ in specs], matrix


def plot_delta_heatmap(paired: pd.DataFrame, path: str, dpi: int = 200) -> str:
    """Heatmap of step-1 minus report test R²; green wins, red loses."""
    cells, columns, matrix = _delta_matrix(paired)
    if np.all(np.isnan(matrix)):
        raise ValueError("no paired deltas for the heatmap")
    limit = max(0.1, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7.5, 8), facecolor=SURFACE)
    image = ax.imshow(
        matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto"
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                ax.text(col, row, "n/a", ha="center", va="center",
                        fontsize=7, color=INK_MUTED)
            else:
                ax.text(col, row, "%+.3f" % value, ha="center", va="center",
                        fontsize=7, color=INK_PRIMARY)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=9, color=INK_PRIMARY)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(
        ["%s  %s" % (report_id, cell) for cell, report_id in cells],
        fontsize=7.5,
        color=INK_PRIMARY,
    )
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.set_title(
        "Δ test R²: step 1 (MC) minus report", loc="left",
        color=INK_PRIMARY, fontsize=12,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Δ R² (green = better)", color=INK_SECONDARY, fontsize=9)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.text(
        0.01, 0.01,
        "BH CV and Suzuki average the 10 random splits; BH Tests average Tests 1-4.",
        fontsize=7, color=INK_MUTED,
    )
    return _save(fig, path, dpi)


def plot_literature_comparison(
    paired: pd.DataFrame,
    baselines_by_dataset: dict,
    path: str,
    with_report: bool,
    dpi: int = 200,
) -> str:
    """Best step-1 cell next to the report and the literature baselines."""
    specs = [
        ("bh", "bh_cv", "Buchwald-Hartwig CV"),
        ("suzuki", "suzuki", "Suzuki-Miyaura"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax, (dataset, kind, title) in zip(axes, specs):
        table = aggregate(_subset(paired, kind)).dropna(subset=["test_r2_mean"])
        if table.empty:
            ax.text(0.5, 0.5, "no successful rows", ha="center", va="center",
                    transform=ax.transAxes, color=INK_MUTED)
            ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
            continue
        best = table.sort_values("test_r2_mean").iloc[-1]
        labels = ["Step 1 best\n(%s)" % best["report_id"]]
        values = [best["test_r2_mean"]]
        errors = [best["test_r2_std"]]
        colors = ["#2a78d6"]
        if with_report and not pd.isna(best["report_test_r2_mean"]):
            labels.append("Report best\n(%s)" % best["report_id"])
            values.append(best["report_test_r2_mean"])
            errors.append(np.nan)
            colors.append(INK_MUTED)
        for baseline in baselines_by_dataset.get(dataset, []):
            labels.append(baseline["label"])
            values.append(baseline["mean"])
            errors.append(baseline.get("std", np.nan))
            colors.append(BASELINE_COLORS.get(baseline["label"], INK_MUTED))

        x = np.arange(len(values))
        ax.bar(x, values, color=colors, width=0.62, edgecolor=SURFACE)
        ax.errorbar(
            x,
            values,
            yerr=np.nan_to_num(errors, nan=0.0),
            fmt="none",
            ecolor=INK_MUTED,
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.015, "%.3f" % value, ha="center",
                    fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        _style_axes(ax, "")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylabel("Test R²", color=INK_SECONDARY, fontsize=9)
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
    fig.text(
        0.01, 0.01,
        "The step-1 bar is the best cell on these splits; literature bars are "
        "from other splits (Yield-BERT 0.95/0.81, DRFP + GB 0.95/0.85).",
        fontsize=7, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path, dpi)


def write_baseline_table(
    paired: pd.DataFrame,
    baselines_by_dataset: dict,
    path: str,
) -> str:
    """Best step-1 cell next to each literature baseline, per dataset."""
    rows = []
    for dataset, kind in (("bh", "bh_cv"), ("suzuki", "suzuki")):
        table = aggregate(_subset(paired, kind)).dropna(subset=["test_r2_mean"])
        if table.empty:
            continue
        best = table.sort_values("test_r2_mean").iloc[-1]
        for baseline in baselines_by_dataset.get(dataset, []):
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline["label"],
                    "baseline_r2": baseline["mean"],
                    "best_cell": best["cell"],
                    "best_report_id": best["report_id"],
                    "best_step1_r2_mc": best["test_r2_mean"],
                    "delta_vs_baseline": best["test_r2_mean"] - baseline["mean"],
                    "note": BASELINE_NOTE,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def _load_baselines(path: str) -> dict:
    """Loads custom baselines (dataset,label,mean,std[,lo,hi]) from a CSV."""
    frame = pd.read_csv(path)
    _require_columns(frame, ["dataset", "label", "mean"], path)
    baselines = {}
    for row in frame.itertuples():
        entry = {"label": row.label, "mean": float(row.mean)}
        if hasattr(row, "std") and not pd.isna(row.std):
            entry["std"] = float(row.std)
        if hasattr(row, "lo") and hasattr(row, "hi"):
            if not pd.isna(row.lo) and not pd.isna(row.hi):
                entry["lo"] = float(row.lo)
                entry["hi"] = float(row.hi)
        baselines.setdefault(row.dataset, []).append(entry)
    return baselines


def make_figures(
    step1_csv: str,
    report_csv: str = None,
    output_dir: str = None,
    baselines_csv: str = None,
    with_baselines: bool = True,
    dpi: int = 200,
) -> list:
    """Builds every figure; returns the written paths."""
    step1 = load_step1(step1_csv)
    paired = (
        pair_with_report(step1, load_report(report_csv))
        if report_csv
        else without_report(step1)
    )
    with_report = report_csv is not None
    if not with_baselines:
        baselines = {}
    elif baselines_csv:
        baselines = _load_baselines(baselines_csv)
    else:
        baselines = LITERATURE_BASELINES
    output_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(step1_csv)), "figures"
    )
    os.makedirs(output_dir, exist_ok=True)

    leaderboards = [
        (
            "bh_cv",
            "Buchwald-Hartwig CV: step-1 test R² by configuration",
            "step1_leaderboard_bh_cv.png",
            baselines.get("bh", []),
        ),
        (
            "bh_test",
            "BH Tests 1-4: step-1 test R² by configuration",
            "step1_leaderboard_bh_tests.png",
            [],
        ),
        (
            "suzuki",
            "Suzuki-Miyaura: step-1 test R² by configuration",
            "step1_leaderboard_suzuki.png",
            baselines.get("suzuki", []),
        ),
    ]
    paths = []
    for kind, title, name, reference in leaderboards:
        path = plot_leaderboard(
            paired,
            kind,
            title,
            reference,
            os.path.join(output_dir, name),
            with_report,
            dpi,
        )
        if path is not None:
            paths.append(path)
    paths.append(
        plot_train_test_gap(
            paired, os.path.join(output_dir, "step1_train_test_gap.png"), dpi
        )
    )
    if with_report:
        paths.append(
            plot_scatter(
                paired, os.path.join(output_dir, "step1_report_vs_step1.png"), dpi
            )
        )
        paths.append(
            plot_delta_heatmap(
                paired, os.path.join(output_dir, "step1_delta_heatmap.png"), dpi
            )
        )
    if with_baselines:
        paths.append(
            plot_literature_comparison(
                paired,
                baselines,
                os.path.join(output_dir, "step1_vs_literature.png"),
                with_report,
                dpi,
            )
        )
        paths.append(
            write_baseline_table(
                paired, baselines, os.path.join(output_dir, "step1_baselines.csv")
            )
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1_csv",
        default="../logs/step1/step1_per_split.csv",
        help="merged step-1 per-split CSV",
    )
    parser.add_argument(
        "--report_csv",
        default=None,
        help="the report's approach1_per_split.csv (enables paired figures)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="figure folder (default: <step1_csv folder>/figures)",
    )
    parser.add_argument(
        "--baselines_csv",
        default=None,
        help="optional CSV with dataset,label,mean,std[,lo,hi] to replace the "
        "built-in Yield-BERT/DRFP/Kwon reference values",
    )
    parser.add_argument(
        "--no_baselines",
        action="store_true",
        help="do not draw literature reference lines",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    paths = make_figures(
        args.step1_csv,
        report_csv=args.report_csv,
        output_dir=args.output_dir,
        baselines_csv=args.baselines_csv,
        with_baselines=not args.no_baselines,
        dpi=args.dpi,
    )
    print("Wrote %d files:" % len(paths))
    for path in paths:
        print("  %s" % path)


if __name__ == "__main__":
    main()
