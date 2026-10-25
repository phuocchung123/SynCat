"""Tests for the step-1 result figures (headless, CPU)."""

import os
import sys

import pandas as pd
import pytest

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plot_step1_results import (  # noqa: E402
    LITERATURE_BASELINES,
    make_figures,
)

CELLS = [
    ("ind-reactants-mean-linear", "G1"),
    ("ind-none-rn-mlp", "G6"),
    ("comb-reactants-mean-mlp", "G10"),
    ("comb-reactants-rn-mlp", "G12"),
]
SPLITS = {
    "bh": ["fullcv01_split70", "fullcv02_split70", "test1", "test2"],
    "suzuki": ["split_0", "split_1"],
}


def _make_rows(report: bool) -> list:
    rows = []
    for dataset, splits in SPLITS.items():
        for split_index, split in enumerate(splits):
            for cell_index, (cell, _) in enumerate(CELLS):
                train_r2 = 0.90 + 0.01 * cell_index
                test_r2 = 0.55 + 0.05 * cell_index + 0.02 * split_index
                row = {
                    "dataset": dataset,
                    "split": split,
                    "split_kind": "test" if split.startswith("test") else "cv",
                    "cell": cell,
                    "status": "success",
                    "train_r2": train_r2 if report else train_r2 + 0.05,
                    "test_r2": test_r2 if report else test_r2 + 0.08,
                }
                if not report:
                    row["test_r2_det"] = test_r2 + 0.06
                rows.append(row)
    return rows


@pytest.fixture()
def csv_pair(tmp_path):
    step1_path = tmp_path / "step1_per_split.csv"
    report_path = tmp_path / "report_per_split.csv"
    pd.DataFrame(_make_rows(report=False)).to_csv(step1_path, index=False)
    pd.DataFrame(_make_rows(report=True)).to_csv(report_path, index=False)
    return str(step1_path), str(report_path)


def test_make_figures_with_report(csv_pair, tmp_path):
    step1_csv, report_csv = csv_pair
    out = tmp_path / "figures"

    paths = make_figures(step1_csv, report_csv, str(out), dpi=60)

    expected = {
        "step1_leaderboard_bh_cv.png",
        "step1_leaderboard_bh_tests.png",
        "step1_leaderboard_suzuki.png",
        "step1_train_test_gap.png",
        "step1_report_vs_step1.png",
        "step1_delta_heatmap.png",
        "step1_vs_literature.png",
        "step1_baselines.csv",
    }
    assert expected == {os.path.basename(path) for path in paths}
    for path in paths:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    baselines = pd.read_csv(out / "step1_baselines.csv")
    assert {"Yield-BERT (2021)", "DRFP + GB (2022)"}.issubset(
        set(baselines["baseline"])
    )
    assert set(baselines["dataset"]) == {"bh", "suzuki"}


def test_make_figures_without_report(csv_pair, tmp_path):
    step1_csv, _ = csv_pair
    out = tmp_path / "figures_no_report"

    paths = make_figures(step1_csv, None, str(out), dpi=60)

    names = {os.path.basename(path) for path in paths}
    assert "step1_leaderboard_bh_cv.png" in names
    assert "step1_train_test_gap.png" in names
    assert "step1_report_vs_step1.png" not in names
    assert "step1_delta_heatmap.png" not in names


def test_literature_baselines_cover_both_datasets():
    assert LITERATURE_BASELINES["bh"]
    assert LITERATURE_BASELINES["suzuki"]
    for dataset in ("bh", "suzuki"):
        labels = {entry["label"] for entry in LITERATURE_BASELINES[dataset]}
        assert "Yield-BERT (2021)" in labels
        assert "DRFP + GB (2022)" in labels
