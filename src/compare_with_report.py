"""
Side-by-side comparison of the step-1 grid with the original report.

Reads the merged step-1 per-split table (``step1_per_split.csv``) and the
report's per-split table, pairs runs by ``(dataset, split, cell)``, and prints
and saves three tables:

    - BH CV (10 splits, mean ± SD over splits)
    - BH Tests 1-4 (one row per test and cell)
    - Suzuki (10 splits, mean ± SD over splits)

For every cell the table shows its report id (G1-G16), the report's train/test
R², the step-1 train R², the step-1 deterministic and MC test R², and the paired
difference from the report (step 1 minus report, averaged over the same splits).

Usage (from src/):
    python compare_with_report.py \
        D:/workspace/syncat_folder/20261009_analyse_syncat_/approach1_per_split.csv \
        --step1_csv ../logs/step1/step1_per_split.csv \
        --output_dir ../logs/step1/
"""

import argparse
import os

import numpy as np
import pandas as pd

# Report ids of the 16 reported cells (sum pooling was added later).
REPORT_ID_BY_CELL = {
    "ind-reactants-mean-linear": "G1",
    "ind-reactants-mean-mlp": "G2",
    "ind-none-mean-linear": "G3",
    "ind-none-mean-mlp": "G4",
    "ind-none-rn-linear": "G5",
    "ind-none-rn-mlp": "G6",
    "ind-reactants-rn-linear": "G7",
    "ind-reactants-rn-mlp": "G8",
    "comb-reactants-mean-linear": "G9",
    "comb-reactants-mean-mlp": "G10",
    "comb-reactants-rn-linear": "G11",
    "comb-reactants-rn-mlp": "G12",
    "comb-none-mean-linear": "G13",
    "comb-none-mean-mlp": "G14",
    "comb-none-rn-linear": "G15",
    "comb-none-rn-mlp": "G16",
}
REPORT_ID_ORDER = {report_id: i for i, report_id in enumerate(REPORT_ID_BY_CELL.values())}

KEY_COLUMNS = ["dataset", "split", "cell"]
DELTA_COLUMNS = ["delta_test_r2_det", "delta_test_r2_mc"]
AGGREGATED_COLUMNS = [
    "report_train_r2",
    "report_test_r2",
    "step1_train_r2",
    "step1_test_r2_det",
    "step1_test_r2_mc",
] + DELTA_COLUMNS


def _load_table(path: str, side: str) -> pd.DataFrame:
    """Loads a per-split table, keeps successful rows and renames its metrics."""
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["train_r2", "test_r2"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("%s is missing columns %s" % (path, missing))
    if "status" in frame.columns:
        frame = frame[frame["status"] == "success"]

    if side == "report":
        renames = {"train_r2": "report_train_r2", "test_r2": "report_test_r2"}
    else:
        renames = {
            "train_r2": "step1_train_r2",
            "test_r2": "step1_test_r2_mc",
            "test_mae": "step1_test_mae_mc",
            "test_r2_det": "step1_test_r2_det",
            "test_mae_det": "step1_test_mae_det",
        }
    frame = frame.rename(columns=renames)
    for optional in ("step1_test_r2_det", "step1_test_mae_det"):
        if optional not in frame.columns:
            frame[optional] = np.nan
    frame["report_id"] = frame["cell"].map(REPORT_ID_BY_CELL)
    return frame[frame["report_id"].notna()].copy()


def _paired(step1: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Pairs step-1 and report rows on (dataset, split, cell) and adds deltas."""
    step_cols = KEY_COLUMNS + [
        "report_id",
        "step1_train_r2",
        "step1_test_r2_det",
        "step1_test_r2_mc",
    ]
    report_cols = KEY_COLUMNS + ["report_train_r2", "report_test_r2"]
    paired = step1[step_cols].merge(
        report[report_cols], on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )
    paired["delta_test_r2_det"] = (
        paired["step1_test_r2_det"] - paired["report_test_r2"]
    )
    paired["delta_test_r2_mc"] = paired["step1_test_r2_mc"] - paired["report_test_r2"]
    return paired


def _aggregate(paired: pd.DataFrame, group_columns) -> pd.DataFrame:
    """Mean and sample SD (ddof=1) over splits, per group."""
    rows = []
    for keys, group in paired.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        entry = dict(zip(group_columns, keys))
        entry["n_splits"] = len(group)
        for column in AGGREGATED_COLUMNS:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            entry[column] = float(np.mean(values)) if len(values) else np.nan
            entry[column + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(entry)

    table = pd.DataFrame(rows)
    if len(table):
        table["_order"] = table["report_id"].map(REPORT_ID_ORDER)
        sort_columns = [c for c in ("split", "_order") if c in table.columns]
        table = table.sort_values(sort_columns).drop(columns="_order")
    return table


def _fmt(value) -> str:
    return "n/a" if value is None or pd.isna(value) else "%.4f" % value


def _fmt_mean_std(mean, std) -> str:
    if mean is None or pd.isna(mean):
        return "n/a"
    if std is None or pd.isna(std):
        return "%.4f" % mean
    return "%.4f ± %.4f" % (mean, std)


def _display(table: pd.DataFrame) -> pd.DataFrame:
    """Formats a numeric table for printing (means ± SD for step-1 metrics)."""
    display = pd.DataFrame()
    if "split" in table.columns:
        display["split"] = table["split"]
    display["report_id"] = table["report_id"]
    display["cell"] = table["cell"]
    display["n_splits"] = table["n_splits"]
    display["report_train_r2"] = table["report_train_r2"].map(_fmt)
    display["report_test_r2"] = table["report_test_r2"].map(_fmt)
    display["step1_train_r2"] = table["step1_train_r2"].map(_fmt)
    for metric in ("det", "mc"):
        display["step1_test_r2_" + metric] = [
            _fmt_mean_std(mean, std)
            for mean, std in zip(
                table["step1_test_r2_" + metric],
                table["step1_test_r2_" + metric + "_std"],
            )
        ]
        display["delta_test_r2_" + metric] = table[
            "delta_test_r2_" + metric
        ].map(_fmt)
    return display


def _print_and_save(name: str, table: pd.DataFrame, output_dir: str) -> None:
    path = os.path.join(output_dir, name + ".csv")
    table.to_csv(path, index=False)
    print("\n=== %s (%s)" % (name.replace("_", " ").upper(), path))
    print(_display(table).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_csv", help="the report's approach1_per_split.csv")
    parser.add_argument(
        "--step1_csv",
        default="../logs/step1/step1_per_split.csv",
        help="merged step-1 per-split CSV (from merge_grid_results.py)",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs/step1/",
        help="folder for the three comparison tables",
    )
    args = parser.parse_args()

    step1 = _load_table(args.step1_csv, "step1")
    report = _load_table(args.report_csv, "report")
    paired = _paired(step1, report)

    n_step1 = len(step1)
    if len(paired) < n_step1:
        print(
            "WARNING: %d/%d step-1 rows have no matching report row"
            % (n_step1 - len(paired), n_step1)
        )
    if paired.empty:
        raise SystemExit("no (dataset, split, cell) pair matches the report")

    os.makedirs(args.output_dir, exist_ok=True)
    bh_cv = paired[(paired["dataset"] == "bh") & (paired["split"].str.startswith("fullcv"))]
    bh_test = paired[(paired["dataset"] == "bh") & (paired["split"].str.startswith("test"))]
    suzuki = paired[paired["dataset"] == "suzuki"]

    tables = [
        ("step1_vs_report_bh_cv", bh_cv, ["report_id", "cell"]),
        ("step1_vs_report_bh_test", bh_test, ["split", "report_id", "cell"]),
        ("step1_vs_report_suzuki", suzuki, ["report_id", "cell"]),
    ]
    for name, subset, group_columns in tables:
        if subset.empty:
            print("WARNING: no paired rows for %s; skipping this table" % name)
            continue
        _print_and_save(
            name, _aggregate(subset, group_columns), args.output_dir
        )


if __name__ == "__main__":
    main()
