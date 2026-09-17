"""
Collects the test results of Suzuki splits trained one at a time
(`main_finetune.py --stage train --split_ids <id>`) into one table.

Usage (from src/):
    python collect_split_results.py                 # splits 0..9
    python collect_split_results.py --split_ids 0 1 2
"""

import argparse
import os
from types import SimpleNamespace

import pandas as pd

from suzuki_splits import TEST_METRICS, experiment_name, result_paths, summarize_results

TABLE_COLUMNS = [
    "split_id",
    "seed",
    "status",
    "best_epoch",
    "n_test",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
    "training_runtime_sec",
]


def load_split_row(log_dir: str, split_id: int) -> dict:
    """Last recorded result row of a split, or a "not run" placeholder."""
    paths = result_paths(SimpleNamespace(log_dir=log_dir, split_ids=[split_id]))
    if not os.path.isfile(paths["results"]):
        return {"split_id": split_id, "status": "not run"}
    return pd.read_csv(paths["results"]).iloc[-1].to_dict()


def collect(log_dir: str, split_ids) -> tuple:
    """Returns (per-split results DataFrame, summary DataFrame over successful runs)."""
    rows = [load_split_row(log_dir, split_id) for split_id in split_ids]
    results = pd.DataFrame(rows)
    ran = results[results["status"] != "not run"]
    summary = summarize_results(ran.to_dict("records")) if len(ran) else None
    return results, summary


def print_results(results: pd.DataFrame, summary) -> None:
    table = results.reindex(columns=TABLE_COLUMNS)
    # keep integer columns integral when "not run" rows add NaNs
    for column in ("seed", "best_epoch", "n_test"):
        table[column] = table[column].astype("Int64")
    print(table.to_string(index=False, float_format=lambda v: "%.4f" % v))
    if summary is not None:
        print()
        print(summary.to_string(index=False, float_format=lambda v: "%.4f" % v))
    missing = results.loc[results["status"] != "success", "split_id"].tolist()
    if missing:
        print("\nSplits without a successful result: %s" % missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--log_dir", type=str, default="../logs/suzuki_regression/")
    args = parser.parse_args()

    results, summary = collect(args.log_dir, args.split_ids)
    print_results(results, summary)

    name = experiment_name(args.split_ids) + "_collected"
    results.to_csv(os.path.join(args.log_dir, name + "_results.csv"), index=False)
    if summary is not None:
        summary.to_csv(os.path.join(args.log_dir, name + "_summary.csv"), index=False)
    print("\nSaved %s_results.csv / _summary.csv in %s (metrics: %s)"
          % (name, args.log_dir, ", ".join(TEST_METRICS)))
