"""
Merges the per-split CSV files of the step-1 grid jobs into one table.

Each parallel job writes its own ``approach1_per_split.csv``. This script
concatenates them, fails on duplicated ``(dataset, split, cell)`` rows (which
would mean the same run was written twice), writes ``step1_per_split.csv`` and
builds ``step1_summary.csv`` with the grid's `summarize_grid`.

Usage (from src/):
    python merge_grid_results.py ../logs/step1_bh_g1_8/approach1_per_split.csv \
        ../logs/step1_bh_g9_16/approach1_per_split.csv \
        ../logs/step1_suzuki_g1_8/approach1_per_split.csv \
        ../logs/step1_suzuki_g9_16/approach1_per_split.csv \
        --output_dir ../logs/step1/
"""

import argparse
import os

import pandas as pd

from run_approach1_grid import summarize_grid

KEY_COLUMNS = ["dataset", "split", "cell"]


def merge(inputs):
    """
    Concatenates the input CSV files and rejects duplicated runs.

    Parameters
    ----------
    inputs : list of str
        Paths of the per-split CSV files.

    Returns
    -------
    pandas.DataFrame
        All rows, in input order.

    Raises
    ------
    ValueError
        If a ``(dataset, split, cell)`` row appears more than once.
    """
    frames = []
    for path in inputs:
        frame = pd.read_csv(path)
        missing = [c for c in KEY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError("%s is missing columns %s" % (path, missing))
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    duplicated = merged.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicated.any():
        duplicates = merged.loc[duplicated, KEY_COLUMNS].to_dict("records")
        raise ValueError(
            "duplicated dataset/split/cell rows in the inputs: %s" % duplicates
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="approach1_per_split.csv files of the parallel step-1 jobs",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs/step1/",
        help="folder for step1_per_split.csv and step1_summary.csv",
    )
    args = parser.parse_args()

    merged = merge(args.inputs)
    os.makedirs(args.output_dir, exist_ok=True)
    per_split_path = os.path.join(args.output_dir, "step1_per_split.csv")
    summary_path = os.path.join(args.output_dir, "step1_summary.csv")

    merged.to_csv(per_split_path, index=False)
    summary = summarize_grid(merged)
    summary.to_csv(summary_path, index=False)

    print("Merged %d rows from %d files" % (len(merged), len(args.inputs)))
    print("Per-split CSV: %s" % per_split_path)
    print("Summary CSV:   %s" % summary_path)
    if len(summary):
        print("\nSummary:\n%s" % summary.to_string(index=False))


if __name__ == "__main__":
    main()
