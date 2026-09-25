"""
Prepares the npz files of every Buchwald-Hartwig dataset in `Data/raw/BH`.

`FullCV_<id>.csv` carries one train/test column per training fraction
(`split_70` ... `split_2.5`), so each file is prepared once per split column;
`Test<id>.csv` carries a single `split` column. Every combination gets its own
npz folder and is prepared by one `main_finetune.py --prepare_only` process.

Folders that already hold files are skipped, so the script can be interrupted
(Ctrl+C) and re-run to continue. Unknown arguments are forwarded to
main_finetune.py.

Usage (from src/):
    python prepare_bh.py --dry_run          # print the commands only
    python prepare_bh.py                    # prepare everything
    python prepare_bh.py --split_columns split_70 --cv_ids 1 2 3
"""

import argparse
import os
import subprocess
import sys
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

SPLIT_COLUMNS = (
    "split_70",
    "split_50",
    "split_30",
    "split_20",
    "split_10",
    "split_5",
    "split_2.5",
)


def jobs(args):
    """
    Builds the (raw file, split column, npz folder) triples to prepare.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; `cv_ids`, `test_ids` and `split_columns` select the
        subset to prepare.

    Returns
    -------
    list of tuple
        One (data_csv, split_column, npz_folder) triple per npz folder.
    """
    out = []
    for cv_id in args.cv_ids:
        for column in args.split_columns:
            # "split_2.5" -> "2_5", so that the folder name stays path-friendly
            suffix = column.replace("split_", "").replace(".", "_")
            out.append(
                (
                    "raw/BH/FullCV_%02d.csv" % cv_id,
                    column,
                    "npz/bh/fullcv%02d_split%s" % (cv_id, suffix),
                )
            )
    for test_id in args.test_ids:
        out.append(
            ("raw/BH/Test%d.csv" % test_id, "split", "npz/bh/test%d" % test_id)
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # nargs="*" so that "--test_ids" with no value prepares the CV files only
    parser.add_argument("--cv_ids", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--test_ids", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument(
        "--split_columns", type=str, nargs="*", default=list(SPLIT_COLUMNS)
    )
    parser.add_argument("--Data_folder", type=str, default="../Data/")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the commands without preparing anything",
    )
    args, forwarded = parser.parse_known_args()
    os.chdir(SRC_DIR)  # main_finetune.py uses paths relative to src/

    planned = jobs(args)
    missing = [
        csv
        for csv, _, _ in planned
        if not os.path.isfile(os.path.join(args.Data_folder, csv))
    ]
    if missing:
        parser.error("raw file(s) not found: %s" % sorted(set(missing)))
    print("=== %d npz folders to prepare" % len(planned), flush=True)
    failures = []
    for data_csv, column, npz_folder in planned:
        target = os.path.join(args.Data_folder, npz_folder)
        if os.path.isdir(target) and os.listdir(target):
            print("=== %s: already prepared, skipping" % npz_folder, flush=True)
            continue
        command = [
            sys.executable,
            "main_finetune.py",
            "--prepare_only",
            "--data_csv", data_csv,
            "--npz_folder", npz_folder,
            "--train_test_split",
            "--split_column", column,
            "--reaction_column", "rxn",
            "--y_column", "Output",
            "--Data_folder", args.Data_folder,
        ] + forwarded
        print("=== %s (%s): %s" % (npz_folder, column, " ".join(command)), flush=True)
        if args.dry_run:
            continue
        start = time.time()
        code = subprocess.call(command)
        print(
            "=== %s: exit code %d after %.1f min"
            % (npz_folder, code, (time.time() - start) / 60),
            flush=True,
        )
        if code != 0:
            failures.append(npz_folder)

    if failures:
        print("\n=== FAILED: %s" % failures, flush=True)
        sys.exit(1)
