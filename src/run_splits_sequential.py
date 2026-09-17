"""
Trains the Suzuki splits one after another, each in its own process
(`main_finetune.py --stage train --split_ids <id>`), then prints the test
results of every split.

Splits that already have a successful result are skipped, so the script can be
stopped (Ctrl+C) and re-run to continue. Unknown arguments are forwarded to
main_finetune.py.

Usage (from src/):
    python run_splits_sequential.py --epochs 100 --patience 10 --num_workers 0
    python run_splits_sequential.py --split_ids 3 4 5 --epochs 100
"""

import argparse
import os
import subprocess
import sys
import time

from collect_split_results import collect, load_split_row, print_results

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--log_dir", type=str, default="../logs/suzuki_regression/")
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="retrain splits that already have a successful result",
    )
    args, forwarded = parser.parse_known_args()
    os.chdir(SRC_DIR)  # main_finetune.py uses paths relative to src/

    for split_id in args.split_ids:
        previous = load_split_row(args.log_dir, split_id)["status"]
        if previous == "success" and not args.rerun_successful:
            print("=== split %d: already successful, skipping" % split_id, flush=True)
            continue
        command = [
            sys.executable,
            "main_finetune.py",
            "--stage", "train",
            "--split_ids", str(split_id),
            "--log_dir", args.log_dir,
            "--skip_aggregate",
        ] + forwarded
        if previous != "not run":
            command.append("--overwrite_results")
        print("=== split %d: %s" % (split_id, " ".join(command)), flush=True)
        start = time.time()
        code = subprocess.call(command)
        print(
            "=== split %d: exit code %d after %.1f min"
            % (split_id, code, (time.time() - start) / 60),
            flush=True,
        )

    print("\n=== TEST RESULTS PER SPLIT")
    print_results(*collect(args.log_dir, args.split_ids))
