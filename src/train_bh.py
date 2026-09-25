"""
Trains one model per prepared Buchwald-Hartwig npz folder and collects the test
results into one table.

Each run is one dataset: a `FullCV_<id>` file under one split column, or a
`Test<id>` file. The npz folders are the ones written by `prepare_bh.py`
(`Data/npz/bh/fullcv<id>_split<column>` and `Data/npz/bh/test<id>`), so this
script never touches the raw data.

Results are appended to `<log_dir>/bh_results.csv` after every run, and a run
that already succeeded is skipped, so the script can be interrupted (Ctrl+C) and
re-run to continue. Every model, monitor log and plot of a run stays in its own
folder under `<log_dir>/runs/<dataset>/`.

All the model and training options of `main_finetune.py` are accepted
(`--architecture`, `--attention_on`, `--epochs`, `--patience`, `--gpus`, ...).

Usage (from src/):
    python train_bh.py --epochs 100 --patience 10
    python train_bh.py --cv_ids 1 2 3 --test_ids --epochs 100
    python train_bh.py --architecture cross_center --log_dir ../logs/bh_crosscenter/
"""

import copy
import os
import time
import traceback

import numpy as np
import pandas as pd

from finetune import finetune
from main_finetune import build_parser
from prepare_bh import SPLIT_COLUMNS
from utils import set_seed, setup_logging

TEST_METRICS = ("mae", "rmse", "r2", "pearson")

TABLE_COLUMNS = [
    "dataset",
    "kind",
    "split_column",
    "status",
    "seed",
    "best_epoch",
    "n_train",
    "n_valid",
    "n_test",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
    "train_runtime_sec",
    "error",
]


def datasets(args) -> list:
    """
    Lists the runs to train, in the order they are executed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; `cv_ids`, `split_columns` and `test_ids` select them.

    Returns
    -------
    list of dict
        One entry per run, with its name, kind, split column and npz folder.
    """
    runs = []
    for cv_id in args.cv_ids:
        for column in args.split_columns:
            suffix = column.replace("split_", "").replace(".", "_")
            name = "fullcv%02d_split%s" % (cv_id, suffix)
            runs.append(
                {
                    "dataset": name,
                    "kind": "cv",
                    "split_column": column,
                    "npz_folder": "npz/bh/" + name,
                }
            )
    for test_id in args.test_ids:
        name = "test%d" % test_id
        runs.append(
            {
                "dataset": name,
                "kind": "test",
                "split_column": "split",
                "npz_folder": "npz/bh/" + name,
            }
        )
    return runs


def load_results(path: str) -> pd.DataFrame:
    """
    Reads the result table written by earlier runs, if it exists.

    Parameters
    ----------
    path : str
        Path of the results CSV.

    Returns
    -------
    pd.DataFrame
        The recorded rows, or an empty table with the expected columns.
    """
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=TABLE_COLUMNS)


def run_args(args, run: dict, run_dir: str):
    """
    Builds the argument namespace of one run.

    Parameters
    ----------
    args : argparse.Namespace
        The shared arguments.
    run : dict
        The run description, as returned by `datasets`.
    run_dir : str
        Folder that receives this run's checkpoint, monitor log and images.

    Returns
    -------
    argparse.Namespace
        A copy of `args` pointing at this run's npz folder and output folders.
    """
    one = copy.copy(args)
    # finetune builds npz paths by concatenation and expects a trailing separator
    one.Data_folder = os.path.join(args.Data_folder, "")
    one.npz_folder = run["npz_folder"]
    one.monitor_folder = os.path.join(run_dir, "monitor") + os.sep
    one.image_folder = os.path.join(run_dir, "images") + os.sep
    one.model_path = run_dir + os.sep
    one.model_name = "model.pt"
    os.makedirs(one.monitor_folder, exist_ok=True)
    os.makedirs(one.image_folder, exist_ok=True)
    return one


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and sample standard deviation (ddof=1) of the test metrics, per kind.

    Parameters
    ----------
    results : pd.DataFrame
        The recorded result rows.

    Returns
    -------
    pd.DataFrame
        One row per (kind, split column, metric).
    """
    success = results[results["status"] == "success"]
    rows = []
    for (kind, column), group in success.groupby(["kind", "split_column"]):
        for metric in TEST_METRICS:
            values = pd.to_numeric(group["test_" + metric], errors="coerce").dropna()
            rows.append(
                {
                    "kind": kind,
                    "split_column": column,
                    "metric": "test_" + metric,
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "n_runs": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument("--cv_ids", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--test_ids", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument(
        "--split_columns", type=str, nargs="*", default=["split_70"],
        choices=list(SPLIT_COLUMNS),
    )
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="retrain datasets that already have a successful result",
    )
    args = parser.parse_args()
    # the BH tables hold percent yields in "Output"; the npz files already do too
    args.reaction_column = "rxn"
    args.y_column = "Output"

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logging(log_filename=os.path.join(args.log_dir, "bh_training.log"))
    results_path = os.path.join(args.log_dir, "bh_results.csv")
    summary_path = os.path.join(args.log_dir, "bh_summary.csv")

    runs = datasets(args)
    logger.info("--- %d BH datasets to train" % len(runs))
    for run in runs:
        folder = os.path.join(args.Data_folder, run["npz_folder"])
        missing = [
            f
            for f in ("train.npz", "valid.npz", "test.npz")
            if not os.path.isfile(os.path.join(folder, f))
        ]
        if missing:
            parser.error(
                "%s is not prepared (missing %s); run prepare_bh.py first"
                % (run["npz_folder"], missing)
            )

    for run in runs:
        results = load_results(results_path)
        done = results[
            (results["dataset"] == run["dataset"]) & (results["status"] == "success")
        ]
        if len(done) and not args.rerun_successful:
            logger.info("--- %s: already successful, skipping" % run["dataset"])
            continue

        run_dir = os.path.join(args.log_dir, "runs", run["dataset"])
        one = run_args(args, run, run_dir)
        checkpoint = os.path.join(run_dir, "model.pt")
        if args.rerun_successful and os.path.isfile(checkpoint):
            # finetune resumes from an existing checkpoint; a rerun starts over
            os.remove(checkpoint)

        set_seed(args.seed)
        logger.info("--- training %s (%s)" % (run["dataset"], run["npz_folder"]))
        row = {
            "dataset": run["dataset"],
            "kind": run["kind"],
            "split_column": run["split_column"],
            "seed": args.seed,
            "status": "success",
            "error": "",
        }
        start = time.time()
        try:
            result = finetune(one, save_embedding=False)
            row.update(
                {
                    "best_epoch": result["best_epoch"],
                    "n_train": result["n_train"],
                    "n_valid": result["n_valid"],
                    "n_test": result["n_test"],
                    "train_runtime_sec": round(result["train_runtime_sec"], 1),
                }
            )
            for metric in TEST_METRICS:
                row["test_" + metric] = result["test_metrics"].get(metric)
            logger.info(
                "--- %s: MAE %s, RMSE %s, R2 %s"
                % (
                    run["dataset"],
                    row["test_mae"],
                    row["test_rmse"],
                    row["test_r2"],
                )
            )
        except Exception as error:  # one failed dataset must not stop the rest
            row["status"] = "failed"
            row["error"] = str(error).replace("\n", " ")[:300]
            row["train_runtime_sec"] = round(time.time() - start, 1)
            logger.error(
                "--- %s FAILED: %s" % (run["dataset"], traceback.format_exc())
            )

        results = pd.concat([load_results(results_path), pd.DataFrame([row])])
        results = results.reindex(columns=TABLE_COLUMNS)
        results.to_csv(results_path, index=False)
        summarize(results).to_csv(summary_path, index=False)

    results = load_results(results_path)
    print("\n=== RESULTS (%s)" % results_path)
    print(results.to_string(index=False))
    print("\n=== SUMMARY (%s)" % summary_path)
    print(summarize(results).to_string(index=False))
