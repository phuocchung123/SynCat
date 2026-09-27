"""
Trains the grid reactant_pooling {mean, rn} x head {linear, mlp} on the
Buchwald-Hartwig and Suzuki-Miyaura splits and collects train/test metrics.

All configurations of a split share its data, seed and every other option of
`main_finetune.py`; attention is always on the reactants. Splits and seeds are
those of the existing pipelines:

- BH: the npz folders of `prepare_bh.py` (`FullCV_<id>` under `--split_columns`,
  and `Test<id>`), seed `--seed` for every dataset, as in `train_bh.py`;
- Suzuki: the prepared `split_<id>` folders, seed from their
  `split_metadata.json` (`--seed` + id), as in the `--stage` pipeline.

After training, the selected checkpoint is also scored on the whole training
set, in eval mode; `finetune` already scores the test set.

Results are appended to `<log_dir>/grid_results.csv` after every run, and a run
that already succeeded is skipped, so the script can be interrupted (Ctrl+C)
and re-run to continue. `<log_dir>/grid_summary.csv` holds the mean and sample
standard deviation (ddof=1) over the splits of each dataset group and
configuration. Every checkpoint, monitor log and plot stays in
`<log_dir>/runs/<pooling>_<head>/<dataset>/`.

`--subset N` trains on the first N reactions of every train/valid/test file
instead (copies written to `<log_dir>/subset_npz/`), for a quick smoke test.

Usage (from src/):
    python pooling_grid.py --subset 512 --epochs 2 --cv_ids 1 --test_ids \\
        --split_ids 0 --num_workers 0 --log_dir ../logs/pooling_grid_smoke/
    python pooling_grid.py --epochs 100 --patience 10
"""

import itertools
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import GraphDataset
from finetune import finetune
from main_finetune import build_parser
from model import HEADS, REACTANT_POOLINGS, model
from multi_gpu import resolve_gpu_ids
from prepare_bh import SPLIT_COLUMNS
from suzuki_splits import METADATA_FILE, split_dir, subset_graph_data
from train_bh import datasets as bh_datasets
from train_bh import run_args
from utils import collate_reaction_graphs, set_seed, setup_logging
from validation import validation

METRICS = ("r2", "mae", "rmse")
SUBSETS = ("train", "valid", "test")

TABLE_COLUMNS = (
    [
        "dataset",
        "group",
        "reactant_pooling",
        "head",
        "status",
        "seed",
        "best_epoch",
        "n_params",
        "n_train",
        "n_valid",
        "n_test",
    ]
    + ["%s_%s" % (subset, m) for subset in ("train", "val", "test") for m in METRICS]
    + ["train_loss_first", "train_loss_last", "train_runtime_sec", "error"]
)


def grid_runs(args) -> list:
    """
    Lists the datasets to train, with their group and seed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; `datasets`, `cv_ids`, `split_columns`, `test_ids` and
        `split_ids` select them.

    Returns
    -------
    list of dict
        One entry per dataset, with its name, group ("bh_cv", "bh_test" or
        "suzuki"), npz folder (relative to `Data_folder`) and seed.
    """
    runs = []
    if "bh" in args.datasets:
        for run in bh_datasets(args):
            runs.append(dict(run, group="bh_" + run["kind"], seed=int(args.seed)))
    if "suzuki" in args.datasets:
        for split_id in args.split_ids:
            folder = split_dir(args, split_id)
            metadata_path = os.path.join(folder, METADATA_FILE)
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(
                    "%s is missing; prepare the split with --stage prepare first"
                    % metadata_path
                )
            with open(metadata_path) as f:
                seed = int(json.load(f)["seed"])
            runs.append(
                {
                    "dataset": "suzuki_split_%d" % split_id,
                    "group": "suzuki",
                    "npz_folder": os.path.join(args.processed_npz_dir, "split_%d" % split_id),
                    "seed": seed,
                }
            )
    return runs


def write_subset(args, run: dict) -> str:
    """
    Writes the first `args.subset` reactions of each npz file of a dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; `subset`, `Data_folder` and `log_dir` are used.
    run : dict
        The dataset, as returned by `grid_runs`.

    Returns
    -------
    str
        Folder holding the truncated train/valid/test npz files.
    """
    out = os.path.join(args.log_dir, "subset_npz", run["dataset"])
    os.makedirs(out, exist_ok=True)
    for subset in SUBSETS:
        target = os.path.join(out, subset + ".npz")
        if os.path.isfile(target):
            continue
        source = os.path.join(args.Data_folder, run["npz_folder"], subset + ".npz")
        with np.load(source, allow_pickle=True) as npz:
            rmol, pmol = list(npz["rmol"]), list(npz["pmol"])
            reaction = npz["reaction"].item()
        positions = np.arange(min(args.subset, len(reaction["y"])))
        rmol, pmol, reaction = subset_graph_data(rmol, pmol, reaction, positions)
        np.savez_compressed(target, rmol=rmol, pmol=pmol, reaction=reaction)
    return out


def score_training_set(one) -> tuple:
    """
    Scores the selected checkpoint of a run on its whole training set.

    Parameters
    ----------
    one : argparse.Namespace
        The run's arguments, as passed to `finetune`.

    Returns
    -------
    tuple
        Regression metrics (see `validation.compute_regression_metrics`) and the
        number of parameters of the model.
    """
    gpus = resolve_gpu_ids(one)
    device = torch.device("cuda:%d" % gpus[0]) if gpus else torch.device("cpu")
    checkpoint = torch.load(
        one.model_path + one.model_name, weights_only=False, map_location=device
    )
    net = model.from_config(checkpoint["model_config"]).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])

    train_set = GraphDataset(one.Data_folder + one.npz_folder + "/train.npz")
    # unlike the training loader, no reaction is dropped
    loader = DataLoader(
        dataset=train_set,
        batch_size=int(min(one.batch_size, len(train_set))),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=one.num_workers,
    )
    metrics, _, _, _, _ = validation(one, net, loader, device)
    return metrics, sum(p.numel() for p in net.parameters())


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and sample standard deviation (ddof=1) of the train/test metrics over
    the splits of each dataset group and configuration.

    Parameters
    ----------
    results : pd.DataFrame
        The recorded result rows.

    Returns
    -------
    pd.DataFrame
        One row per (group, reactant_pooling, head, metric).
    """
    success = results[results["status"] == "success"]
    rows = []
    keys = ["group", "reactant_pooling", "head"]
    for (group, pooling, head), runs in success.groupby(keys, sort=False):
        for column in ["%s_%s" % (s, m) for s in ("train", "test") for m in METRICS]:
            values = pd.to_numeric(runs[column], errors="coerce").dropna()
            rows.append(
                {
                    "group": group,
                    "reactant_pooling": pooling,
                    "head": head,
                    "metric": column,
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "n_runs": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def load_results(path: str) -> pd.DataFrame:
    """Reads the result table written by earlier runs, if it exists."""
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=TABLE_COLUMNS)


def train_one(args, run: dict, pooling: str, head: str) -> dict:
    """
    Trains and scores one configuration on one dataset.

    Parameters
    ----------
    args : argparse.Namespace
        The shared arguments.
    run : dict
        The dataset, as returned by `grid_runs`.
    pooling : str
        The `reactant_pooling` of this run.
    head : str
        The `head` of this run.

    Returns
    -------
    dict
        The result row (see `TABLE_COLUMNS`).
    """
    run_dir = os.path.join(args.log_dir, "runs", "%s_%s" % (pooling, head), run["dataset"])
    one = run_args(args, run, run_dir)
    one.reactant_pooling, one.head = pooling, head
    one.seed = run["seed"]
    if args.subset:
        one.Data_folder = os.path.join(write_subset(args, run), "")
        one.npz_folder = "."
    checkpoint = one.model_path + one.model_name
    if args.rerun_successful and os.path.isfile(checkpoint):
        # finetune resumes from an existing checkpoint; a rerun starts over
        os.remove(checkpoint)

    row = {
        "dataset": run["dataset"],
        "group": run["group"],
        "reactant_pooling": pooling,
        "head": head,
        "status": "success",
        "seed": run["seed"],
        "error": "",
    }
    start = time.time()
    try:
        set_seed(one.seed)
        result = finetune(one, save_embedding=False)
        train_metrics, n_params = score_training_set(one)
        with open(one.monitor_folder + "history.json") as f:
            train_losses = json.load(f)["train"]["loss"]
        row.update(
            {
                "best_epoch": result["best_epoch"],
                "n_params": n_params,
                "n_train": result["n_train"],
                "n_valid": result["n_valid"],
                "n_test": result["n_test"],
                "train_loss_first": train_losses[0],
                "train_loss_last": train_losses[-1],
                "train_runtime_sec": round(result["train_runtime_sec"], 1),
            }
        )
        for subset, metrics in (
            ("train", train_metrics),
            ("val", result["val_metrics"]),
            ("test", result["test_metrics"]),
        ):
            for metric in METRICS:
                row["%s_%s" % (subset, metric)] = metrics.get(metric)
    except Exception as error:  # one failed run must not stop the rest
        row["status"] = "failed"
        row["error"] = str(error).replace("\n", " ")[:300]
        row["train_runtime_sec"] = round(time.time() - start, 1)
        print(traceback.format_exc(), flush=True)
    return row


def print_summary(summary: pd.DataFrame) -> None:
    """Prints "mean +/- std" per dataset group and configuration."""
    if summary.empty:
        return
    table = summary.assign(
        value=[
            "%.4f +/- %s" % (m, "n/a" if pd.isna(s) else "%.4f" % s)
            for m, s in zip(summary["mean"], summary["std"])
        ]
    ).pivot_table(
        index=["group", "reactant_pooling", "head"],
        columns="metric",
        values="value",
        aggfunc="first",
        sort=False,
    )
    print(table.to_string())


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(log_dir="../logs/pooling_grid/", split_ids=list(range(10)))
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["bh", "suzuki"],
        choices=["bh", "suzuki"],
    )
    parser.add_argument("--cv_ids", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--test_ids", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument(
        "--split_columns", type=str, nargs="*", default=["split_70"],
        choices=list(SPLIT_COLUMNS),
    )
    parser.add_argument(
        "--poolings", type=str, nargs="+", default=list(REACTANT_POOLINGS),
        choices=list(REACTANT_POOLINGS),
    )
    parser.add_argument(
        "--heads", type=str, nargs="+", default=list(HEADS), choices=list(HEADS)
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="train on the first N reactions of every file only (smoke test; 0 = all)",
    )
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="retrain runs that already have a successful result",
    )
    args = parser.parse_args()
    # the grid compares poolings of the attended reactants
    args.attention_on = "reactants"

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "grid.log")
    results_path = os.path.join(args.log_dir, "grid_results.csv")
    summary_path = os.path.join(args.log_dir, "grid_summary.csv")

    runs = grid_runs(args)
    for run in runs:
        folder = os.path.join(args.Data_folder, run["npz_folder"])
        missing = [
            f for f in ("train.npz", "valid.npz", "test.npz")
            if not os.path.isfile(os.path.join(folder, f))
        ]
        if missing:
            parser.error("%s is not prepared (missing %s)" % (folder, missing))
    configs = list(itertools.product(args.poolings, args.heads))
    print("=== %d datasets x %d configurations" % (len(runs), len(configs)), flush=True)

    for run in runs:
        for pooling, head in configs:
            name = "%s [%s + %s]" % (run["dataset"], pooling, head)
            results = load_results(results_path)
            done = results[
                (results["dataset"] == run["dataset"])
                & (results["reactant_pooling"] == pooling)
                & (results["head"] == head)
                & (results["status"] == "success")
            ]
            if len(done) and not args.rerun_successful:
                print("=== %s: already successful, skipping" % name, flush=True)
                continue

            print("=== %s: training (seed %d)" % (name, run["seed"]), flush=True)
            row = train_one(args, run, pooling, head)
            # finetune points the root logger at the run's monitor.log
            logger = setup_logging(log_filename=log_path)
            logger.info("--- %s: %s" % (name, row))
            print(
                "=== %s: %s, train R2 %s, test R2 %s, loss %s -> %s"
                % (
                    name,
                    row["status"],
                    row.get("train_r2"),
                    row.get("test_r2"),
                    row.get("train_loss_first"),
                    row.get("train_loss_last"),
                ),
                flush=True,
            )

            results = results[
                ~(
                    (results["dataset"] == run["dataset"])
                    & (results["reactant_pooling"] == pooling)
                    & (results["head"] == head)
                )
            ]
            results = pd.concat([results, pd.DataFrame([row])])
            results = results.reindex(columns=TABLE_COLUMNS)
            results.to_csv(results_path, index=False)
            summarize(results).to_csv(summary_path, index=False)

    results = load_results(results_path)
    print("\n=== RESULTS (%s)" % results_path)
    print(results.to_string(index=False))
    print("\n=== SUMMARY, mean +/- std over splits (%s)" % summary_path)
    print_summary(summarize(results))
