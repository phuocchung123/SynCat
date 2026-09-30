"""
Experiment runner for SynCat approach 2 (pairwise reactant tokens) and the MLP
regression head, over the prepared Buchwald-Hartwig and Suzuki-Miyaura npz data.

Four cells are evaluated on every dataset/split, always with
``attention_on="reactants"``:

1. current model   reactant_tokens=ind  head=linear
2. approach 3      reactant_tokens=comb head=linear
3. approach 2      reactant_tokens=comb head=mlp
4. ind/mlp control reactant_tokens=ind  head=mlp

Only the four cells differ; every other hyperparameter, the seed and the split
are shared. Training goes through the existing ``finetune`` path, and the
best-validation checkpoint it selects is then scored on the complete train and
test subsets with ``validation``. Nothing about the normal baseline training is
changed; this script only drives and records it.

Each dataset/split/cell gets its own checkpoint and output directory, so a cell
can never resume another cell's checkpoint.

Usage (from src/):
    python run_approach2_grid.py --smoke
    python run_approach2_grid.py --epochs 100 --patience 10
    python run_approach2_grid.py --cells approach_2 --epochs 100
    python run_approach2_grid.py --datasets Suzuki-Miyaura --suzuki_split_ids 0 1

Long runs are resumable: pass a fixed `--run_name` and re-run the same command
after a time limit. A dataset/split/cell combination that already has a
`success` row in `grid_results.csv` is skipped; an interrupted one is retrained
from scratch (`--rerun_successful` retrains successful combinations too).

The smoke mode runs one BH split and one Suzuki split for one epoch on small
random subsets, writing temporary npz files next to the results; the prepared
source data is never modified. It is a standalone script and never runs under
pytest.
"""

import copy
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
from model import model
from multi_gpu import resolve_gpu_ids
from utils import collate_reaction_graphs, set_seed, setup_logging
from validation import validation

# The four architectures compared; only these three settings vary.
CELLS = (
    {
        "label": "current_model",
        "description": "current model (ind/linear)",
        "reactant_tokens": "ind",
        "head": "linear",
    },
    {
        "label": "approach_3",
        "description": "approach 3 (comb/linear)",
        "reactant_tokens": "comb",
        "head": "linear",
    },
    {
        "label": "approach_2",
        "description": "approach 2 (comb/mlp)",
        "reactant_tokens": "comb",
        "head": "mlp",
    },
    {
        "label": "ind_mlp",
        "description": "MLP control (ind/mlp)",
        "reactant_tokens": "ind",
        "head": "mlp",
    },
)

ATTENTION_ON = "reactants"
CELL_LABELS = tuple(cell["label"] for cell in CELLS)

# Smoke limits (samples per subset, and the fixed training settings).
SMOKE_TRAIN = 96
SMOKE_VALID = 24
SMOKE_TEST = 24

RESULT_COLUMNS = [
    "dataset",
    "split",
    "seed",
    "label",
    "reactant_tokens",
    "head",
    "attention_on",
    "train_r2",
    "train_mae",
    "train_rmse",
    "test_r2",
    "test_mae",
    "test_rmse",
    "trainable_parameters",
    "epochs_ran",
    "train_runtime_sec",
    "seconds_per_epoch",
    "status",
    "error",
]

SUMMARY_METRICS = (
    "train_r2",
    "train_mae",
    "train_rmse",
    "test_r2",
    "test_mae",
    "test_rmse",
    "seconds_per_epoch",
)


def _subset_mol(mol: dict, positions: np.ndarray) -> dict:
    """Slice a prepared mol dict to `positions` (as `suzuki_splits` does)."""
    n_csum = np.concatenate([[0], np.cumsum(mol["n_node"])])
    e_csum = np.concatenate([[0], np.cumsum(mol["n_edge"])])
    node_rows = np.concatenate(
        [np.arange(n_csum[p], n_csum[p + 1]) for p in positions]
    ).astype(int)
    edge_rows = np.concatenate(
        [np.arange(e_csum[p], e_csum[p + 1]) for p in positions]
    ).astype(int)
    return {
        "n_node": mol["n_node"][positions],
        "n_edge": mol["n_edge"][positions],
        "dummy": [mol["dummy"][p] for p in positions],
        "node_attr": mol["node_attr"][node_rows],
        "edge_attr": mol["edge_attr"][edge_rows],
        "src": mol["src"][edge_rows],
        "dst": mol["dst"][edge_rows],
    }


def write_subset(source_npz: str, target_npz: str, n: int, seed: int) -> None:
    """
    Writes a deterministic `n`-sample subset of a prepared npz to `target_npz`.

    The source file is only read; it is never modified.
    """
    with np.load(source_npz, allow_pickle=True) as npz:
        rmol = list(npz["rmol"])
        pmol = list(npz["pmol"])
        reaction = npz["reaction"].item()
    total = len(reaction["y"])
    n = min(int(n), total)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(total, size=n, replace=False)).astype(int)

    subset = {
        "rmol": [_subset_mol(mol, positions) for mol in rmol],
        "pmol": [_subset_mol(mol, positions) for mol in pmol],
        "reaction": {
            "y": np.asarray(reaction["y"])[positions],
            "rsmi": [reaction["rsmi"][p] for p in positions],
        },
    }
    os.makedirs(os.path.dirname(target_npz), exist_ok=True)
    np.savez_compressed(target_npz, **subset)


def tasks(args) -> list:
    """
    The (dataset, split, npz folder, seed) runs to execute.

    npz folders are relative to `--Data_folder`, exactly as `finetune` expects.
    """
    out = []
    if "Buchwald-Hartwig" in args.datasets:
        for cv_id in args.cv_ids:
            name = "fullcv%02d_split70" % cv_id
            out.append(
                {
                    "dataset": "Buchwald-Hartwig",
                    "split": name,
                    "npz_folder": "npz/bh/" + name,
                    "seed": int(args.seed),
                }
            )
        for test_id in args.test_ids:
            name = "test%d" % test_id
            out.append(
                {
                    "dataset": "Buchwald-Hartwig",
                    "split": name,
                    "npz_folder": "npz/bh/" + name,
                    "seed": int(args.seed),
                }
            )
    if "Suzuki-Miyaura" in args.datasets:
        for split_id in args.suzuki_split_ids:
            name = "split_%d" % split_id
            out.append(
                {
                    "dataset": "Suzuki-Miyaura",
                    "split": name,
                    "npz_folder": "processed/suzuki/npz/" + name,
                    "seed": int(args.seed) + int(split_id),
                }
            )
    return out


def cell_dir(run_root: str, task: dict, cell: dict) -> str:
    """Unique checkpoint/output directory of one dataset/split/cell."""
    return os.path.join(
        run_root,
        "runs",
        "%s_%s_%s" % (task["dataset"], task["split"], cell["label"]),
    )


def load_results(path: str) -> pd.DataFrame:
    """Previously recorded rows, or an empty table with the result columns."""
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def result_keys(results: pd.DataFrame) -> set:
    """The (dataset, split, label) keys present in a results table."""
    if len(results) == 0:
        return set()
    return {
        (row["dataset"], row["split"], row["label"])
        for _, row in results.iterrows()
    }


def successful_keys(results: pd.DataFrame) -> set:
    """The (dataset, split, label) keys that already finished successfully."""
    if len(results) == 0:
        return set()
    return result_keys(results[results["status"] == "success"])


def build_split_args(
    args,
    task: dict,
    cell: dict,
    run_dir: str,
    npz_folder: str,
    data_folder: str,
):
    """Namespace for one cell, pointing at its own inputs and output folder."""
    one = copy.copy(args)
    one.Data_folder = data_folder
    one.npz_folder = npz_folder
    one.monitor_folder = os.path.join(run_dir, "monitor") + os.sep
    one.image_folder = os.path.join(run_dir, "images") + os.sep
    one.model_path = run_dir + os.sep
    one.model_name = "model.pt"
    one.seed = task["seed"]
    one.attention_on = ATTENTION_ON
    one.reactant_tokens = cell["reactant_tokens"]
    one.head = cell["head"]
    one.track_test_each_epoch = False
    os.makedirs(one.monitor_folder, exist_ok=True)
    os.makedirs(one.image_folder, exist_ok=True)
    return one


def read_epochs_ran(monitor_folder: str) -> int:
    """Number of completed epochs recorded in `monitor/history.json`."""
    path = os.path.join(monitor_folder, "history.json")
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        history = json.load(f)
    return int(len(history.get("epoch", [])))


def evaluate_checkpoint(one, device):
    """
    Score the best-validation checkpoint of `one` on its complete train and test
    subsets, returning (train_metrics, test_metrics, trainable_parameters).
    """
    checkpoint = torch.load(
        one.model_path + one.model_name, weights_only=False, map_location=device
    )
    net = model.from_config(checkpoint["model_config"]).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    folder = one.Data_folder + one.npz_folder + "/"
    metrics = {}
    for name in ("train", "test"):
        dataset = GraphDataset(folder + name + ".npz")
        loader = DataLoader(
            dataset,
            batch_size=int(np.min([one.batch_size, len(dataset)])),
            shuffle=False,
            collate_fn=collate_reaction_graphs,
            num_workers=one.num_workers,
            drop_last=False,
        )
        metrics[name] = validation(one, net, loader, device)[0]

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return metrics["train"], metrics["test"], int(trainable)


def run_cell(args, device, task: dict, cell: dict, run_root: str, logger) -> dict:
    """Train and evaluate one dataset/split/cell; failures are recorded, not raised."""
    run_dir = cell_dir(run_root, task, cell)
    os.makedirs(run_dir, exist_ok=True)
    row = {
        "dataset": task["dataset"],
        "split": task["split"],
        "seed": task["seed"],
        "label": cell["label"],
        "reactant_tokens": cell["reactant_tokens"],
        "head": cell["head"],
        "attention_on": ATTENTION_ON,
        "status": "failed",
        "error": "",
    }
    logger.info(
        "--- %s | %s | %s (%s/%s)"
        % (
            task["dataset"],
            task["split"],
            cell["description"],
            cell["reactant_tokens"],
            cell["head"],
        )
    )
    start = time.time()
    try:
        one = build_split_args(
            args, task, cell, run_dir, task["npz_folder"], task["data_folder"]
        )
        # finetune resumes when a checkpoint exists: start every cell fresh.
        checkpoint_path = one.model_path + one.model_name
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

        set_seed(task["seed"])
        result = finetune(one, save_embedding=False)
        train_metrics, test_metrics, trainable = evaluate_checkpoint(one, device)
        epochs_ran = read_epochs_ran(one.monitor_folder)
        runtime = float(result["train_runtime_sec"])
        row.update(
            {
                "train_r2": train_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "trainable_parameters": trainable,
                "epochs_ran": epochs_ran,
                "train_runtime_sec": runtime,
                "seconds_per_epoch": (
                    runtime / epochs_ran if epochs_ran > 0 else float("nan")
                ),
                "status": "success",
            }
        )
        logger.info(
            "--- %s | %s | %s: test R2 %s MAE %.4f RMSE %.4f | %.1fs (%.1fs/epoch)"
            % (
                task["dataset"],
                task["split"],
                cell["label"],
                "n/a" if test_metrics["r2"] is None else "%.4f" % test_metrics["r2"],
                test_metrics["mae"],
                test_metrics["rmse"],
                runtime,
                row["seconds_per_epoch"],
            )
        )
    except Exception as error:  # one failed cell must not stop the rest
        row["error"] = str(error).replace("\n", " ")[:300]
        row["train_runtime_sec"] = round(time.time() - start, 1)
        logger.error(
            "--- %s | %s | %s FAILED: %s"
            % (task["dataset"], task["split"], cell["label"], traceback.format_exc())
        )
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) across successful splits."""
    success = results[results["status"] == "success"] if len(results) else results
    rows = []
    for (dataset, label), group in success.groupby(["dataset", "label"], sort=False):
        row = {
            "dataset": dataset,
            "label": label,
            "reactant_tokens": group["reactant_tokens"].iloc[0],
            "head": group["head"].iloc[0],
            "attention_on": group["attention_on"].iloc[0],
            "trainable_parameters": group["trainable_parameters"].iloc[0],
            "n_runs": int(len(group)),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric + "_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_smoke_subsets(args, run_root: str, logger) -> None:
    """
    Writes small deterministic subset npz files for the smoke run.

    Each dataset/split gets one temp folder shared by all four cells, so the
    cells only differ in architecture. The prepared source data is never touched.
    """
    specs = []
    if "Buchwald-Hartwig" in args.datasets:
        specs.append(("Buchwald-Hartwig", "fullcv01_split70", "npz/bh/fullcv01_split70"))
    if "Suzuki-Miyaura" in args.datasets:
        specs.append(
            ("Suzuki-Miyaura", "split_0", "processed/suzuki/npz/split_0")
        )

    sizes = {"train": SMOKE_TRAIN, "valid": SMOKE_VALID, "test": SMOKE_TEST}
    for dataset, split_name, npz_folder in specs:
        source_dir = os.path.join(args.Data_folder, npz_folder)
        target_dir = os.path.join(run_root, "smoke_npz", "%s_%s" % (dataset, split_name))
        for name, n in sizes.items():
            source = os.path.join(source_dir, name + ".npz")
            target = os.path.join(target_dir, name + ".npz")
            write_subset(source, target, n, seed=args.seed)
            logger.info("--- smoke subset %s -> %s" % (source, target))


if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["Buchwald-Hartwig", "Suzuki-Miyaura"],
        default=["Buchwald-Hartwig", "Suzuki-Miyaura"],
    )
    parser.add_argument("--cv_ids", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--test_ids", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument(
        "--suzuki_split_ids", type=int, nargs="*", default=list(range(10))
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="+",
        choices=list(CELL_LABELS),
        default=list(CELL_LABELS),
        help="restrict the grid to some of the four cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one BH split, one Suzuki split, one epoch, small subsets",
    )
    parser.add_argument("--output_dir", type=str, default="../logs/approach2_grid/")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="retrain dataset/split/cell combinations that already succeeded",
    )
    args = parser.parse_args()

    if args.smoke:
        # one split per dataset, one epoch, and the fixed small-subset settings
        args.datasets = ["Buchwald-Hartwig", "Suzuki-Miyaura"]
        args.cv_ids = [1]
        args.test_ids = []
        args.suzuki_split_ids = [0]
        args.epochs = 1
        args.batch_size = 8
        args.num_workers = 0
        args.layer = 1
        args.emb_dim = 32
        args.patience = 0

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, run_name)
    os.makedirs(run_root, exist_ok=True)
    logger = setup_logging(log_filename=os.path.join(run_root, "grid.log"))

    selected_cells = [cell for cell in CELLS if cell["label"] in args.cells]
    logger.info("=" * 80)
    logger.info(
        "Approach-2 grid (%s): %d dataset splits x %d cells, attention_on=%s"
        % ("smoke" if args.smoke else "full", len(tasks(args)), len(selected_cells), ATTENTION_ON)
    )

    data_folder = os.path.join(args.Data_folder, "")
    if args.smoke:
        prepare_smoke_subsets(args, run_root, logger)

    gpus = resolve_gpu_ids(args)
    device = torch.device("cuda:%d" % gpus[0]) if gpus else torch.device("cpu")
    logger.info("--- device: %s" % device)

    results_path = os.path.join(run_root, "grid_results.csv")
    summary_path = os.path.join(run_root, "grid_summary.csv")
    all_tasks = tasks(args)
    run_keys = {
        (task["dataset"], task["split"], cell["label"])
        for task in all_tasks
        for cell in selected_cells
    }
    existing = load_results(results_path)
    if args.rerun_successful and len(existing):
        keep = ~existing.apply(
            lambda r: (r["dataset"], r["split"], r["label"]) in run_keys, axis=1
        )
        existing = existing[keep]
    completed = successful_keys(existing)
    rows = existing.to_dict("records")

    for task in all_tasks:
        if args.smoke:
            # point at the temp subsets and use an absolute folder
            task = dict(task)
            task["data_folder"] = ""
            task["npz_folder"] = os.path.join(
                run_root, "smoke_npz", "%s_%s" % (task["dataset"], task["split"])
            )
        else:
            task = dict(task)
            task["data_folder"] = data_folder

        for cell in selected_cells:
            key = (task["dataset"], task["split"], cell["label"])
            if key in completed:
                logger.info("--- skip %s (already successful)" % (key,))
                continue
            row = run_cell(args, device, task, cell, run_root, logger)
            rows.append(row)
            table = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
            table.to_csv(results_path, index=False)

    table = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    summarize(table).to_csv(summary_path, index=False)
    logger.info("=" * 80)
    logger.info("Per-split results: %s" % results_path)
    logger.info("Summary: %s" % summary_path)
    print("\n=== RESULTS (%s)" % results_path)
    print(table.to_string(index=False))
    print("\n=== SUMMARY (%s)" % summary_path)
    print(summarize(table).to_string(index=False))
