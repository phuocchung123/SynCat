"""
Experiment runner for SynCat approach 1: relation-network pooling of the
attended reactant tokens, evaluated together with the pairwise reactant tokens
and the switchable head.

The grid is always run with ``attention_on="reactants"`` and crosses the three
switches, giving eight cells:

    reactant_tokens {ind, comb} x reactant_pooling {mean, rn} x head {linear, mlp}

    ind/mean/linear  = current model
    comb/mean/linear = approach 3
    comb/mean/mlp    = approach 2
    comb/rn/linear, comb/rn/mlp = approach 1
    ind/rn/linear,  ind/rn/mlp  = option B
    ind/mean/mlp     = MLP control (the eighth cell, not named in the brief)

Only the three settings differ; every other hyperparameter, the seed and the
split are shared. Training goes through the existing ``finetune`` path; the
best-validation checkpoint it selects is then scored on the complete train and
test subsets with ``validation``. Nothing about the baseline training changes.

LAPTOP PROFILE (the default and the only profile this script runs)
------------------------------------------------------------------
One epoch, 64 train / 32 valid / 32 test rows, ``--emb_dim 32``,
``--num_workers 0``; BH uses batch 8, Suzuki batch 4. The numbers it produces are
a CORRECTNESS CHECK ONLY and are explicitly labelled as not results (see the
``profile`` and ``note`` columns and ``PROFILE_NOT_RESULTS.txt``).

Each cell runs in its own subprocess (as ``run_splits_sequential.py`` does), so
one broken cell cannot take the grid down. ``TQDM_DISABLE=1`` is set in every
child, because tqdm's Windows file writer has previously killed cells with
``OSError: [Errno 22]``. Each cell gets its own run directory and any
pre-existing ``model.pt`` is deleted first, because ``finetune`` resumes from any
checkpoint it finds. Peak memory is the child's peak RSS via psutil (None when
psutil is unavailable); there is no GPU here, so CUDA memory is meaningless.

The server commands for the real settings (full data, ``--emb_dim 384``,
``--batch_size 128``, ``--epochs 100 --patience 10``, all 10 BH CV splits through
``train_bh.py`` and all 10 Suzuki splits through ``run_splits_sequential.py``)
are written to ``<log_dir>/server_commands.txt`` together with a one-line cost
note per dataset. This script does NOT run them.

Usage (from src/):
    python run_approach1_grid.py
    python run_approach1_grid.py --log_dir ../logs/approach1_grid/
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import GraphDataset  # noqa: E402
from finetune import finetune  # noqa: E402
from model import (  # noqa: E402
    ATTENTION_TARGETS,
    HEADS,
    REACTION_COMBINE_DIMS,
    REACTANT_POOLINGS,
    REACTANT_TOKENS,
    model,
)
from multi_gpu import resolve_gpu_ids  # noqa: E402
from utils import collate_reaction_graphs, set_seed, setup_logging  # noqa: E402
from validation import validation  # noqa: E402

try:  # optional, not in requirements.txt
    import psutil
except ImportError:  # pragma: no cover - depends on the environment
    psutil = None


# ---------------------------------------------------------------------------
# shared subsetting helpers (reused from run_approach2_grid when importable)
# ---------------------------------------------------------------------------


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
    """Write a deterministic `n`-sample subset of a prepared npz; never edits it."""
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


try:
    # run_approach2_grid imports main_finetune, which imports rdkit.Chem; that is
    # blocked on the test machine, so fall back to the local copies above.
    from run_approach2_grid import (  # noqa: E402,F811
        _subset_mol,
        write_subset,
    )
except ImportError:  # pragma: no cover - environment dependent
    pass


# ---------------------------------------------------------------------------
# grid definition
# ---------------------------------------------------------------------------

ATTENTION_ON = "reactants"
PROFILE = "laptop_smoke_correctness_only"
NOTE = "not a result"

CELLS = (
    {
        "label": "current_model",
        "description": "current model (ind/mean/linear)",
        "reactant_tokens": "ind",
        "reactant_pooling": "mean",
        "head": "linear",
    },
    {
        "label": "approach_3",
        "description": "approach 3 (comb/mean/linear)",
        "reactant_tokens": "comb",
        "reactant_pooling": "mean",
        "head": "linear",
    },
    {
        "label": "approach_2",
        "description": "approach 2 (comb/mean/mlp)",
        "reactant_tokens": "comb",
        "reactant_pooling": "mean",
        "head": "mlp",
    },
    {
        "label": "approach_1_linear",
        "description": "approach 1 (comb/rn/linear)",
        "reactant_tokens": "comb",
        "reactant_pooling": "rn",
        "head": "linear",
    },
    {
        "label": "approach_1_mlp",
        "description": "approach 1 (comb/rn/mlp)",
        "reactant_tokens": "comb",
        "reactant_pooling": "rn",
        "head": "mlp",
    },
    {
        "label": "option_b_linear",
        "description": "option B (ind/rn/linear)",
        "reactant_tokens": "ind",
        "reactant_pooling": "rn",
        "head": "linear",
    },
    {
        "label": "option_b_mlp",
        "description": "option B (ind/rn/mlp)",
        "reactant_tokens": "ind",
        "reactant_pooling": "rn",
        "head": "mlp",
    },
    {
        "label": "ind_mean_mlp",
        "description": "MLP control (ind/mean/mlp)",
        "reactant_tokens": "ind",
        "reactant_pooling": "mean",
        "head": "mlp",
    },
)

TASKS = (
    {
        "dataset": "Buchwald-Hartwig",
        "split": "fullcv01_split70",
        "npz_folder": "npz/bh/fullcv01_split70",
        "batch_size": 8,
    },
    {
        "dataset": "Suzuki-Miyaura",
        "split": "split_0",
        "npz_folder": "processed/suzuki/npz/split_0",
        "batch_size": 4,
    },
)

# Laptop profile: exact caps, not scaled up.
SMOKE_TRAIN = 64
SMOKE_VALID = 32
SMOKE_TEST = 32
SMOKE_EMB_DIM = 32
SMOKE_NUM_WORKERS = 0
SMOKE_EPOCHS = 1

RESULT_COLUMNS = [
    "dataset",
    "split",
    "seed",
    "label",
    "description",
    "reactant_tokens",
    "reactant_pooling",
    "head",
    "attention_on",
    "profile",
    "note",
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
    "peak_rss_mb",
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
    "peak_rss_mb",
)


def build_parser() -> argparse.ArgumentParser:
    """
    Parser for the grid, with the training defaults of ``main_finetune.py``.

    It is declared locally instead of imported from ``main_finetune`` because
    importing that module pulls in ``rdkit.Chem`` (via ``prepare_data``), which
    the Windows Application Control policy blocks on the test machine.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpus", type=str, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--attention_layer", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument(
        "--attention_on",
        type=str,
        default="reactants",
        choices=list(ATTENTION_TARGETS),
    )
    parser.add_argument(
        "--reaction_combine",
        type=str,
        default="concat",
        choices=sorted(REACTION_COMBINE_DIMS),
    )
    parser.add_argument(
        "--reactant_tokens",
        type=str,
        default="ind",
        choices=list(REACTANT_TOKENS),
    )
    parser.add_argument(
        "--reactant_pooling",
        type=str,
        default="mean",
        choices=list(REACTANT_POOLINGS),
    )
    parser.add_argument("--head", type=str, default="linear", choices=list(HEADS))
    parser.add_argument("--emb_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--Data_folder", type=str, default="../Data/")
    parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    parser.add_argument("--image_folder", type=str, default="../Image/")
    parser.add_argument("--model_path", type=str, default="../Data/model/")
    parser.add_argument("--model_name", type=str, default="model_yield.pt")
    parser.add_argument("--npz_folder", type=str, default="npz/npz_yield")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--track_test_each_epoch", action="store_true")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../logs/approach1_grid/",
        help="NEW output directory; logs/smoke_grid/ is never touched",
    )
    parser.add_argument(
        "--run-cell",
        dest="run_cell",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


# ---------------------------------------------------------------------------
# one cell (child process)
# ---------------------------------------------------------------------------


def _read_epochs_ran(monitor_folder: str) -> int:
    """Number of completed epochs recorded in `monitor/history.json`."""
    path = os.path.join(monitor_folder, "history.json")
    if not os.path.isfile(path):
        return 0
    with open(path) as handle:
        history = json.load(handle)
    return int(len(history.get("epoch", [])))


def _evaluate_checkpoint(one, device):
    """Score the best checkpoint on the complete train and test subsets."""
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


def _run_one_cell(spec_path: str) -> None:
    """Train and score a single cell, writing the metrics to a JSON file."""
    with open(spec_path) as handle:
        spec = json.load(handle)

    args = build_parser().parse_args([])
    for key, value in spec["args"].items():
        setattr(args, key, value)
    args.Data_folder = spec["data_folder"]
    args.npz_folder = spec["npz_folder"]
    args.monitor_folder = spec["monitor_folder"]
    args.image_folder = spec["image_folder"]
    args.model_path = spec["model_path"]
    args.model_name = "model.pt"
    args.attention_on = spec["attention_on"]
    args.reactant_tokens = spec["reactant_tokens"]
    args.reactant_pooling = spec["reactant_pooling"]
    args.head = spec["head"]
    args.seed = spec["seed"]
    args.track_test_each_epoch = False
    os.makedirs(args.monitor_folder, exist_ok=True)
    os.makedirs(args.image_folder, exist_ok=True)

    checkpoint_path = args.model_path + args.model_name
    if os.path.isfile(checkpoint_path):  # finetune resumes from any checkpoint
        os.remove(checkpoint_path)

    set_seed(args.seed)
    gpus = resolve_gpu_ids(args)
    device = torch.device("cuda:%d" % gpus[0]) if gpus else torch.device("cpu")
    result = finetune(args, save_embedding=False)
    train_metrics, test_metrics, trainable = _evaluate_checkpoint(args, device)
    epochs_ran = _read_epochs_ran(args.monitor_folder)
    runtime = float(result["train_runtime_sec"])

    out = {
        "train_r2": train_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "trainable_parameters": trainable,
        "epochs_ran": epochs_ran,
        "train_runtime_sec": runtime,
        "seconds_per_epoch": runtime / epochs_ran if epochs_ran > 0 else None,
    }
    with open(spec["result_path"], "w") as handle:
        json.dump(out, handle)


# ---------------------------------------------------------------------------
# grid driver (parent process)
# ---------------------------------------------------------------------------


def _load_y(npz_path: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as npz:
        return np.asarray(npz["reaction"].item()["y"], dtype=float)


def _prepare_subsets(args, log_dir: str, logger) -> dict:
    """Write the small deterministic subset npz files; return their folder per task."""
    folders = {}
    sizes = {"train": SMOKE_TRAIN, "valid": SMOKE_VALID, "test": SMOKE_TEST}
    for task in TASKS:
        source_dir = os.path.join(args.Data_folder, task["npz_folder"])
        target_dir = os.path.join(
            log_dir, "smoke_npz", "%s_%s" % (task["dataset"], task["split"])
        )
        for name, n in sizes.items():
            target = os.path.join(target_dir, name + ".npz")
            write_subset(
                os.path.join(source_dir, name + ".npz"), target, n, seed=args.seed
            )
            if np.std(_load_y(target)) == 0.0:
                raise RuntimeError(
                    "subset %s has zero target variance; R2/Pearson would be None"
                    % target
                )
        folders[task["dataset"]] = target_dir
        logger.info("--- smoke subsets for %s -> %s" % (task["dataset"], target_dir))
    return folders


def _run_child(cmd: list, env: dict, log_path: str):
    """Run one cell in a subprocess; return (exit_code, peak_rss_mb)."""
    peak = 0
    with open(log_path, "ab") as log_handle:
        process = subprocess.Popen(
            cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT
        )
        if psutil is not None:
            try:
                proc = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        peak = max(peak, proc.memory_info().rss)
                    except Exception:
                        pass
                    time.sleep(0.2)
            except Exception:
                peak = 0
        code = process.wait()
    peak_mb = None if psutil is None else round(peak / (1024.0 * 1024.0), 1)
    return code, peak_mb


def _cell_spec(args, task: dict, cell: dict, run_dir: str, npz_folder: str) -> dict:
    """The JSON description a child process needs to train and score one cell."""
    return {
        "args": {
            key: getattr(args, key)
            for key in (
                "batch_size",
                "epochs",
                "device",
                "gpus",
                "num_workers",
                "layer",
                "attention_layer",
                "num_heads",
                "reaction_combine",
                "emb_dim",
                "dropout",
                "lr",
                "weight_decay",
                "patience",
            )
        },
        "data_folder": "",
        "npz_folder": npz_folder,
        "monitor_folder": os.path.join(run_dir, "monitor") + os.sep,
        "image_folder": os.path.join(run_dir, "images") + os.sep,
        "model_path": run_dir + os.sep,
        "attention_on": ATTENTION_ON,
        "reactant_tokens": cell["reactant_tokens"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
        "seed": task["seed"],
        "result_path": os.path.join(run_dir, "cell_result.json"),
    }


def _run_cell(args, task: dict, cell: dict, npz_folder: str, log_dir: str, logger) -> dict:
    """Train and score one cell via a subprocess; failures are recorded, not raised."""
    run_dir = os.path.join(
        log_dir,
        "runs",
        "%s_%s_%s" % (task["dataset"], task["split"], cell["label"]),
    )
    os.makedirs(run_dir, exist_ok=True)
    row = {
        "dataset": task["dataset"],
        "split": task["split"],
        "seed": task["seed"],
        "label": cell["label"],
        "description": cell["description"],
        "reactant_tokens": cell["reactant_tokens"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
        "attention_on": ATTENTION_ON,
        "profile": PROFILE,
        "note": NOTE,
        "status": "failed",
        "error": "",
    }
    logger.info(
        "--- %s | %s | %s (%s/%s/%s)"
        % (
            task["dataset"],
            task["split"],
            cell["label"],
            cell["reactant_tokens"],
            cell["reactant_pooling"],
            cell["head"],
        )
    )
    start = time.time()
    spec = _cell_spec(args, task, cell, run_dir, npz_folder)
    spec_path = os.path.join(run_dir, "cell_spec.json")
    with open(spec_path, "w") as handle:
        json.dump(spec, handle)

    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"  # tqdm's Windows file writer has killed cells before
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    cmd = [sys.executable, os.path.abspath(__file__), "--run-cell", spec_path]
    code, peak_mb = _run_child(cmd, env, os.path.join(run_dir, "cell.log"))

    result_path = spec["result_path"]
    if code == 0 and os.path.isfile(result_path):
        with open(result_path) as handle:
            metrics = json.load(handle)
        row.update(metrics)
        row["peak_rss_mb"] = peak_mb
        row["status"] = "success"
        logger.info(
            "--- %s | %s | %s: test R2 %s MAE %.4f RMSE %.4f | %.1fs (%.1fs/epoch)"
            % (
                task["dataset"],
                task["split"],
                cell["label"],
                "n/a" if metrics["test_r2"] is None else "%.4f" % metrics["test_r2"],
                metrics["test_mae"],
                metrics["test_rmse"],
                metrics["train_runtime_sec"],
                metrics["seconds_per_epoch"] or float("nan"),
            )
        )
    else:
        row["error"] = "child exit code %d" % code
        row["train_runtime_sec"] = round(time.time() - start, 1)
        logger.error(
            "--- %s | %s | %s FAILED (exit %d); see %s"
            % (task["dataset"], task["split"], cell["label"], code,
               os.path.join(run_dir, "cell.log"))
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
            "description": group["description"].iloc[0],
            "reactant_tokens": group["reactant_tokens"].iloc[0],
            "reactant_pooling": group["reactant_pooling"].iloc[0],
            "head": group["head"].iloc[0],
            "attention_on": group["attention_on"].iloc[0],
            "profile": PROFILE,
            "note": NOTE,
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


def write_server_commands(log_dir: str) -> None:
    """Write the exact real-setting commands and per-dataset cost notes."""
    cv_ids = " ".join(str(i) for i in range(1, 11))
    suzuki_ids = " ".join(str(i) for i in range(10))
    lines = [
        "# Approach-1 server commands. NOT RUN by run_approach1_grid.py.",
        "# Real settings: --emb_dim 384 --batch_size 128 --epochs 100 --patience 10,",
        "# full data, attention_on=reactants, seed 42 (train_bh/run_splits defaults).",
        "#",
        "# COST (STEP 1(c), measured CPU, B=128, D=384, forward+backward):",
        "#   Buchwald-Hartwig: 21 reactant tokens / 210 pairs, ~1.2 s/batch;",
        "#                     baseline epoch ~82.8 s (logs/smoke_grid/grid_results.csv).",
        "#   Suzuki-Miyaura:   105 reactant tokens / 5460 pairs, ~70 s/batch,",
        "#                     2.15 GB pair-descriptor tensor.",
        "",
    ]
    for cell in CELLS:
        flags = (
            "--attention_on reactants --reactant_tokens %s --reactant_pooling %s "
            "--head %s --emb_dim 384 --batch_size 128 --epochs 100 --patience 10"
            % (
                cell["reactant_tokens"],
                cell["reactant_pooling"],
                cell["head"],
            )
        )
        lines.append("# %s: %s" % (cell["label"], cell["description"]))
        lines.append(
            "python train_bh.py --cv_ids %s --test_ids 1 2 3 4 %s "
            "--log_dir ../logs/approach1_server/%s/bh/"
            % (cv_ids, flags, cell["label"])
        )
        lines.append(
            "python run_splits_sequential.py --split_ids %s %s "
            "--log_dir ../logs/approach1_server/%s/suzuki/"
            % (suzuki_ids, flags, cell["label"])
        )
        lines.append("")
    with open(os.path.join(log_dir, "server_commands.txt"), "w") as handle:
        handle.write("\n".join(lines))


def run_grid(args) -> None:
    """Apply the laptop profile and run the eight cells on both datasets."""
    args.epochs = SMOKE_EPOCHS
    args.emb_dim = SMOKE_EMB_DIM
    args.num_workers = SMOKE_NUM_WORKERS
    args.patience = 0
    args.attention_on = ATTENTION_ON

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_filename=os.path.join(log_dir, "grid.log"))
    write_server_commands(log_dir)
    with open(os.path.join(log_dir, "PROFILE_NOT_RESULTS.txt"), "w") as handle:
        handle.write(
            "Profile: %s\nThese metrics are a correctness check on tiny subsets,\n"
            "NOT results. Do not report R2/MAE/RMSE from this directory.\n" % PROFILE
        )

    logger.info("=" * 80)
    logger.info(
        "Approach-1 grid (%s): %d datasets x %d cells, attention_on=%s"
        % (PROFILE, len(TASKS), len(CELLS), ATTENTION_ON)
    )
    smoke_folders = _prepare_subsets(args, log_dir, logger)

    results_path = os.path.join(log_dir, "grid_results.csv")
    summary_path = os.path.join(log_dir, "grid_summary.csv")
    rows = []
    for task in TASKS:
        npz_folder = smoke_folders[task["dataset"]]
        for cell in CELLS:
            task_args = argparse.Namespace(**vars(args))
            task_args.batch_size = task["batch_size"]
            task = dict(task)
            task["seed"] = args.seed
            rows.append(_run_cell(task_args, task, cell, npz_folder, log_dir, logger))
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


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.run_cell:
        _run_one_cell(parsed.run_cell)
    else:
        run_grid(parsed)
