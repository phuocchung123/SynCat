"""
Factorial ablation runner for SynCat: crosses all 16 cells of:
    reactant_tokens {ind, comb}
    x attention_on {reactants, none}
    x reactant_pooling {mean, rn}
    x head {linear, mlp}
over Buchwald-Hartwig (FullCV 01-10 at split_70, plus Test1-4) and
Suzuki-Miyaura (split_0-9).

The 16 cells:
    0  ind/reactants/mean/linear   baseline
    1  ind/reactants/mean/mlp      baseline_mlp
    2  ind/reactants/rn/linear     B_linear
    3  ind/reactants/rn/mlp        B_mlp
    4  ind/none/mean/linear        control_linear
    5  ind/none/mean/mlp           control_mlp
    6  ind/none/rn/linear          A_linear
    7  ind/none/rn/mlp             A_mlp
    8  comb/reactants/mean/linear  approach3
    9  comb/reactants/mean/mlp     approach2
    10 comb/reactants/rn/linear    approach1_linear
    11 comb/reactants/rn/mlp       approach1_mlp
    12 comb/none/mean/linear       comb_none_mean_linear
    13 comb/none/mean/mlp          comb_none_mean_mlp
    14 comb/none/rn/linear         comb_none_rn_linear
    15 comb/none/rn/mlp            comb_none_rn_mlp

Only the four switches differ across cells; all hyperparameters, random seeds,
and data splits are shared. Training executes in an isolated child subprocess per
split. Any existing model.pt and monitor/history.json are removed before training
to ensure a clean run.

Metrics and Runtime
-------------------
seconds_per_epoch = train_runtime_sec / epochs_ran (wall time including
per-epoch validation and plotting, providing the exact same overhead for every cell).
Peak GPU memory is tracked via torch.cuda.max_memory_allocated(device) (None on CPU).
Peak CPU memory is tracked via psutil RSS (when psutil is available).

Output and Resumption
---------------------
Per (dataset, cell): <log_dir>/<dataset_slug>/<label>/results.csv, updated atomically
after every split. Artefacts (logs, monitor history, plots) are saved in
<log_dir>/<dataset_slug>/<label>/runs/<split>/.
Existing successful runs are skipped unless --rerun_successful is passed.
Merged results across all completed tasks are written atomically to
<log_dir>/grid_results.csv and <log_dir>/grid_summary.csv.
"""

import argparse
import glob
import itertools
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

try:
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None

DATASETS = ("Buchwald-Hartwig", "Suzuki-Miyaura")
DATASET_SLUGS = {
    "Buchwald-Hartwig": "bh",
    "Suzuki-Miyaura": "suzuki",
}

CELL_LABELS_MAP = {
    ("ind", "reactants", "mean", "linear"): "baseline",
    ("ind", "reactants", "mean", "mlp"): "baseline_mlp",
    ("ind", "reactants", "rn", "linear"): "B_linear",
    ("ind", "reactants", "rn", "mlp"): "B_mlp",
    ("ind", "none", "mean", "linear"): "control_linear",
    ("ind", "none", "mean", "mlp"): "control_mlp",
    ("ind", "none", "rn", "linear"): "A_linear",
    ("ind", "none", "rn", "mlp"): "A_mlp",
    ("comb", "reactants", "mean", "linear"): "approach3",
    ("comb", "reactants", "mean", "mlp"): "approach2",
    ("comb", "reactants", "rn", "linear"): "approach1_linear",
    ("comb", "reactants", "rn", "mlp"): "approach1_mlp",
    ("comb", "none", "mean", "linear"): "comb_none_mean_linear",
    ("comb", "none", "mean", "mlp"): "comb_none_mean_mlp",
    ("comb", "none", "rn", "linear"): "comb_none_rn_linear",
    ("comb", "none", "rn", "mlp"): "comb_none_rn_mlp",
}


def _build_cells():
    cells = []
    for tok, att, pool, head in itertools.product(
        REACTANT_TOKENS, ("reactants", "none"), REACTANT_POOLINGS, HEADS
    ):
        label = CELL_LABELS_MAP[(tok, att, pool, head)]
        cell_str = "%s/%s/%s/%s" % (tok, att, pool, head)
        cells.append(
            {
                "label": label,
                "cell": cell_str,
                "reactant_tokens": tok,
                "attention_on": att,
                "reactant_pooling": pool,
                "head": head,
                "description": "%s (%s)" % (label, cell_str),
            }
        )
    return tuple(cells)


CELLS = _build_cells()
CELL_LABELS = tuple(c["label"] for c in CELLS)
CELL_BY_LABEL = {c["label"]: c for c in CELLS}

SMOKE_TRAIN = 64
SMOKE_VALID = 32
SMOKE_TEST = 32
SMOKE_EMB_DIM = 32
SMOKE_BATCH_SIZE = 8
SMOKE_NUM_WORKERS = 0

RESULT_COLUMNS = [
    "dataset",
    "kind",
    "split",
    "seed",
    "label",
    "cell",
    "reactant_tokens",
    "attention_on",
    "reactant_pooling",
    "head",
    "profile",
    "note",
    "epochs",
    "patience",
    "emb_dim",
    "batch_size",
    "lr",
    "best_epoch",
    "epochs_ran",
    "n_train",
    "n_valid",
    "n_test",
    "train_r2",
    "train_mae",
    "train_rmse",
    "test_r2",
    "test_mae",
    "test_rmse",
    "test_pearson",
    "trainable_parameters",
    "train_runtime_sec",
    "seconds_per_epoch",
    "peak_gpu_mem_mb",
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
    "peak_gpu_mem_mb",
)

SUMMARY_COLUMNS = [
    "dataset",
    "kind",
    "label",
    "cell",
    "reactant_tokens",
    "attention_on",
    "reactant_pooling",
    "head",
    "n_runs",
    "n_failed",
    "trainable_parameters",
    "train_r2_mean",
    "train_r2_std",
    "train_mae_mean",
    "train_mae_std",
    "train_rmse_mean",
    "train_rmse_std",
    "test_r2_mean",
    "test_r2_std",
    "test_r2_pm",
    "test_mae_mean",
    "test_mae_std",
    "test_mae_pm",
    "test_rmse_mean",
    "test_rmse_std",
    "test_rmse_pm",
    "seconds_per_epoch_mean",
    "seconds_per_epoch_std",
    "peak_gpu_mem_mb_mean",
    "peak_gpu_mem_mb_std",
]


def _subset_mol(mol: dict, positions: np.ndarray) -> dict:
    """Slice a prepared mol dict to `positions`."""
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
    """Write a deterministic `n`-sample subset of a prepared npz."""
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


def _load_y(npz_path: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as npz:
        return np.asarray(npz["reaction"].item()["y"], dtype=float)


def build_parser() -> argparse.ArgumentParser:
    """Parser for the grid runner with defaults aligned with main_finetune."""
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

    # Ablation grid specific CLI options
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "full"],
        help="Run profile: smoke (1 split each, small subsets) or full.",
    )
    parser.add_argument(
        "--smoke_epochs",
        type=int,
        default=1,
        help="Number of epochs in smoke mode (default: 1).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS),
        help="Datasets to evaluate (Buchwald-Hartwig, Suzuki-Miyaura).",
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="+",
        default=None,
        help="Subset of cell labels to run (default: all 16 cells).",
    )
    parser.add_argument(
        "--bh_cv_ids",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Buchwald-Hartwig CV split IDs (1..10).",
    )
    parser.add_argument(
        "--bh_test_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Buchwald-Hartwig Test split IDs (1..4).",
    )
    parser.add_argument(
        "--suzuki_split_ids",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Suzuki-Miyaura split IDs (0..9).",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Slurm array task ID (0..31): K // 16 dataset, K %% 16 cell.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned trainings and verify data files without training.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Rebuild grid_results.csv and grid_summary.csv without training.",
    )
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="Rerun splits even if previously succeeded.",
    )
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Retain model.pt files after successful evaluation.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../logs/grid/",
        help="Output directory for grid results and logs.",
    )
    parser.add_argument(
        "--run-cell",
        dest="run_cell",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def task_id_to_dataset_and_cell(task_id: int):
    """Maps array task ID (0..31) to (dataset_name, cell_dict)."""
    if task_id < 0 or task_id >= 32:
        raise ValueError("task_id must be between 0 and 31 (got %d)" % task_id)
    dataset = DATASETS[task_id // 16]
    cell = CELLS[task_id % 16]
    return dataset, cell


def tasks(args=None, profile: str = None) -> list:
    """
    List split tasks according to profile and CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Arguments namespace. If None, default parser values are used.
    profile : str, optional
        'smoke' or 'full'. Defaults to args.profile.

    Returns
    -------
    list of dict
        Task dicts with keys dataset, kind, split, npz_folder, seed.
    """
    if args is None:
        args = build_parser().parse_args([])
    if profile is None:
        profile = getattr(args, "profile", "smoke")

    bh_cv_ids = list(getattr(args, "bh_cv_ids", range(1, 11)))
    bh_test_ids = list(getattr(args, "bh_test_ids", [1, 2, 3, 4]))
    suzuki_split_ids = list(getattr(args, "suzuki_split_ids", range(10)))

    if profile == "smoke":
        bh_cv_ids = bh_cv_ids[:1]
        bh_test_ids = bh_test_ids[:1]
        suzuki_split_ids = suzuki_split_ids[:1]

    data_folder = getattr(args, "Data_folder", "../Data/")
    seed = getattr(args, "seed", 42)
    task_list = []

    # Buchwald-Hartwig CV splits
    for cv_id in bh_cv_ids:
        split_name = "fullcv%02d_split70" % cv_id
        task_list.append(
            {
                "dataset": "Buchwald-Hartwig",
                "kind": "cv",
                "split": split_name,
                "npz_folder": "npz/bh/%s" % split_name,
                "seed": seed,
            }
        )

    # Buchwald-Hartwig Test splits
    for test_id in bh_test_ids:
        split_name = "test%d" % test_id
        task_list.append(
            {
                "dataset": "Buchwald-Hartwig",
                "kind": "test",
                "split": split_name,
                "npz_folder": "npz/bh/%s" % split_name,
                "seed": seed,
            }
        )

    # Suzuki-Miyaura splits
    for s_id in suzuki_split_ids:
        split_name = "split_%d" % s_id
        npz_folder = "processed/suzuki/npz/%s" % split_name
        meta_path = os.path.join(data_folder, npz_folder, "split_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as handle:
                meta = json.load(handle)
            s_seed = int(meta["seed"])
        else:
            s_seed = int(seed) + int(s_id)
        task_list.append(
            {
                "dataset": "Suzuki-Miyaura",
                "kind": "cv",
                "split": split_name,
                "npz_folder": npz_folder,
                "seed": s_seed,
            }
        )

    return task_list


def _read_epochs_ran(monitor_folder: str) -> int:
    path = os.path.join(monitor_folder, "history.json")
    if not os.path.isfile(path):
        return 0
    with open(path) as handle:
        history = json.load(handle)
    return int(len(history.get("epoch", [])))


def _evaluate_checkpoint(args, device):
    checkpoint_file = os.path.join(args.model_path, args.model_name)
    checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=device)
    if "model_config" in checkpoint:
        net = model.from_config(checkpoint["model_config"]).to(device)
    else:
        net = model(
            node_in_feats=155,
            edge_in_feats=9,
            num_layer=args.layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.dropout,
            num_attention_layer=args.attention_layer,
            num_heads=args.num_heads,
            reaction_combine=args.reaction_combine,
            attention_on=args.attention_on,
            reactant_tokens=args.reactant_tokens,
            reactant_pooling=args.reactant_pooling,
            head=args.head,
        ).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    folder = os.path.join(args.Data_folder, args.npz_folder) + os.sep
    metrics = {}
    for name in ("train", "test"):
        dataset = GraphDataset(os.path.join(folder, name + ".npz"))
        loader = DataLoader(
            dataset,
            batch_size=int(np.min([args.batch_size, len(dataset)])),
            shuffle=False,
            collate_fn=collate_reaction_graphs,
            num_workers=args.num_workers,
            drop_last=False,
        )
        metrics[name] = validation(args, net, loader, device)[0]

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return metrics["train"], metrics["test"], int(trainable)


def _run_one_cell(spec_path: str) -> None:
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
    os.makedirs(args.model_path, exist_ok=True)

    checkpoint_path = os.path.join(args.model_path, args.model_name)
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    history_path = os.path.join(args.monitor_folder, "history.json")
    if os.path.isfile(history_path):
        os.remove(history_path)

    set_seed(args.seed)
    gpus = resolve_gpu_ids(args)
    device = torch.device("cuda:%d" % gpus[0]) if gpus else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = finetune(args, save_embedding=False)
    train_metrics, test_metrics, trainable = _evaluate_checkpoint(args, device)
    epochs_ran = _read_epochs_ran(args.monitor_folder)
    runtime = float(result["train_runtime_sec"])
    seconds_per_epoch = runtime / epochs_ran if epochs_ran > 0 else None

    peak_gpu = None
    if device.type == "cuda":
        peak_gpu = round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)

    if not spec.get("keep_checkpoints", False) and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    out = {
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(epochs_ran),
        "n_train": int(result["n_train"]),
        "n_valid": int(result["n_valid"]),
        "n_test": int(result["n_test"]),
        "train_r2": train_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_pearson": test_metrics["pearson"],
        "trainable_parameters": int(trainable),
        "train_runtime_sec": runtime,
        "seconds_per_epoch": seconds_per_epoch,
        "peak_gpu_mem_mb": peak_gpu,
    }
    with open(spec["result_path"], "w") as handle:
        json.dump(out, handle)


def _run_child(cmd: list, env: dict, log_path: str):
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
        "attention_on": cell["attention_on"],
        "reactant_tokens": cell["reactant_tokens"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
        "seed": task["seed"],
        "keep_checkpoints": getattr(args, "keep_checkpoints", False),
        "result_path": os.path.join(run_dir, "cell_result.json"),
    }


def _prepare_smoke_subsets(args, tasks_list: list, log_dir: str, logger) -> dict:
    smoke_folders = {}
    sizes = {"train": SMOKE_TRAIN, "valid": SMOKE_VALID, "test": SMOKE_TEST}
    for task in tasks_list:
        slug = DATASET_SLUGS.get(task["dataset"], task["dataset"].lower())
        source_dir = os.path.join(args.Data_folder, task["npz_folder"])
        target_dir = os.path.join(log_dir, "smoke_npz", "%s_%s" % (slug, task["split"]))
        for name, n in sizes.items():
            target = os.path.join(target_dir, name + ".npz")
            if not os.path.isfile(target):
                write_subset(
                    os.path.join(source_dir, name + ".npz"), target, n, seed=args.seed
                )
            if np.std(_load_y(target)) == 0.0:
                raise RuntimeError(
                    "subset %s has zero target variance; R2/Pearson would be None" % target
                )
        smoke_folders[(task["dataset"], task["split"])] = target_dir
        if logger:
            logger.info("--- smoke subsets for %s/%s -> %s" % (task["dataset"], task["split"], target_dir))
    return smoke_folders


def _execute_split_run(args, task: dict, cell: dict, npz_folder: str, run_dir: str, profile_info: tuple, logger):
    profile, note = profile_info
    spec = _cell_spec(args, task, cell, run_dir, npz_folder)
    spec_path = os.path.join(run_dir, "cell_spec.json")
    with open(spec_path, "w") as handle:
        json.dump(spec, handle)

    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    cmd = [sys.executable, os.path.abspath(__file__), "--run-cell", spec_path]
    log_path = os.path.join(run_dir, "cell.log")

    start_time = time.time()
    code, peak_rss = _run_child(cmd, env, log_path)
    res_path = spec["result_path"]

    row = {
        "dataset": task["dataset"],
        "kind": task["kind"],
        "split": task["split"],
        "seed": task["seed"],
        "label": cell["label"],
        "cell": cell["cell"],
        "reactant_tokens": cell["reactant_tokens"],
        "attention_on": cell["attention_on"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
        "profile": profile,
        "note": note,
        "epochs": args.epochs,
        "patience": args.patience,
        "emb_dim": args.emb_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "peak_rss_mb": peak_rss,
    }

    if code == 0 and os.path.isfile(res_path):
        with open(res_path) as handle:
            metrics = json.load(handle)
        row.update(metrics)
        row["status"] = "success"
        row["error"] = ""
        if logger:
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
        row["status"] = "failed"
        row["error"] = "child exit code %d" % code
        row["train_runtime_sec"] = round(time.time() - start_time, 1)
        for col in RESULT_COLUMNS:
            if col not in row:
                row[col] = None
        if logger:
            logger.error(
                "--- %s | %s | %s FAILED (exit %d); see %s"
                % (task["dataset"], task["split"], cell["label"], code, log_path)
            )
    return row


def _build_summary_row(d: str, k: str, label: str, subset: pd.DataFrame, success_subset: pd.DataFrame) -> dict:
    n_runs = int(len(success_subset))
    n_failed = int((subset["status"] != "success").sum())
    first_row = success_subset.iloc[0]
    cell_info = CELL_BY_LABEL.get(label, {})

    row = {
        "dataset": d,
        "kind": k,
        "label": label,
        "cell": cell_info.get("cell", first_row.get("cell", "")),
        "reactant_tokens": cell_info.get("reactant_tokens", first_row.get("reactant_tokens", "")),
        "attention_on": cell_info.get("attention_on", first_row.get("attention_on", "")),
        "reactant_pooling": cell_info.get("reactant_pooling", first_row.get("reactant_pooling", "")),
        "head": cell_info.get("head", first_row.get("head", "")),
        "n_runs": n_runs,
        "n_failed": n_failed,
        "trainable_parameters": (
            int(first_row["trainable_parameters"])
            if pd.notna(first_row.get("trainable_parameters"))
            else np.nan
        ),
    }

    for metric in SUMMARY_METRICS:
        vals = pd.to_numeric(success_subset[metric], errors="coerce").dropna()
        row[metric + "_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        row[metric + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    for pm in ("test_r2", "test_mae", "test_rmse"):
        m_val = row[pm + "_mean"]
        s_val = row[pm + "_std"]
        if np.isnan(m_val):
            row[pm + "_pm"] = "nan ± nan"
        elif np.isnan(s_val):
            row[pm + "_pm"] = "%.4f ± nan" % m_val
        else:
            row[pm + "_pm"] = "%.4f ± %.4f" % (m_val, s_val)

    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) across successful splits."""
    if len(results) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    dataset_kind_pairs = []
    for d in DATASETS:
        for k in ("cv", "test"):
            if ((results["dataset"] == d) & (results["kind"] == k)).any():
                dataset_kind_pairs.append((d, k))
    for pair in results[["dataset", "kind"]].drop_duplicates().itertuples(index=False):
        if (pair.dataset, pair.kind) not in dataset_kind_pairs:
            dataset_kind_pairs.append((pair.dataset, pair.kind))

    cell_order = [c["label"] for c in CELLS]
    rows = []

    for d, k in dataset_kind_pairs:
        group_df = results[(results["dataset"] == d) & (results["kind"] == k)]
        labels_present = group_df["label"].unique()
        ordered_labels = [lbl for lbl in cell_order if lbl in labels_present] + [
            lbl for lbl in labels_present if lbl not in CELL_BY_LABEL
        ]

        for label in ordered_labels:
            subset = group_df[group_df["label"] == label]
            success_subset = subset[subset["status"] == "success"]
            if len(success_subset) > 0:
                rows.append(_build_summary_row(d, k, label, subset, success_subset))

    summary_df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    return summary_df.reindex(columns=SUMMARY_COLUMNS)


def merge_results(log_dir: str) -> tuple:
    """Merges all per-task results.csv into grid_results.csv and grid_summary.csv."""
    pattern = os.path.join(log_dir, "*", "*", "results.csv")
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        if os.path.dirname(f) == os.path.abspath(log_dir):
            continue
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=RESULT_COLUMNS)

    for col in RESULT_COLUMNS:
        if col not in all_df.columns:
            all_df[col] = None
    all_df = all_df.reindex(columns=RESULT_COLUMNS)

    grid_results_path = os.path.join(log_dir, "grid_results.csv")
    grid_summary_path = os.path.join(log_dir, "grid_summary.csv")

    tmp_res = grid_results_path + ".tmp"
    all_df.to_csv(tmp_res, index=False)
    os.replace(tmp_res, grid_results_path)

    summary_df = summarize(all_df)
    tmp_sum = grid_summary_path + ".tmp"
    summary_df.to_csv(tmp_sum, index=False)
    os.replace(tmp_sum, grid_summary_path)

    return all_df, summary_df


def _dry_run_check(args, planned_trainings: list) -> None:
    print("=" * 80)
    print("DRY RUN: %d planned trainings" % len(planned_trainings))
    print("=" * 80)
    missing_files = []
    unique_npz_dirs = set()

    for item in planned_trainings:
        task_id = item["task_id"]
        task = item["task"]
        cell = item["cell"]
        dataset = task["dataset"]
        kind = task["kind"]
        split = task["split"]
        seed = task["seed"]
        cell_str = cell["cell"]
        label = cell["label"]

        full_npz_dir = os.path.join(args.Data_folder, task["npz_folder"])
        unique_npz_dirs.add(full_npz_dir)
        for req in ("train.npz", "valid.npz", "test.npz"):
            p = os.path.join(full_npz_dir, req)
            if not os.path.isfile(p):
                missing_files.append(p)
        if dataset == "Suzuki-Miyaura":
            meta_p = os.path.join(full_npz_dir, "split_metadata.json")
            if not os.path.isfile(meta_p):
                missing_files.append(meta_p)

        print(
            "task_id=%2d | dataset=%-17s | kind=%-4s | split=%-16s | seed=%4d | label=%-21s | cell=%s | "
            "epochs=%d patience=%d emb_dim=%d batch_size=%d lr=%g"
            % (
                task_id,
                dataset,
                kind,
                split,
                seed,
                label,
                cell_str,
                args.epochs,
                args.patience,
                args.emb_dim,
                args.batch_size,
                args.lr,
            )
        )

    print("-" * 80)
    print("Total planned trainings: %d" % len(planned_trainings))
    print("Unique split directories checked: %d" % len(unique_npz_dirs))
    if missing_files:
        print("ERROR: %d required files are MISSING:" % len(missing_files))
        for mf in missing_files[:10]:
            print("  MISSING:", mf)
        if len(missing_files) > 10:
            print("  ... and %d more" % (len(missing_files) - 10))
        sys.exit(1)
    else:
        print("All required npz and metadata files are PRESENT.")
    print("=" * 80)


def _resolve_planned_trainings(args, all_tasks: list) -> list:
    planned = []
    if args.task_id is not None:
        target_dataset, target_cell = task_id_to_dataset_and_cell(args.task_id)
        for task in all_tasks:
            if task["dataset"] == target_dataset:
                planned.append({"task_id": args.task_id, "task": task, "cell": target_cell})
        return planned

    selected_datasets = args.datasets or list(DATASETS)
    selected_cells = [CELL_BY_LABEL[lbl] for lbl in args.cells] if args.cells else list(CELLS)

    for task in all_tasks:
        if task["dataset"] not in selected_datasets:
            continue
        dataset_idx = DATASETS.index(task["dataset"])
        for cell in selected_cells:
            cell_idx = CELLS.index(cell)
            t_id = dataset_idx * 16 + cell_idx
            planned.append({"task_id": t_id, "task": task, "cell": cell})
    return planned


def _load_existing_cell_results(results_csv: str) -> tuple:
    existing_rows = []
    existing_status = {}
    if os.path.isfile(results_csv):
        try:
            prev_df = pd.read_csv(results_csv)
            for _, r in prev_df.iterrows():
                row_dict = r.to_dict()
                existing_rows.append(row_dict)
                existing_status[row_dict["split"]] = row_dict.get("status")
        except Exception:
            pass
    return existing_rows, existing_status


def _save_cell_results(results_csv: str, existing_rows: list, row: dict, split: str) -> None:
    replaced = False
    for idx, ex in enumerate(existing_rows):
        if ex.get("split") == split:
            existing_rows[idx] = row
            replaced = True
            break
    if not replaced:
        existing_rows.append(row)

    table = pd.DataFrame(existing_rows).reindex(columns=RESULT_COLUMNS)
    tmp_path = results_csv + ".tmp"
    table.to_csv(tmp_path, index=False)
    os.replace(tmp_path, results_csv)


def _process_cell_tasks(args, items: list, smoke_folders: dict, profile_info: tuple, log_dir: str, logger) -> None:
    dataset = items[0]["task"]["dataset"]
    label = items[0]["cell"]["label"]
    cell = items[0]["cell"]
    profile_str, note_str = profile_info

    slug = DATASET_SLUGS.get(dataset, dataset.lower())
    cell_dir = os.path.join(log_dir, slug, label)
    os.makedirs(cell_dir, exist_ok=True)
    results_csv = os.path.join(cell_dir, "results.csv")

    existing_rows, existing_status = _load_existing_cell_results(results_csv)

    for item in items:
        task = item["task"]
        split = task["split"]

        if not args.rerun_successful and existing_status.get(split) == "success":
            logger.info("--- %s | %s | %s already succeeded; skipping" % (dataset, split, label))
            continue

        run_dir = os.path.join(cell_dir, "runs", split)
        os.makedirs(run_dir, exist_ok=True)
        if args.profile == "smoke":
            npz_folder = smoke_folders[(dataset, split)]
        else:
            npz_folder = os.path.join(args.Data_folder, task["npz_folder"])

        row = _execute_split_run(args, task, cell, npz_folder, run_dir, profile_info, logger)
        _save_cell_results(results_csv, existing_rows, row, split)
        existing_status[split] = row["status"]

    merge_results(log_dir)


def run_grid(args) -> None:
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    if args.collect:
        all_df, sum_df = merge_results(log_dir)
        print("\n=== SUMMARY (%s)" % os.path.join(log_dir, "grid_summary.csv"))
        print(sum_df.to_string(index=False))
        return

    profile = args.profile
    if profile == "smoke":
        args.epochs = args.smoke_epochs
        args.emb_dim = SMOKE_EMB_DIM
        args.batch_size = SMOKE_BATCH_SIZE
        args.num_workers = SMOKE_NUM_WORKERS
        args.patience = 0
        profile_str = "laptop_smoke_correctness_only"
        note_str = "not a result"
        with open(os.path.join(log_dir, "PROFILE_NOT_RESULTS.txt"), "w") as handle:
            handle.write(
                "Profile: %s\nThese metrics are a correctness check on tiny subsets,\n"
                "NOT results. Do not report R2/MAE/RMSE from this directory.\n"
                % profile_str
            )
    else:
        profile_str = "full"
        note_str = ""

    all_tasks = tasks(args, profile=profile)
    planned_trainings = _resolve_planned_trainings(args, all_tasks)

    if args.dry_run:
        _dry_run_check(args, planned_trainings)
        return

    logger = setup_logging(log_filename=os.path.join(log_dir, "grid.log"))
    logger.info("=" * 80)
    logger.info("Ablation grid runner: profile=%s, planned=%d trainings" % (profile, len(planned_trainings)))

    smoke_folders = {}
    if profile == "smoke":
        smoke_folders = _prepare_smoke_subsets(args, all_tasks, log_dir, logger)

    by_cell_task = {}
    for item in planned_trainings:
        key = (item["task"]["dataset"], item["cell"]["label"])
        by_cell_task.setdefault(key, []).append(item)

    for items in by_cell_task.values():
        _process_cell_tasks(args, items, smoke_folders, (profile_str, note_str), log_dir, logger)

    all_res, sum_res = merge_results(log_dir)
    logger.info("=" * 80)
    logger.info("Results table: %s" % os.path.join(log_dir, "grid_results.csv"))
    logger.info("Summary table: %s" % os.path.join(log_dir, "grid_summary.csv"))
    print("\n=== SUMMARY (%s)" % os.path.join(log_dir, "grid_summary.csv"))
    print(sum_res.to_string(index=False))


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.run_cell:
        _run_one_cell(parsed.run_cell)
    else:
        run_grid(parsed)
