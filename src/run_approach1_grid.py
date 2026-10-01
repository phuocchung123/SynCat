"""
Standalone experiment runner for Approach 1 grid evaluation.

Runs the 8-cell comparison across reactant_tokens, reactant_pooling, and head:
  1. current_model:      reactant_tokens=ind,  reactant_pooling=mean, head=linear
  2. approach_3:         reactant_tokens=comb, reactant_pooling=mean, head=linear
  3. approach_2:         reactant_tokens=comb, reactant_pooling=mean, head=mlp
  4. approach_1_linear:  reactant_tokens=comb, reactant_pooling=rn,   head=linear
  5. approach_1_mlp:     reactant_tokens=comb, reactant_pooling=rn,   head=mlp
  6. option_b_linear:    reactant_tokens=ind,  reactant_pooling=rn,   head=linear
  7. option_b_mlp:       reactant_tokens=ind,  reactant_pooling=rn,   head=mlp
  8. ind_mean_mlp:       reactant_tokens=ind,  reactant_pooling=mean, head=mlp

Evaluated on Buchwald-Hartwig and Suzuki-Miyaura splits.
Supports both fast smoke testing and full cluster execution with resumption.
"""

import argparse
import copy
import gc
import json
import logging
import os
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import GraphDataset
from finetune import _build_model, finetune
from model import model
from utils import collate_reaction_graphs, set_seed, setup_logging
from validation import validation

CELLS = [
    {
        "label": "current_model",
        "approach": "current model",
        "reactant_tokens": "ind",
        "reactant_pooling": "mean",
        "head": "linear",
    },
    {
        "label": "approach_3",
        "approach": "approach 3",
        "reactant_tokens": "comb",
        "reactant_pooling": "mean",
        "head": "linear",
    },
    {
        "label": "approach_2",
        "approach": "approach 2",
        "reactant_tokens": "comb",
        "reactant_pooling": "mean",
        "head": "mlp",
    },
    {
        "label": "approach_1_linear",
        "approach": "approach 1",
        "reactant_tokens": "comb",
        "reactant_pooling": "rn",
        "head": "linear",
    },
    {
        "label": "approach_1_mlp",
        "approach": "approach 1",
        "reactant_tokens": "comb",
        "reactant_pooling": "rn",
        "head": "mlp",
    },
    {
        "label": "option_b_linear",
        "approach": "option B",
        "reactant_tokens": "ind",
        "reactant_pooling": "rn",
        "head": "linear",
    },
    {
        "label": "option_b_mlp",
        "approach": "option B",
        "reactant_tokens": "ind",
        "reactant_pooling": "rn",
        "head": "mlp",
    },
    {
        "label": "ind_mean_mlp",
        "approach": "option B",
        "reactant_tokens": "ind",
        "reactant_pooling": "mean",
        "head": "mlp",
    },
]

METRICS = [
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
]


def _get_peak_memory_mb() -> Optional[float]:
    """Get peak process RSS in megabytes if psutil is available."""
    try:
        import psutil

        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def _slice_npz(src_path: str, dst_path: str, n_samples: int) -> None:
    """Creates a temporary NPZ file holding only the first n_samples."""
    with np.load(src_path, allow_pickle=True) as data:
        rmol = list(data["rmol"])
        pmol = list(data["pmol"])
        reaction = data["reaction"].item()

        total = len(reaction["y"])
        n = min(n_samples, total)

        sliced_rmol = []
        for slot_dict in rmol:
            n_nodes = slot_dict["n_node"][:n]
            tot_nodes = int(np.sum(n_nodes))
            n_edges = slot_dict["n_edge"][:n]
            tot_edges = int(np.sum(n_edges))
            sliced_rmol.append(
                {
                    "n_node": n_nodes,
                    "n_edge": n_edges,
                    "dummy": slot_dict["dummy"][:n],
                    "node_attr": slot_dict["node_attr"][:tot_nodes],
                    "edge_attr": slot_dict["edge_attr"][:tot_edges],
                    "src": slot_dict["src"][:tot_edges],
                    "dst": slot_dict["dst"][:tot_edges],
                }
            )

        sliced_pmol = []
        for slot_dict in pmol:
            n_nodes = slot_dict["n_node"][:n]
            tot_nodes = int(np.sum(n_nodes))
            n_edges = slot_dict["n_edge"][:n]
            tot_edges = int(np.sum(n_edges))
            sliced_pmol.append(
                {
                    "n_node": n_nodes,
                    "n_edge": n_edges,
                    "dummy": slot_dict["dummy"][:n],
                    "node_attr": slot_dict["node_attr"][:tot_nodes],
                    "edge_attr": slot_dict["edge_attr"][:tot_edges],
                    "src": slot_dict["src"][:tot_edges],
                    "dst": slot_dict["dst"][:tot_edges],
                }
            )

        sliced_reaction = {
            "y": reaction["y"][:n],
            "rsmi": reaction["rsmi"][:n],
        }

        extra = {}
        if "sample_ids" in data:
            extra["sample_ids"] = data["sample_ids"][:n]
        if "source_row_indices" in data:
            extra["source_row_indices"] = data["source_row_indices"][:n]

        np.savez_compressed(
            dst_path,
            rmol=sliced_rmol,
            pmol=sliced_pmol,
            reaction=sliced_reaction,
            **extra,
        )


def _prepare_smoke_data(
    base_data_folder: str,
    bh_split: str,
    suzuki_split: str,
    temp_dir: str,
    datasets: str = "all",
) -> Dict[str, str]:
    """Slices small smoke datasets for BH and Suzuki."""
    splits = {}

    if datasets in ("bh", "all"):
        bh_src = os.path.join(base_data_folder, "npz", "bh", bh_split)
        bh_dst = os.path.join(temp_dir, "bh", bh_split)
        os.makedirs(bh_dst, exist_ok=True)
        _slice_npz(os.path.join(bh_src, "train.npz"), os.path.join(bh_dst, "train.npz"), 64)
        _slice_npz(os.path.join(bh_src, "valid.npz"), os.path.join(bh_dst, "valid.npz"), 32)
        _slice_npz(os.path.join(bh_src, "test.npz"), os.path.join(bh_dst, "test.npz"), 32)
        splits["bh"] = bh_dst

    if datasets in ("suzuki", "all"):
        suz_src = os.path.join(base_data_folder, "processed", "suzuki", "npz", suzuki_split)
        suz_dst = os.path.join(temp_dir, "suzuki", suzuki_split)
        os.makedirs(suz_dst, exist_ok=True)
        _slice_npz(os.path.join(suz_src, "train.npz"), os.path.join(suz_dst, "train.npz"), 64)
        _slice_npz(os.path.join(suz_src, "valid.npz"), os.path.join(suz_dst, "valid.npz"), 32)
        _slice_npz(os.path.join(suz_src, "test.npz"), os.path.join(suz_dst, "test.npz"), 32)
        splits["suzuki"] = suz_dst

    return splits


def run_cell(
    base_args: argparse.Namespace,
    dataset_name: str,
    split_name: str,
    cell: dict,
    npz_dir: str,
    run_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """Runs a single cell of the experiment grid."""
    row = {
        "dataset": dataset_name,
        "split": split_name,
        "cell": cell["label"],
        "label": cell["label"],
        "approach": cell.get("approach", ""),
        "tokens": cell["reactant_tokens"],
        "reactant_tokens": cell["reactant_tokens"],
        "pooling": cell["reactant_pooling"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
        "attention_on": getattr(base_args, "attention_on", "reactants"),
        "seed": base_args.seed,
        "train_r2": None,
        "train_mae": None,
        "train_rmse": None,
        "test_r2": None,
        "test_mae": None,
        "test_rmse": None,
        "trainable_parameters": None,
        "epochs_ran": None,
        "train_runtime_sec": None,
        "seconds_per_epoch": None,
        "peak_memory_mb": None,
        "status": "failed",
        "error": "",
    }

    checkpoint_file = os.path.join(run_dir, "model.pt")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    run_opts = copy.copy(base_args)
    run_opts.reactant_tokens = cell["reactant_tokens"]
    run_opts.reactant_pooling = cell["reactant_pooling"]
    run_opts.head = cell["head"]
    run_opts.attention_on = getattr(base_args, "attention_on", "reactants")
    run_opts.model_path = os.path.join(run_dir, "")
    run_opts.model_name = "model.pt"
    run_opts.monitor_folder = os.path.join(run_dir, "monitor", "")
    run_opts.image_folder = os.path.join(run_dir, "images", "")
    run_opts.Data_folder = os.path.dirname(os.path.abspath(npz_dir)) + os.sep
    run_opts.npz_folder = os.path.basename(os.path.abspath(npz_dir))

    os.makedirs(run_opts.monitor_folder, exist_ok=True)
    os.makedirs(run_opts.image_folder, exist_ok=True)

    set_seed(run_opts.seed)
    logger.info(
        "=== RUNNING: dataset=%s split=%s cell=%s (tokens=%s, pooling=%s, head=%s) ==="
        % (
            dataset_name,
            split_name,
            cell["label"],
            cell["reactant_tokens"],
            cell["reactant_pooling"],
            cell["head"],
        )
    )

    try:
        result = finetune(run_opts, save_embedding=False)

        train_set = GraphDataset(os.path.join(npz_dir, "train.npz"))
        node_dim = train_set.rmol_node_attr[0].shape[1]
        edge_dim = train_set.rmol_edge_attr[0].shape[1]

        # Reload best checkpoint to evaluate on complete train set
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if "model_config" in checkpoint:
            eval_net = model.from_config(checkpoint["model_config"]).to(device)
        else:
            eval_net = _build_model(run_opts, node_dim, edge_dim).to(device)
        eval_net.load_state_dict(checkpoint["model_state_dict"])
        eval_net.eval()

        trainable_params = sum(
            p.numel() for p in eval_net.parameters() if p.requires_grad
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=int(np.min([run_opts.batch_size, len(train_set)])),
            shuffle=False,
            collate_fn=collate_reaction_graphs,
            num_workers=run_opts.num_workers,
        )
        train_res = validation(
            run_opts, eval_net, train_loader, device, loss_fn=None
        )
        train_metrics = train_res[0] if isinstance(train_res, tuple) else train_res

        del eval_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        history_path = os.path.join(run_opts.monitor_folder, "history.json")
        epochs_ran = 0
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                h = json.load(f)
                epochs_ran = len(h.get("epoch", []))
        if epochs_ran == 0:
            epochs_ran = max(1, result.get("best_epoch", 0) + 1)

        train_runtime = result["train_runtime_sec"]
        sec_per_epoch = train_runtime / epochs_ran if epochs_ran > 0 else 0.0

        test_m = result["test_metrics"]
        row.update(
            {
                "status": "success",
                "train_r2": train_metrics.get("r2"),
                "train_mae": train_metrics.get("mae"),
                "train_rmse": train_metrics.get("rmse"),
                "test_r2": test_m.get("r2"),
                "test_mae": test_m.get("mae"),
                "test_rmse": test_m.get("rmse"),
                "trainable_parameters": trainable_params,
                "epochs_ran": epochs_ran,
                "train_runtime_sec": train_runtime,
                "seconds_per_epoch": sec_per_epoch,
                "peak_memory_mb": _get_peak_memory_mb(),
            }
        )
        logger.info(
            "--- Cell %s DONE: test MAE=%.4f RMSE=%.4f R2=%s | sec/epoch=%.2f"
            % (
                cell["label"],
                test_m["mae"],
                test_m["rmse"],
                "n/a" if test_m["r2"] is None else "%.4f" % test_m["r2"],
                sec_per_epoch,
            )
        )
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        logger.error("--- Cell %s FAILED: %s" % (cell["label"], row["error"]))
        logger.error(traceback.format_exc())

    return row


def summarize_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Group by dataset and cell to produce mean and sample std (ddof=1)."""
    success = df[df["status"] == "success"]
    rows = []
    group_cols = [
        "dataset",
        "cell",
        "approach",
        "reactant_tokens",
        "reactant_pooling",
        "head",
        "attention_on",
    ]
    actual_cols = [c for c in group_cols if c in success.columns]
    for keys, group in success.groupby(actual_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        entry = dict(zip(actual_cols, keys))
        entry["n_successful"] = len(group)
        for m in METRICS:
            if m in group.columns:
                vals = pd.to_numeric(group[m], errors="coerce").dropna()
                entry[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                entry[f"{m}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def _emit_server_commands(output_dir: str) -> None:
    """Write server_commands.txt with exact production commands and sizing notes."""
    filepath = os.path.join(output_dir, "server_commands.txt")
    lines = [
        "# ============================================================================",
        "# Server Commands for Approach 1 Evaluation Grid",
        "# ============================================================================",
        "# Sizing notes (measured at full settings: emb_dim 384, batch_size 128 on CPU):",
        "# - Buchwald-Hartwig: 6 reactant slots -> T = 21 tokens, 210 pair terms.",
        "#   Forward+backward: ~1.2 s/batch. 14 splits run via slurm/approach1_grid_bh.sbatch.",
        "# - Suzuki-Miyaura: 14 reactant slots -> T = 105 tokens, 5460 pair terms.",
        "#   Forward+backward: ~70 s/batch on CPU with 2.15 GB pair tensor.",
        "#   Real training runs on GPU server via slurm/approach1_grid_suzuki.sbatch.",
        "# ============================================================================",
        "",
        "# Submit SLURM batch jobs from repo root:",
        "#   sbatch slurm/approach1_grid_smoke.sbatch",
        "#   sbatch slurm/approach1_grid_bh.sbatch",
        "#   sbatch slurm/approach1_grid_suzuki.sbatch",
        "",
    ]
    for cell in CELLS:
        label = cell["label"]
        tokens = cell["reactant_tokens"]
        pooling = cell["reactant_pooling"]
        head = cell["head"]
        lines.append(f"# Cell: {label} ({cell['approach']})")
        lines.append(
            f"# BH: python train_bh.py --reactant_tokens {tokens} --reactant_pooling {pooling} "
            f"--head {head} --emb_dim 384 --batch_size 128 --epochs 100 --patience 10 "
            f"--cv_ids 1 2 3 4 5 6 7 8 9 10 --split_columns split_70"
        )
        lines.append(
            f"# Suzuki: python run_splits_sequential.py --reactant_tokens {tokens} --reactant_pooling {pooling} "
            f"--head {head} --emb_dim 384 --batch_size 128 --epochs 100 --patience 10 "
            f"--split_ids 0 1 2 3 4 5 6 7 8 9"
        )
        lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _resolve_data_folder(data_folder: str) -> str:
    """Finds valid Data directory from possible repo locations."""
    if os.path.exists(os.path.join(data_folder, "npz")):
        return data_folder
    candidates = [
        "Data/",
        "../Data/",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "npz")):
            return candidate
    return data_folder


def _resolve_eval_datasets(
    args: argparse.Namespace,
    temp_splits: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, str, str, str]]:
    """Builds list of (dataset_name, split_name, npz_dir, rxn_col, y_col) tuples."""
    eval_datasets = []
    if args.smoke and temp_splits is not None:
        if args.datasets in ("bh", "all") and "bh" in temp_splits:
            eval_datasets.append(("bh", "fullcv01_split70", temp_splits["bh"], "rxn", "Output"))
        if args.datasets in ("suzuki", "all") and "suzuki" in temp_splits:
            eval_datasets.append(("suzuki", "split_0", temp_splits["suzuki"], "rxn", "y"))
        return eval_datasets

    if args.datasets in ("bh", "all"):
        if args.bh_splits is not None:
            bh_splits_to_run = args.bh_splits
        else:
            bh_splits_to_run = []
            for cv_id in args.cv_ids:
                for column in args.split_columns:
                    suffix = column.replace("split_", "").replace(".", "_")
                    bh_splits_to_run.append("fullcv%02d_split%s" % (cv_id, suffix))
            for test_id in args.test_ids:
                bh_splits_to_run.append("test%d" % test_id)
        for s in bh_splits_to_run:
            npz_dir = os.path.join(args.Data_folder, "npz", "bh", s)
            eval_datasets.append(("bh", s, npz_dir, "rxn", "Output"))

    if args.datasets in ("suzuki", "all"):
        if args.suzuki_splits is not None:
            suzuki_splits_to_run = args.suzuki_splits
        else:
            suzuki_splits_to_run = ["split_%d" % i for i in args.split_ids]
        for s in suzuki_splits_to_run:
            npz_dir = os.path.join(args.Data_folder, "processed", "suzuki", "npz", s)
            eval_datasets.append(("suzuki", s, npz_dir, "rxn", "y"))

    return eval_datasets


def _load_existing_results(per_split_csv: str) -> List[dict]:
    """Load existing results if resuming."""
    if os.path.isfile(per_split_csv):
        df_existing = pd.read_csv(per_split_csv)
        return df_existing.to_dict("records")
    return []


def _save_results(
    results: List[dict],
    per_split_csv: str,
    summary_csv: str,
    is_smoke: bool = False,
    output_dir: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Saves per-split and summary CSV files."""
    df_results = pd.DataFrame(results)
    df_results.to_csv(per_split_csv, index=False)
    summary_df = summarize_grid(df_results)
    summary_df.to_csv(summary_csv, index=False)

    if is_smoke and output_dir:
        smoke_res_csv = os.path.join(output_dir, "approach1_smoke_results.csv")
        smoke_sum_csv = os.path.join(output_dir, "approach1_smoke_summary.csv")
        df_results.to_csv(smoke_res_csv, index=False)
        summary_df.to_csv(smoke_sum_csv, index=False)

    return df_results, summary_df


def build_grid_parser() -> argparse.ArgumentParser:
    """Builds argument parser for Approach 1 grid runner."""
    parser = argparse.ArgumentParser(description="Approach 1 Grid Runner")
    # Base training arguments
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
        choices=["reactants", "reactants_products", "all", "none"],
    )
    parser.add_argument(
        "--reaction_combine",
        type=str,
        default="concat",
        choices=["both", "concat", "concat_sub", "diff", "interaction", "mul", "prod_only", "sum"],
    )
    parser.add_argument("--reactant_tokens", type=str, default="ind", choices=["ind", "comb"])
    parser.add_argument("--reactant_pooling", type=str, default="mean", choices=["mean", "rn"])
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--emb_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--schedule", type=str, default="none", choices=["none", "step", "linear", "cosine"])
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--Data_folder", type=str, default="../Data/")
    parser.add_argument("--data_csv", type=str, default="raw/suzuki/random_split_0.tsv")
    parser.add_argument("--npz_folder", type=str, default="npz/npz_yield")
    parser.add_argument("--reaction_column", type=str, default="rxn")
    parser.add_argument("--y_column", type=str, default="Output")
    parser.add_argument("--model_path", type=str, default="../Data/model/")
    parser.add_argument("--model_name", type=str, default="model_yield.pt")
    parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    parser.add_argument("--image_folder", type=str, default="../Image/")
    parser.add_argument("--track_test_each_epoch", action="store_true")
    parser.add_argument("--save_embedding", action="store_true")
    parser.add_argument("--store_attention", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--log_dir", type=str, default=None)

    # Grid evaluation specific arguments
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run fast smoke test on tiny subsets (1 BH split, 1 Suzuki split, 1 epoch)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["bh", "suzuki", "all"],
        help="which datasets to run: 'bh', 'suzuki', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--cv_ids",
        type=int,
        nargs="*",
        default=list(range(1, 11)),
        help="BH FullCV ids (1..10)",
    )
    parser.add_argument(
        "--test_ids",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4],
        help="BH Test ids (1..4)",
    )
    parser.add_argument(
        "--split_columns",
        type=str,
        nargs="*",
        default=["split_70"],
        help="BH split columns (e.g. split_70)",
    )
    parser.add_argument(
        "--split_ids",
        type=int,
        nargs="*",
        default=list(range(10)),
        help="Suzuki split ids (0..9)",
    )
    parser.add_argument(
        "--bh_splits",
        type=str,
        nargs="*",
        default=None,
        help="explicit BH split folder names (overrides --cv_ids / --test_ids if passed)",
    )
    parser.add_argument(
        "--suzuki_splits",
        type=str,
        nargs="*",
        default=None,
        help="explicit Suzuki split folder names (overrides --split_ids if passed)",
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="*",
        default=["all"],
        choices=[
            "all",
            "current_model",
            "approach_3",
            "approach_2",
            "approach_1_linear",
            "approach_1_mlp",
            "option_b_linear",
            "option_b_mlp",
            "ind_mean_mlp",
        ],
        help="which experiment cells to run (default: 'all')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../logs/approach1_grid/",
        help="directory to write logs, checkpoints and CSV results",
    )
    parser.add_argument(
        "--rerun_successful",
        action="store_true",
        help="retrain cells that already have a successful result in the CSV",
    )
    return parser


def main():
    parser = build_grid_parser()
    args = parser.parse_args()

    if getattr(args, "log_dir", None) and args.log_dir != "../logs/suzuki_regression/":
        args.output_dir = args.log_dir

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(log_filename=os.path.join(args.output_dir, "grid.log"))
    logger.info("Starting Approach 1 Grid Runner (smoke=%s, datasets=%s)" % (args.smoke, args.datasets))

    args.Data_folder = _resolve_data_folder(args.Data_folder)
    _emit_server_commands(args.output_dir)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    per_split_csv = os.path.join(args.output_dir, "approach1_per_split.csv")
    summary_csv = os.path.join(args.output_dir, "approach1_summary.csv")

    active_cells = CELLS
    if "all" not in args.cells:
        active_cells = [c for c in CELLS if c["label"] in args.cells]

    temp_dir_obj = None
    try:
        temp_splits = None
        if args.smoke:
            logger.info("Setting up smoke test configuration...")
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_splits = _prepare_smoke_data(
                args.Data_folder,
                "fullcv01_split70",
                "split_0",
                temp_dir_obj.name,
                datasets=args.datasets,
            )
            # Override parameters for fast smoke execution
            args.epochs = 1
            args.batch_size = 8
            args.num_workers = 0
            args.layer = 1
            args.emb_dim = 32

        eval_datasets = _resolve_eval_datasets(args, temp_splits)
        results = _load_existing_results(per_split_csv)

        for dataset_name, split_name, npz_dir, rxn_col, y_col in eval_datasets:
            dataset_args = copy.copy(args)
            dataset_args.reaction_column = rxn_col
            dataset_args.y_column = y_col

            for cell in active_cells:
                already_done = any(
                    r.get("dataset") == dataset_name
                    and r.get("split") == split_name
                    and r.get("cell", r.get("label")) == cell["label"]
                    and r.get("status") == "success"
                    for r in results
                )
                if already_done and not args.rerun_successful:
                    logger.info(
                        "--- %s / %s / %s: already successful, skipping"
                        % (dataset_name, split_name, cell["label"])
                    )
                    continue

                run_dir = os.path.join(
                    args.output_dir, "runs", dataset_name, split_name, cell["label"]
                )
                os.makedirs(run_dir, exist_ok=True)
                row = run_cell(
                    dataset_args,
                    dataset_name,
                    split_name,
                    cell,
                    npz_dir,
                    run_dir,
                    device,
                    logger,
                )

                existing_idx = next(
                    (
                        i
                        for i, r in enumerate(results)
                        if r.get("dataset") == dataset_name
                        and r.get("split") == split_name
                        and r.get("cell", r.get("label")) == cell["label"]
                    ),
                    None,
                )
                if existing_idx is not None:
                    results[existing_idx] = row
                else:
                    results.append(row)

                _save_results(results, per_split_csv, summary_csv, is_smoke=args.smoke, output_dir=args.output_dir)

        if len(results):
            df_results, summary_df = _save_results(
                results, per_split_csv, summary_csv, is_smoke=args.smoke, output_dir=args.output_dir
            )
            logger.info("=== GRID COMPLETE ===")
            logger.info("Per-split CSV: %s" % per_split_csv)
            logger.info("Summary CSV:   %s" % summary_csv)
            print("\nSummary Results:\n", summary_df.to_string())

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
