"""
Standalone experiment runner for Approach 2 grid evaluation.

Runs the 4-cell comparison on Buchwald-Hartwig and Suzuki-Miyaura:
1. current_model: reactant_tokens=ind, head=linear
2. approach_3:    reactant_tokens=comb, head=linear
3. approach_2:    reactant_tokens=comb, head=mlp
4. mlp_control:   reactant_tokens=ind, head=mlp

All cells use attention_on="reactants" and identical hyperparameters.
Each split and cell gets a unique output/checkpoint folder.
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import GraphDataset
from finetune import _build_model, finetune
from main_finetune import build_parser
from model import model
from utils import collate_reaction_graphs, set_seed, setup_logging
from validation import validation

CELLS = [
    {"label": "current_model", "reactant_tokens": "ind", "head": "linear"},
    {"label": "approach_3", "reactant_tokens": "comb", "head": "linear"},
    {"label": "approach_2", "reactant_tokens": "comb", "head": "mlp"},
    {"label": "mlp_control", "reactant_tokens": "ind", "head": "mlp"},
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
) -> Dict[str, str]:
    """Slices small smoke datasets for BH and Suzuki."""
    splits = {}

    # BH
    bh_src = os.path.join(base_data_folder, "npz", "bh", bh_split)
    bh_dst = os.path.join(temp_dir, "bh", bh_split)
    os.makedirs(bh_dst, exist_ok=True)
    _slice_npz(os.path.join(bh_src, "train.npz"), os.path.join(bh_dst, "train.npz"), 64)
    _slice_npz(os.path.join(bh_src, "valid.npz"), os.path.join(bh_dst, "valid.npz"), 16)
    _slice_npz(os.path.join(bh_src, "test.npz"), os.path.join(bh_dst, "test.npz"), 16)
    splits["bh"] = bh_dst

    # Suzuki
    suz_src = os.path.join(base_data_folder, "processed", "suzuki", "npz", suzuki_split)
    suz_dst = os.path.join(temp_dir, "suzuki", suzuki_split)
    os.makedirs(suz_dst, exist_ok=True)
    _slice_npz(os.path.join(suz_src, "train.npz"), os.path.join(suz_dst, "train.npz"), 64)
    _slice_npz(os.path.join(suz_src, "valid.npz"), os.path.join(suz_dst, "valid.npz"), 16)
    _slice_npz(os.path.join(suz_src, "test.npz"), os.path.join(suz_dst, "test.npz"), 16)
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
        "seed": base_args.seed,
        "label": cell["label"],
        "reactant_tokens": cell["reactant_tokens"],
        "head": cell["head"],
        "attention_on": "reactants",
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
        "status": "failed",
        "error": "",
    }

    checkpoint_file = os.path.join(run_dir, "model.pt")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    run_opts = copy.copy(base_args)
    run_opts.reactant_tokens = cell["reactant_tokens"]
    run_opts.head = cell["head"]
    run_opts.attention_on = "reactants"
    run_opts.model_path = os.path.join(run_dir, "")
    run_opts.model_name = "model.pt"
    run_opts.Data_folder = os.path.dirname(os.path.abspath(npz_dir)) + os.sep
    run_opts.npz_folder = os.path.basename(os.path.abspath(npz_dir))

    os.makedirs(run_opts.monitor_folder, exist_ok=True)
    os.makedirs(run_opts.image_folder, exist_ok=True)

    set_seed(run_opts.seed)
    logger.info(
        "=== RUNNING: dataset=%s split=%s cell=%s (tokens=%s, head=%s) ==="
        % (dataset_name, split_name, cell["label"], cell["reactant_tokens"], cell["head"])
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
        train_metrics = train_res[0]

        history_path = os.path.join(run_opts.monitor_folder, "history.json")
        epochs_ran = 0
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                h = json.load(f)
                epochs_ran = len(h.get("epoch", []))
        if epochs_ran == 0:
            epochs_ran = run_opts.epochs

        train_runtime = result["train_runtime_sec"]
        sec_per_epoch = train_runtime / epochs_ran if epochs_ran > 0 else 0.0

        test_m = result["test_metrics"]
        row.update(
            {
                "status": "success",
                "train_r2": train_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_r2": test_m["r2"],
                "test_mae": test_m["mae"],
                "test_rmse": test_m["rmse"],
                "trainable_parameters": trainable_params,
                "epochs_ran": epochs_ran,
                "train_runtime_sec": train_runtime,
                "seconds_per_epoch": sec_per_epoch,
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
    group_cols = ["dataset", "label", "reactant_tokens", "head", "attention_on"]
    for keys, group in success.groupby(group_cols):
        entry = dict(zip(group_cols, keys))
        entry["n_successful"] = len(group)
        for m in METRICS:
            vals = pd.to_numeric(group[m], errors="coerce").dropna()
            entry[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            entry[f"{m}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def build_grid_parser() -> argparse.ArgumentParser:
    """Builds parser extending main_finetune defaults."""
    parser = build_parser()
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
    parser.set_defaults(split_ids=list(range(10)))
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
        choices=["all", "current_model", "approach_3", "approach_2", "mlp_control"],
        help="which experiment cells to run (default: 'all')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../logs/approach2_grid/",
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
    logger.info("Starting Approach 2 Grid Runner (smoke=%s, datasets=%s)" % (args.smoke, args.datasets))

    if not os.path.exists(os.path.join(args.Data_folder, "npz")) and os.path.isdir("Data"):
        args.Data_folder = "Data/"

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    per_split_csv = os.path.join(args.output_dir, "approach2_per_split.csv")
    summary_csv = os.path.join(args.output_dir, "approach2_summary.csv")

    active_cells = CELLS
    if "all" not in args.cells:
        active_cells = [c for c in CELLS if c["label"] in args.cells]

    temp_dir_obj = None
    try:
        if args.smoke:
            logger.info("Setting up smoke test configuration...")
            temp_dir_obj = tempfile.TemporaryDirectory()
            bh_split = "fullcv01_split70"
            suzuki_split = "split_0"
            smoke_splits = _prepare_smoke_data(
                args.Data_folder, bh_split, suzuki_split, temp_dir_obj.name
            )

            # Override parameters for fast smoke execution
            args.epochs = 1
            args.batch_size = 8
            args.num_workers = 0
            args.layer = 1
            args.emb_dim = 32

            eval_datasets = []
            if args.datasets in ("bh", "all"):
                eval_datasets.append(("bh", bh_split, smoke_splits["bh"], "rxn", "Output"))
            if args.datasets in ("suzuki", "all"):
                eval_datasets.append(("suzuki", suzuki_split, smoke_splits["suzuki"], "rxn", "y"))
        else:
            eval_datasets = []
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

        # Load existing results if resuming
        if os.path.isfile(per_split_csv):
            df_existing = pd.read_csv(per_split_csv)
            results = df_existing.to_dict("records")
        else:
            results = []

        for dataset_name, split_name, npz_dir, rxn_col, y_col in eval_datasets:
            dataset_args = copy.copy(args)
            dataset_args.reaction_column = rxn_col
            dataset_args.y_column = y_col

            for cell in active_cells:
                # Check if already successful
                already_done = any(
                    r.get("dataset") == dataset_name
                    and r.get("split") == split_name
                    and r.get("label") == cell["label"]
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

                # Update or append result
                existing_idx = next(
                    (
                        i
                        for i, r in enumerate(results)
                        if r.get("dataset") == dataset_name
                        and r.get("split") == split_name
                        and r.get("label") == cell["label"]
                    ),
                    None,
                )
                if existing_idx is not None:
                    results[existing_idx] = row
                else:
                    results.append(row)

                # Incremental CSV saving
                df_results = pd.DataFrame(results)
                df_results.to_csv(per_split_csv, index=False)

        if len(results):
            df_results = pd.DataFrame(results)
            summary_df = summarize_grid(df_results)
            summary_df.to_csv(summary_csv, index=False)
            logger.info("=== GRID COMPLETE ===")
            logger.info("Per-split CSV: %s" % per_split_csv)
            logger.info("Summary CSV:   %s" % summary_csv)
            print("\nSummary Results:\n", summary_df.to_string())

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
