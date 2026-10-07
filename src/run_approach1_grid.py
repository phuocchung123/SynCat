"""
Standalone experiment runner for the full Approach 1 ablation grid.

Runs the 16-cell comparison over
    reactant_tokens   {ind, comb}
    x attention_on    {reactants, none}
    x reactant_pooling {mean, rn}
    x head            {linear, mlp}
on Buchwald-Hartwig (10 FullCV 1-10 split_70 and the 4 out-of-sample Test1-4
sets) and Suzuki-Miyaura (split_0..split_9), always with the same seed and
hyperparameters. Cells are labelled "<tokens>-<attention>-<pooling>-<head>" and
grouped into named approaches:

    baseline   ind/reactants/mean/linear
    control    ind/none/mean/{linear,mlp}
    A          ind/none/rn/{linear,mlp}
    B          ind/reactants/rn/{linear,mlp}
    approach_1 comb/reactants/rn/{linear,mlp}
    approach_2 comb/reactants/mean/mlp
    approach_3 comb/reactants/mean/linear
    unlabeled  the remaining cells (still run)

Supports both fast smoke testing and full cluster execution with resumption.
"""

import argparse
import copy
import gc
import itertools
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
from main_finetune import build_parser
from model import HEADS, REACTANT_POOLINGS, REACTANT_TOKENS, model
from multi_gpu import resolve_gpu_ids
from utils import collate_reaction_graphs, set_seed, setup_logging
from validation import validation

# Named approaches of the grid, keyed by (tokens, attention, pooling, head).
# Any combination not listed here is "unlabeled".
APPROACHES = {
    ("ind", "reactants", "mean", "linear"): "baseline",
    ("ind", "none", "mean", "linear"): "control",
    ("ind", "none", "mean", "mlp"): "control",
    ("ind", "none", "rn", "linear"): "A",
    ("ind", "none", "rn", "mlp"): "A",
    ("ind", "reactants", "rn", "linear"): "B",
    ("ind", "reactants", "rn", "mlp"): "B",
    ("comb", "reactants", "rn", "linear"): "approach_1",
    ("comb", "reactants", "rn", "mlp"): "approach_1",
    ("comb", "reactants", "mean", "mlp"): "approach_2",
    ("comb", "reactants", "mean", "linear"): "approach_3",
}


def _build_cells() -> List[dict]:
    """Builds the 16 cells of the grid from the model constants."""
    cells = []
    for tokens, attention_on, pooling, head in itertools.product(
        REACTANT_TOKENS, ("reactants", "none"), REACTANT_POOLINGS, HEADS
    ):
        cells.append(
            {
                "label": "%s-%s-%s-%s" % (tokens, attention_on, pooling, head),
                "approach": APPROACHES.get(
                    (tokens, attention_on, pooling, head), "unlabeled"
                ),
                "reactant_tokens": tokens,
                "attention_on": attention_on,
                "reactant_pooling": pooling,
                "head": head,
            }
        )
    return cells


CELLS = _build_cells()

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
    "peak_gpu_memory_mb",
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
    datasets: str = "all",
) -> Dict[str, str]:
    """Slices small smoke datasets for BH (CV split and test1) and Suzuki."""
    splits = {}

    if datasets in ("bh", "all"):
        for key, split in (("bh", bh_split), ("bh_test", "test1")):
            src_dir = os.path.join(base_data_folder, "npz", "bh", split)
            dst_dir = os.path.join(temp_dir, "bh", split)
            os.makedirs(dst_dir, exist_ok=True)
            _slice_npz(
                os.path.join(src_dir, "train.npz"),
                os.path.join(dst_dir, "train.npz"),
                64,
            )
            _slice_npz(
                os.path.join(src_dir, "valid.npz"),
                os.path.join(dst_dir, "valid.npz"),
                32,
            )
            _slice_npz(
                os.path.join(src_dir, "test.npz"),
                os.path.join(dst_dir, "test.npz"),
                32,
            )
            splits[key] = dst_dir

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
    split_kind: str,
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
        "split_kind": split_kind,
        "cell": cell["label"],
        "approach": cell.get("approach", ""),
        "reactant_tokens": cell["reactant_tokens"],
        "attention_on": cell["attention_on"],
        "reactant_pooling": cell["reactant_pooling"],
        "head": cell["head"],
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
        "peak_gpu_memory_mb": None,
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
    run_opts.attention_on = cell["attention_on"]
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
        "=== RUNNING: dataset=%s split=%s kind=%s cell=%s "
        "(tokens=%s, attention_on=%s, pooling=%s, head=%s) ==="
        % (
            dataset_name,
            split_name,
            split_kind,
            cell["label"],
            cell["reactant_tokens"],
            cell["attention_on"],
            cell["reactant_pooling"],
            cell["head"],
        )
    )

    if device.type == "cuda":
        # PyTorch <= 2.0 only knows a device's peak stats once the caching
        # allocator has been initialised for it; a tiny allocation does that
        # (otherwise reset_peak_memory_stats raises "did you call init?").
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    try:
        try:
            result = finetune(run_opts, save_embedding=False)
        finally:
            # finetune() reconfigures the root logger to the cell's monitor.log;
            # point it back at the grid log so later cells keep logging here.
            setup_logging(
                log_filename=os.path.join(base_args.output_dir, "grid.log")
            )

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

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        )

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
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
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
    """Group by dataset, split kind and cell to produce mean and sample std (ddof=1)."""
    success = df[df["status"] == "success"]
    rows = []
    group_cols = [
        "dataset",
        "split_kind",
        "cell",
        "approach",
        "reactant_tokens",
        "attention_on",
        "reactant_pooling",
        "head",
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
) -> List[Tuple[str, str, str, str, str, str]]:
    """Builds (dataset, split, split_kind, npz_dir, rxn_col, y_col) tuples.

    split_kind is "test" for the BH out-of-sample Test<id> folders and "cv"
    for the BH FullCV folders and all Suzuki splits.
    """
    eval_datasets = []
    if args.smoke and temp_splits is not None:
        if args.datasets in ("bh", "all") and "bh" in temp_splits:
            eval_datasets.append(
                ("bh", "fullcv01_split70", "cv", temp_splits["bh"], "rxn", "Output")
            )
        if args.datasets in ("bh", "all") and "bh_test" in temp_splits:
            eval_datasets.append(
                ("bh", "test1", "test", temp_splits["bh_test"], "rxn", "Output")
            )
        if args.datasets in ("suzuki", "all") and "suzuki" in temp_splits:
            eval_datasets.append(
                ("suzuki", "split_0", "cv", temp_splits["suzuki"], "rxn", "y")
            )
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
            split_kind = "test" if s.startswith("test") else "cv"
            eval_datasets.append(("bh", s, split_kind, npz_dir, "rxn", "Output"))

    if args.datasets in ("suzuki", "all"):
        if args.suzuki_splits is not None:
            suzuki_splits_to_run = args.suzuki_splits
        else:
            suzuki_splits_to_run = ["split_%d" % i for i in args.split_ids]
        for s in suzuki_splits_to_run:
            npz_dir = os.path.join(args.Data_folder, "processed", "suzuki", "npz", s)
            eval_datasets.append(("suzuki", s, "cv", npz_dir, "rxn", "y"))

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
    """Builds the grid parser on top of the shared finetuning options."""
    parser = build_parser()
    parser.set_defaults(split_ids=list(range(10)))

    # Grid evaluation specific arguments
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run fast smoke test on tiny subsets (1 BH CV split, BH test1, "
        "1 Suzuki split, 2 epochs)",
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
        help="which cells to run: 'all', any cell label "
        "(e.g. comb-reactants-rn-mlp), or any approach name (e.g. approach_1)",
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


def _select_cells(selected: List[str]) -> List[dict]:
    """Resolves --cells (all / labels / approach names) to grid cells."""
    if "all" in selected:
        return list(CELLS)
    wanted = set(selected)
    chosen = [c for c in CELLS if c["label"] in wanted or c["approach"] in wanted]
    if not chosen:
        raise ValueError("no grid cell matches --cells %s" % selected)
    return chosen


def main():
    parser = build_grid_parser()
    args = parser.parse_args()

    if getattr(args, "log_dir", None) and args.log_dir != "../logs/suzuki_regression/":
        args.output_dir = args.log_dir

    if len(resolve_gpu_ids(args)) > 1:
        parser.error(
            "the grid runner runs on a single GPU only; DistributedDataParallel "
            "would hide the per-cell peak GPU memory"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(log_filename=os.path.join(args.output_dir, "grid.log"))
    logger.info("Starting Approach 1 Grid Runner (smoke=%s, datasets=%s)" % (args.smoke, args.datasets))

    args.Data_folder = _resolve_data_folder(args.Data_folder)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    per_split_csv = os.path.join(args.output_dir, "approach1_per_split.csv")
    summary_csv = os.path.join(args.output_dir, "approach1_summary.csv")

    active_cells = _select_cells(args.cells)

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
            args.epochs = 2
            args.batch_size = 8
            args.num_workers = 0
            args.layer = 1
            args.emb_dim = 32

        eval_datasets = _resolve_eval_datasets(args, temp_splits)
        results = _load_existing_results(per_split_csv)

        for dataset_name, split_name, split_kind, npz_dir, rxn_col, y_col in eval_datasets:
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
                    split_kind,
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
