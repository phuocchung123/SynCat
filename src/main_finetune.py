import argparse
import os
import sys
from prepare_data import prepare_data, SPLIT_STRATEGIES
from finetune import finetune
from accelerator import ACCELERATORS
from model import ATTENTION_TARGETS, REACTION_COMBINE_DIMS
from utils import configure_warnings_and_logs, set_seed, setup_logging

configure_warnings_and_logs(ignore_warnings=True, disable_rdkit_logs=True)


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser of the pipeline.

    It is exposed so that other entry points (e.g. `train_bh.py`) can reuse the
    same data, model and training options instead of redeclaring them.

    Returns
    -------
    argparse.ArgumentParser
        The parser, without the arguments parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=list(ACCELERATORS),
        help="device to train on: 'auto' takes a GPU if CUDA is available, then a "
        "TPU if torch_xla finds one, then the CPU; 'gpu', 'tpu' or 'cpu' forces "
        "one and fails if it is missing",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="+",
        default=None,
        help="GPU ids to train on, e.g. '--gpus 0 1' or '--gpus all'; more than one "
        "runs DistributedDataParallel (--batch_size stays the total batch, split "
        "evenly across GPUs). Default: the single GPU given by --device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (0 loads batches in the main process)",
    )
    parser.add_argument(
        "--layer", type=int, default=3, help="number of GNN (GIN) layers"
    )
    parser.add_argument(
        "--attention_layer",
        type=int,
        default=1,
        help="number of self-attention layers over the compounds of a reaction",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="number of heads per self-attention layer; must divide --emb_dim "
        "(1 = single-head attention)",
    )
    parser.add_argument(
        "--attention_on",
        type=str,
        default="reactants",
        choices=list(ATTENTION_TARGETS),
        help="which compounds attend to each other: 'reactants' (left of '>>'), "
        "'products', 'both' (each side separately, shared weights), 'all' (every "
        "compound in one shared attention) or 'none' (no attention; each side is "
        "the plain mean of its GNN embeddings)",
    )
    parser.add_argument(
        "--reaction_combine",
        type=str,
        default="concat",
        choices=sorted(REACTION_COMBINE_DIMS),
        help="how the reactant vector r and product vector p form the reaction "
        "vector: concat [r, p]; sum r + p; sub p - r; mul r * p; "
        "concat_sub [r, p, p - r]; interaction [r, p, |p - r|, r * p]",
    )
    parser.add_argument("--emb_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    parser.add_argument("--image_folder", type=str, default="../Image/")
    parser.add_argument("--Data_folder", type=str, default="../Data/")
    parser.add_argument(
        "--data_csv", type=str, default="raw/suzuki/random_split_0.tsv"
    )
    parser.add_argument("--model_path", type=str, default="../Data/model/")
    parser.add_argument("--model_name", type=str, default="model_yield.pt")
    parser.add_argument("--npz_folder", type=str, default="npz/npz_yield")
    parser.add_argument("--y_column", type=str, default="y")
    parser.add_argument(
        "--train_test_split",
        action="store_true",
        help="take the train/test split from --split_column instead of splitting "
        "the table here (the uspto_yields_* datasets ship such a column)",
    )
    parser.add_argument("--split_column", type=str, default="split")
    parser.add_argument("--reaction_column", type=str, default="rxn")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed; with --stage, the seed of split <id> is seed + <id>",
    )
    # split ratios: test = test_ratio of the data, valid = valid_ratio of the rest
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=SPLIT_STRATEGIES,
        default=None,
        help="'ordered': the last test_ratio rows of the file form the test set; "
        "'shuffle': seeded random test split. Default: 'ordered' with --stage "
        "(each random_split_<id>.tsv is already a random permutation), 'shuffle' otherwise",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early stopping patience on validation loss (0 disables)",
    )
    parser.add_argument(
        "--track_test_each_epoch",
        action="store_true",
        help="also score the test set after every epoch (monitoring only)",
    )
    # multi-split Suzuki pipeline (prepare -> validate -> train)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "validate", "train", "all"],
        default=None,
        help="run a stage of the multi-split pipeline instead of the single-file run",
    )
    parser.add_argument("--split_ids", type=int, nargs="+", default=[9])
    parser.add_argument("--raw_split_dir", type=str, default="raw/suzuki")
    parser.add_argument(
        "--split_file_pattern", type=str, default="random_split_{split_id}.tsv"
    )
    parser.add_argument(
        "--processed_npz_dir", type=str, default="processed/suzuki/npz"
    )
    parser.add_argument("--log_dir", type=str, default="../logs/suzuki_regression/")
    parser.add_argument(
        "--ratio_tolerance",
        type=float,
        default=0.01,
        help="max allowed deviation of effective subset ratios from the expected ones",
    )
    parser.add_argument(
        "--drop_invalid_rows",
        action="store_true",
        help="drop rows with missing/invalid reactions or targets instead of failing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate prepared split data even if it is valid",
    )
    parser.add_argument(
        "--skip_aggregate",
        action="store_true",
        help="only record per-split test results; do not compute mean/std across splits",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="write the train/valid/test npz files and stop, without training",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="replace existing experiment result files",
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    if args.num_heads < 1 or args.emb_dim % args.num_heads != 0:
        arg_parser.error(
            "--num_heads (%d) must be a positive divisor of --emb_dim (%d)"
            % (args.num_heads, args.emb_dim)
        )

    if args.prepare_only and args.stage is not None:
        arg_parser.error("--prepare_only cannot be combined with --stage")

    if args.stage is not None:
        from suzuki_splits import run_stage

        sys.exit(run_stage(args))

    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")

    set_seed(args.seed)

    npz_folder = args.Data_folder + args.npz_folder + "/"
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for dirpath, dirnames, files in os.walk(npz_folder):
        if files:
            logger.info("Already exist files in {}".format(dirpath))
        else:
            prepare_data(args)

    if args.prepare_only:
        logger.info("--- prepared %s; stopping before training" % npz_folder)
        sys.exit(0)

    finetune(args)
