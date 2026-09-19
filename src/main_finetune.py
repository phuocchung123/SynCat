import argparse
import os
import sys
from prepare_data import prepare_data, SPLIT_STRATEGIES
from finetune import finetune
from utils import configure_warnings_and_logs, set_seed, setup_logging

configure_warnings_and_logs(ignore_warnings=True, disable_rdkit_logs=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--device", type=int, default=0)
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (0 loads batches in the main process)",
    )
    arg_parser.add_argument(
        "--layer", type=int, default=3, help="number of GNN (GIN) layers"
    )
    arg_parser.add_argument(
        "--attention_layer",
        type=int,
        default=1,
        help="number of self-attention layers over the compounds of a reaction",
    )
    arg_parser.add_argument("--emb_dim", type=int, default=384)
    arg_parser.add_argument("--dropout", type=float, default=0.1)
    arg_parser.add_argument("--lr", type=float, default=1e-3)
    arg_parser.add_argument("--weight_decay", type=float, default=1e-4)
    arg_parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    arg_parser.add_argument("--image_folder", type=str, default="../Image/")
    arg_parser.add_argument("--Data_folder", type=str, default="../Data/")
    arg_parser.add_argument(
        "--data_csv", type=str, default="raw/suzuki/random_split_0.tsv"
    )
    arg_parser.add_argument("--model_path", type=str, default="../Data/model/")
    arg_parser.add_argument("--model_name", type=str, default="model_yield.pt")
    arg_parser.add_argument("--npz_folder", type=str, default="npz/npz_yield")
    arg_parser.add_argument("--y_column", type=str, default="y")
    arg_parser.add_argument("--train_test_split", type=bool, default=False)
    arg_parser.add_argument("--split_column", type=str, default="split")
    arg_parser.add_argument("--reaction_column", type=str, default="rxn")
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed; with --stage, the seed of split <id> is seed + <id>",
    )
    # split ratios: test = test_ratio of the data, valid = valid_ratio of the rest
    arg_parser.add_argument("--test_ratio", type=float, default=0.3)
    arg_parser.add_argument("--valid_ratio", type=float, default=0.1)
    arg_parser.add_argument(
        "--split_strategy",
        type=str,
        choices=SPLIT_STRATEGIES,
        default=None,
        help="'ordered': the last test_ratio rows of the file form the test set; "
        "'shuffle': seeded random test split. Default: 'ordered' with --stage "
        "(each random_split_<id>.tsv is already a random permutation), 'shuffle' otherwise",
    )
    arg_parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early stopping patience on validation loss (0 disables)",
    )
    arg_parser.add_argument(
        "--track_test_each_epoch",
        action="store_true",
        help="also score the test set after every epoch (monitoring only)",
    )
    # multi-split Suzuki pipeline (prepare -> validate -> train)
    arg_parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "validate", "train", "all"],
        default=None,
        help="run a stage of the multi-split pipeline instead of the single-file run",
    )
    arg_parser.add_argument("--split_ids", type=int, nargs="+", default=[9])
    arg_parser.add_argument("--raw_split_dir", type=str, default="raw/suzuki")
    arg_parser.add_argument(
        "--split_file_pattern", type=str, default="random_split_{split_id}.tsv"
    )
    arg_parser.add_argument(
        "--processed_npz_dir", type=str, default="processed/suzuki/npz"
    )
    arg_parser.add_argument("--log_dir", type=str, default="../logs/suzuki_regression/")
    arg_parser.add_argument(
        "--ratio_tolerance",
        type=float,
        default=0.01,
        help="max allowed deviation of effective subset ratios from the expected ones",
    )
    arg_parser.add_argument(
        "--drop_invalid_rows",
        action="store_true",
        help="drop rows with missing/invalid reactions or targets instead of failing",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate prepared split data even if it is valid",
    )
    arg_parser.add_argument(
        "--skip_aggregate",
        action="store_true",
        help="only record per-split test results; do not compute mean/std across splits",
    )
    arg_parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="replace existing experiment result files",
    )
    args = arg_parser.parse_args()

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

    finetune(args)
