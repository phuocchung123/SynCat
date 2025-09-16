import argparse
import os
import random
import numpy as np
import torch
from prepare_data import prepare_data
from finetune import finetune
from utils import configure_warnings_and_logs, setup_logging


configure_warnings_and_logs(ignore_warnings=True, disable_rdkit_logs=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--device", type=int, default=1)
    arg_parser.add_argument("--layer", type=int, default=2)
    arg_parser.add_argument("--emb_dim", type=int, default=256)
    arg_parser.add_argument("--dropout", type=float, default=0.1)
    arg_parser.add_argument("--lr", type=float, default=1e-3)
    arg_parser.add_argument("--weight_decay", type=float, default=1e-4)
    arg_parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    arg_parser.add_argument("--Data_folder", type=str, default="../Data/")
    arg_parser.add_argument(
        "--data_csv", type=str, default="raw/schneider50k_unbalanced.csv.gz"
    )
    arg_parser.add_argument("--model_path", type=str, default="../Data/model/")
    arg_parser.add_argument("--model_name", type=str, default="model_k.pt")
    arg_parser.add_argument("--npz_folder", type=str, default="npz/npz_sch")
    arg_parser.add_argument("--y_column", type=str, default="y")
    arg_parser.add_argument("--train_test_split", type=bool, default=True)
    arg_parser.add_argument("--split_column", type=str, default="split")
    arg_parser.add_argument("--reaction_column", type=str, default="rxn")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()

    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    npz_folder = args.Data_folder + args.npz_folder + "/"
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for dirpath, dirnames, files in os.walk(npz_folder):
        if files:
            logger.info("Already exist files in {}".format(dirpath))
        else:
            prepare_data(args)

    finetune(args)
