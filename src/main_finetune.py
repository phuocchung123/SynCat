import argparse
import os
import random
import numpy as np
import torch
from prepare_data import prepare_data
# from joblib import Parallel, delayed
from utils import configure_warnings_and_logs

configure_warnings_and_logs(ignore_warnings=True, disable_rdkit_logs=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--device", type=int, default=1)
    arg_parser.add_argument("--monitor_folder", type=str, default="../Data/monitor/")
    arg_parser.add_argument("--monitor_name", type=str, default="monitor.txt")
    arg_parser.add_argument("--Data_folder", type=str, default="../Data/")
    arg_parser.add_argument("--data_csv", type=str, default="latest_data/schneider50k.csv")
    arg_parser.add_argument("--model_path", type=str, default="../Data/model/")
    arg_parser.add_argument("--model_name", type=str, default="model_sch_ba.pt")
    arg_parser.add_argument("--data_inference", type=str, default="data/inference.npy")
    arg_parser.add_argument("--npz_inference", type=str, default="npz/npz_inference")
    arg_parser.add_argument("--y_column", type=str, default="y")
    arg_parser.add_argument("--reaction_column", type=str, default="rxn")
    arg_parser.add_argument("--seed", type=int, default=27407)
    args = arg_parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    from finetune import finetune

    npz_folder = args.Data_folder + args.npz_inference + "/"
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for dirpath, dirnames, files in os.walk(npz_folder):
        if files:
            print("Already exist files in {}".format(dirpath))
        else:
            prepare_data(args)

    finetune(args)
