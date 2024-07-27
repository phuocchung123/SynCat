import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from rdkit import rdBase
import warnings
from finetune import finetune
from prepare_data import prepare_data

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings('ignore')

if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=8)
    arg_parser.add_argument('--epochs', type=int, default=20)
    arg_parser.add_argument('--device', type=int, default=0)
    arg_parser.add_argument('--monitor_folder', type=str, default='../Data/monitor/')
    arg_parser.add_argument('--monitor_name', type=str, default='monitor.txt')
    arg_parser.add_argument('--Data_folder', type=str, default='../Data/')
    arg_parser.add_argument('--data_csv', type=str, default='trial_class.csv')
    arg_parser.add_argument('--mapped_data_csv', type=str, default='mapped_data.csv')
    arg_parser.add_argument('--train_set', type=str, default='data_train_trial.npz')
    arg_parser.add_argument('--val_set', type=str, default='data_valid_trial.npz')
    arg_parser.add_argument('--test_set', type=str, default='data_test_trial.npz')
    arg_parser.add_argument('--model_path', type=str, default='../Data/model/')
    arg_parser.add_argument('--model_name', type=str, default='model.pt')
    arg_parser.add_argument('--npz_folder', type=str, default='../Data/npz/')
    arg_parser.add_argument('--reagent_option', type=bool, default=False)
    arg_parser.add_argument('--y_column', type=str, default='class')
    arg_parser.add_argument('--train_test_split', type=bool, default=False)
    arg_parser.add_argument('--split_column', type=str, default='split')
    arg_parser.add_argument('--reaction_column', type=str, default='reactions')
    arg_parser.add_argument('--reagent_column', type=str, default='separated_reagent')
    arg_parser.add_argument('--mapped_reaction_column', type=str, default='mapped_reactions')
    arg_parser.add_argument("--seed", type=int, default=27407)
    args = arg_parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.npz_folder):
        os.makedirs(args.npz_folder)
    for dirpath, dirnames, files in os.walk(args.npz_folder):
        if files:
            print('Already exist files in {}'.format(dirpath))
        else:
            prepare_data(args)

    finetune(args)