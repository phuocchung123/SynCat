import numpy as np
import pandas as pd
from reaction_data import get_graph_data
from utils import configure_warnings_and_logs
from sklearn.model_selection import train_test_split

configure_warnings_and_logs(ignore_warnings=True)


def prepare_data(args):
    data = pd.read_csv(args.Data_folder + args.data_csv)
    if args.reagent_option:
        data = data.fillna("")
    y = data[args.y_column]
    if args.train_test_split:
        data_pretrain = data[data[args.split_column] == "train"]
        data_test = data[data[args.split_column] == "test"]
    else:
        data_pretrain, data_test = train_test_split(
            data, test_size=0.1, stratify=y, random_state=42
        )
    data_train, data_valid = train_test_split(
        data_pretrain,
        test_size=0.1,
        stratify=data_pretrain[args.y_column],
        random_state=42,
    )

    rsmi_list = data[args.reaction_column].values
    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])

    # get_data_train
    rsmi_list_train = data_train[args.reaction_column].values
    y_list_train = data_train[args.y_column].values
    y_list_train = np.eye(y.nunique(), dtype="uint8")[y_list_train]
    filename_train = args.Data_folder + args.npz_folder + "/" + args.train_set

    # get_data_valid
    rsmi_list_valid = data_valid[args.reaction_column].values
    y_list_valid = data_valid[args.y_column].values
    y_list_valid = np.eye(y.nunique(), dtype="uint8")[y_list_valid]
    filename_valid = args.Data_folder + args.npz_folder + "/" + args.val_set

    # get_data_test
    rsmi_list_test = data_test[args.reaction_column].values
    y_list_test = data_test[args.y_column].values
    y_list_test = np.eye(y.nunique(), dtype="uint8")[y_list_test]
    filename_test = args.Data_folder + args.npz_folder + "/" + args.test_set

    if args.reagent_option:
        reagent = data[args.reagent_column].values
        reagent_max_cnt = np.max([smi.count(".") + 1 for smi in reagent])
        reagent_train = data_train[args.reagent_column].values
        reagent_valid = data_valid[args.reagent_column].values
        reagent_test = data_test[args.reagent_column].values

        get_graph_data(
            args,
            rsmi_list_train,
            y_list_train,
            filename_train,
            rmol_max_cnt,
            pmol_max_cnt,
            reagent_train,
            reagent_max_cnt,
        )
        get_graph_data(
            args,
            rsmi_list_valid,
            y_list_valid,
            filename_valid,
            rmol_max_cnt,
            pmol_max_cnt,
            reagent_valid,
            reagent_max_cnt,
        )
        get_graph_data(
            args,
            rsmi_list_test,
            y_list_test,
            filename_test,
            rmol_max_cnt,
            pmol_max_cnt,
            reagent_test,
            reagent_max_cnt,
        )
    else:
        get_graph_data(
            args,
            rsmi_list_train,
            y_list_train,
            filename_train,
            rmol_max_cnt,
            pmol_max_cnt,
        )
        get_graph_data(
            args,
            rsmi_list_valid,
            y_list_valid,
            filename_valid,
            rmol_max_cnt,
            pmol_max_cnt,
        )
        get_graph_data(
            args, rsmi_list_test, y_list_test, filename_test, rmol_max_cnt, pmol_max_cnt
        )
