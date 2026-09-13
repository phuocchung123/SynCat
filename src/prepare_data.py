import numpy as np
import pandas as pd
from reaction_data import get_graph_data
from utils import configure_warnings_and_logs, read_reaction_table, setup_logging
from sklearn.model_selection import train_test_split

configure_warnings_and_logs(ignore_warnings=True)


def prepare_data(args) -> None:
    """
    Prepare and split chemical reaction-yield data, then save reaction information to npz files.

    If `args.train_test_split` is True, the train/test split is taken from
    `args.split_column`. Otherwise, a deterministic split is created with
    `sklearn.model_selection.train_test_split` using `args.seed`, giving a
    train/valid/test ratio of 81/9/10 (test_size=0.1 for the test split, then
    test_size=0.1 again on the remaining 90% for the validation split).

    Rows with a missing/non-numeric reaction or yield value are dropped and the
    number of excluded rows is logged.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing dataset and configuration parameters.

    Returns
    -------
    None
    """
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")
    data = read_reaction_table(args.Data_folder + args.data_csv)

    n_before = len(data)
    data[args.y_column] = pd.to_numeric(data[args.y_column], errors="coerce")
    data = data.dropna(subset=[args.reaction_column, args.y_column])
    data = data[data[args.reaction_column].str.contains(">>", na=False)]
    n_excluded = n_before - len(data)
    logger.info(
        "--- excluded %d/%d samples with missing/invalid reaction or yield values"
        % (n_excluded, n_before)
    )

    if args.train_test_split:
        data_pretrain = data[data[args.split_column] == "train"]
        data_test = data[data[args.split_column] == "test"]
    else:
        data_pretrain, data_test = train_test_split(
            data, test_size=0.1, random_state=args.seed
        )
    data_train, data_valid = train_test_split(
        data_pretrain,
        test_size=0.1,
        random_state=args.seed,
    )

    rsmi_list = data[args.reaction_column].values
    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])

    # get_data_train
    rsmi_list_train = data_train[args.reaction_column].values
    y_list_train = data_train[args.y_column].values
    filename_train = args.Data_folder + args.npz_folder + "/" + "train.npz"

    # get_data_valid
    rsmi_list_valid = data_valid[args.reaction_column].values
    y_list_valid = data_valid[args.y_column].values
    filename_valid = args.Data_folder + args.npz_folder + "/" + "valid.npz"

    # get_data_test
    rsmi_list_test = data_test[args.reaction_column].values
    y_list_test = data_test[args.y_column].values
    filename_test = args.Data_folder + args.npz_folder + "/" + "test.npz"

    logger.info(
        "--- train/valid/test sizes: %d/%d/%d"
        % (len(data_train), len(data_valid), len(data_test))
    )

    get_graph_data(
        rsmi_list_train,
        rmol_max_cnt,
        pmol_max_cnt,
        args,
        filename_train,
        y_list_train,
    )
    get_graph_data(
        rsmi_list_valid,
        rmol_max_cnt,
        pmol_max_cnt,
        args,
        filename_valid,
        y_list_valid,
    )
    get_graph_data(
        rsmi_list_test,
        rmol_max_cnt,
        pmol_max_cnt,
        args,
        filename_test,
        y_list_test,
    )
