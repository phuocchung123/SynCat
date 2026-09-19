import numpy as np
import pandas as pd
from typing import Tuple
from reaction_data import get_graph_data
from utils import configure_warnings_and_logs, read_reaction_table, setup_logging
from sklearn.model_selection import train_test_split

configure_warnings_and_logs(ignore_warnings=True)


SPLIT_STRATEGIES = ("ordered", "shuffle")


def clean_reaction_data(
    data: pd.DataFrame, reaction_column: str, y_column: str
) -> Tuple[pd.DataFrame, int]:
    """
    Validates the reaction/target columns and removes unusable rows.

    A row is unusable when its reaction is missing or lacks ">>", or when its
    yield is missing, non-numeric, or non-finite.

    Parameters
    ----------
    data : pd.DataFrame
        Raw reaction table.
    reaction_column : str
        Name of the reaction SMILES column.
    y_column : str
        Name of the reaction-yield target column.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        The cleaned table (yield column coerced to float) and the number of
        excluded rows.

    Raises
    ------
    ValueError
        If `reaction_column` or `y_column` is not present in the table.
    """
    missing = [c for c in (reaction_column, y_column) if c not in data.columns]
    if missing:
        raise ValueError(
            "Required column(s) %s not found; available columns: %s"
            % (missing, list(data.columns))
        )

    n_before = len(data)
    data = data.copy()
    data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
    data = data[np.isfinite(data[y_column])]
    data = data.dropna(subset=[reaction_column])
    data = data[data[reaction_column].astype(str).str.contains(">>", na=False)]
    return data, n_before - len(data)


def split_reaction_data(
    data: pd.DataFrame,
    test_ratio: float,
    valid_ratio: float,
    seed: int,
    strategy: str = "shuffle",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministically splits a reaction table into train/valid/test subsets.

    The table is first divided into train1 (1 - `test_ratio`) and test
    (`test_ratio`); train1 is then divided into train (1 - `valid_ratio`) and
    valid (`valid_ratio`) with `sklearn.model_selection.train_test_split`.

    With `strategy="ordered"` the first division keeps the row order of the
    table (the last ceil(`test_ratio` * n) rows form the test set), which
    preserves the identity of pre-shuffled split files such as
    `random_split_<id>.tsv`. With `strategy="shuffle"` both divisions are
    shuffled with `seed`.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned reaction table.
    test_ratio : float
        Fraction of the full table used for the test set.
    valid_ratio : float
        Fraction of train1 used for the validation set.
    seed : int
        Random seed for the shuffled divisions.
    strategy : str, optional
        "ordered" or "shuffle" (default is "shuffle").

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The train, valid and test subsets.
    """
    for name, ratio in (("test_ratio", test_ratio), ("valid_ratio", valid_ratio)):
        if not 0.0 < ratio < 1.0:
            raise ValueError("%s must be in (0, 1), got %s" % (name, ratio))
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            "Unknown split strategy %r; expected one of %s"
            % (strategy, SPLIT_STRATEGIES)
        )

    if strategy == "ordered":
        n_test = int(np.ceil(test_ratio * len(data)))
        data_pretrain = data.iloc[: len(data) - n_test]
        data_test = data.iloc[len(data) - n_test:]
    else:
        data_pretrain, data_test = train_test_split(
            data, test_size=test_ratio, random_state=seed
        )
    data_train, data_valid = train_test_split(
        data_pretrain, test_size=valid_ratio, random_state=seed
    )
    return data_train, data_valid, data_test


def prepare_data(args) -> None:
    """
    Prepare and split chemical reaction-yield data, then save reaction information to npz files.

    If `args.train_test_split` is True, the train/test split is taken from
    `args.split_column` (as in the `uspto_yields_*` datasets, whose "split"
    column holds "train"/"test"); the validation set is then carved out of the
    train rows with `args.valid_ratio`. Otherwise, a deterministic split is created with
    `split_reaction_data` using `args.seed`: `args.test_ratio` of the data forms
    the test set and `args.valid_ratio` of the remainder forms the validation
    set (0.3 and 0.1 by default, i.e. a 63/7/30 train/valid/test ratio).

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
    data, n_excluded = clean_reaction_data(data, args.reaction_column, args.y_column)
    logger.info(
        "--- excluded %d/%d samples with missing/invalid reaction or yield values"
        % (n_excluded, n_before)
    )

    if args.train_test_split:
        if args.split_column not in data.columns:
            raise ValueError(
                "--train_test_split needs the column %r; available columns: %s"
                % (args.split_column, list(data.columns))
            )
        data_pretrain = data[data[args.split_column] == "train"]
        data_test = data[data[args.split_column] == "test"]
        if data_pretrain.empty or data_test.empty:
            raise ValueError(
                "Column %r must hold 'train' and 'test' rows; found %s"
                % (args.split_column, data[args.split_column].value_counts().to_dict())
            )
        data_train, data_valid = train_test_split(
            data_pretrain,
            test_size=args.valid_ratio,
            random_state=args.seed,
        )
    else:
        data_train, data_valid, data_test = split_reaction_data(
            data,
            args.test_ratio,
            args.valid_ratio,
            args.seed,
            args.split_strategy or "shuffle",
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
