import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def generate_scaffold(smiles, include_chirality=False):
    """
    Generates a Bemis-Murcko scaffold representation from a SMILES string. The Bemis-Murcko
    scaffold is a way to represent the core structure of a molecule, excluding side chains.

    Args:
    smiles (str): A SMILES string representing the molecule.
    include_chirality (bool): If True, the chiral information will be included in the scaffold.
                              Defaults to False.

    Returns:
    str: A SMILES string representing the Bemis-Murcko scaffold of the input molecule.
    """
    # Generate the Bemis-Murcko scaffold from the provided SMILES string, considering chirality if specified
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )

    return scaffold


def scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    return_smiles=True,
):
    """
    Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Splits a dataset into training, validation, and test sets based on Bemis-Murcko scaffolds.
    This deterministic splitting method ensures that molecules with the same scaffold are
    grouped together. Optionally, the function can filter out examples that contain a null
    value for a specified task before performing the split.

    Args:
    dataset (PyG Dataset): A PyTorch Geometric dataset object to be split.
    smiles_list (list of str): A list of SMILES strings corresponding to each entry in the dataset.
    task_idx (int, optional): The index of the task in the data.y tensor. If provided, the function
                              filters out examples where the specified task column in data.y has the
                              null value before splitting. Defaults to None, which means no filtering.
    null_value (float): The value considered as null in the data.y tensor. Used for filtering when
                        task_idx is provided. Defaults to 0.
    frac_train (float): The fraction of the dataset to include in the training set. Defaults to 0.8.
    frac_valid (float): The fraction of the dataset to include in the validation set. Defaults to 0.1.
    frac_test (float): The fraction of the dataset to include in the test set. Defaults to 0.1.
    return_smiles (bool): If True, the function also returns lists of SMILES strings for the training,
                          validation, and test sets. Defaults to True.

    Returns:
    tuple: Depending on the value of return_smiles, the function returns:
           - If return_smiles is False: Three PyG dataset objects representing the training, validation,
             and test sets.
           - If return_smiles is True: Three PyG dataset objects as above, plus a tuple containing three
             lists of SMILES strings (one for each set).
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        # scaffold = generate_scaffold(smiles)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    print("train set", len(train_idx))
    print("valid set", len(valid_idx))
    print("test set", len(test_idx))

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            (train_smiles, valid_smiles, test_smiles),
        )


def random_split(
    dataset,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42,
    smiles_list=None,
):
    """
    Randomly splits a dataset into training, validation, and test sets. The function
    allows for optional filtering of examples based on a specified task index and null value.
    If a list of SMILES strings is provided, it returns the corresponding SMILES for each
    subset along with the dataset objects.

    Args:
    dataset (PyG Dataset): The dataset to be split.
    task_idx (int, optional): Index of the task in the data.y tensor. If provided, examples with a null
                              value in this task column are filtered out before splitting. Defaults to None.
    null_value (float): The value considered as null in the data.y tensor. Used for filtering when
                        task_idx is provided. Defaults to 0.
    frac_train (float): Fraction of the dataset to include in the training set. Defaults to 0.8.
    frac_valid (float): Fraction of the dataset to include in the validation set. Defaults to 0.1.
    frac_test (float): Fraction of the dataset to include in the test set. Defaults to 0.1.
    seed (int): Random seed for reproducibility. Defaults to 42.
    smiles_list (list of str, optional): List of SMILES strings corresponding to the dataset. If provided,
                                          the function also returns SMILES for each subset.

    Returns:
    tuple: If smiles_list is not provided, returns (train_dataset, valid_dataset, test_dataset).
           If smiles_list is provided, also returns (train_smiles, valid_smiles, test_smiles) for
           the corresponding subsets.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = (
            y_task != null_value
        )  # boolean array that correspond to non null values
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]  # examples containing non
        # null labels in the specified task_idx
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[: int(frac_train * num_mols)]
    valid_idx = all_idx[
        int(frac_train * num_mols) : int(frac_valid * num_mols)
        + int(frac_train * num_mols)
    ]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols) :]

    print("train set", len(train_idx))
    print("valid set", len(valid_idx))
    print("test set", len(test_idx))

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not smiles_list:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            (train_smiles, valid_smiles, test_smiles),
        )


def random_scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Splits a dataset into training, validation, and test sets based on randomized Bemis-Murcko scaffolds.
    This ensures that molecules with similar scaffolds are grouped together, providing a more robust
    evaluation of model generalization. This function also supports filtering out data points based
    on a specified task index and null value.

    Args:
    dataset (PyG Dataset): The PyTorch Geometric dataset object to be split.
    smiles_list (list of str): A list of SMILES strings corresponding to the entries in the dataset.
    task_idx (int, optional): Index of the task in the data.y tensor. If provided, examples with a null
                              value in this task column are filtered out before splitting. Defaults to None.
    null_value (float): The value considered as null in the data.y tensor. Used for filtering when
                        task_idx is provided. Defaults to 0.
    frac_train (float): Fraction of the dataset to include in the training set. Defaults to 0.8.
    frac_valid (float): Fraction of the dataset to include in the validation set. Defaults to 0.1.
    frac_test (float): Fraction of the dataset to include in the test set. Defaults to 0.1.
    seed (int): Seed for the random number generator used in shuffling the scaffolds. Defaults to 0.

    Returns:
    tuple: Three PyTorch Geometric dataset objects representing the training, validation,
           and test sets, in that order.
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset
