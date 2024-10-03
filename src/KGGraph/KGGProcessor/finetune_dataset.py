import sys
from pathlib import Path
import os
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
from joblib import Parallel, delayed
from tqdm import tqdm

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGProcessor.loader import (
    load_tox21_dataset,
    load_bace_dataset,
    load_bbbp_dataset,
    load_clintox_dataset,
    load_sider_dataset,
    load_toxcast_dataset,
)
from KGGraph.KGGEncode.x_feature import x_feature
from KGGraph.KGGEncode.edge_feature import edge_feature


def feature(
    mol,
    decompose_type,
    mask_node=False,
    mask_edge=False,
    mask_node_ratio=0.1,
    mask_edge_ratio=0.1,
    fix_ratio=False,
):
    x_node, x, num_part = x_feature(
        mol,
        decompose_type=decompose_type,
        mask_node=mask_node,
        mask_node_ratio=mask_node_ratio,
        fix_ratio=fix_ratio,
    )
    edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(
        mol,
        decompose_type=decompose_type,
        mask_edge=mask_edge,
        mask_edge_ratio=mask_edge_ratio,
        fix_ratio=fix_ratio,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        decompose_type,
        mask_node,
        mask_edge,
        mask_node_ratio,
        mask_edge_ratio,
        fix_ratio=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset="tox21",
        empty=False,
    ):
        self.dataset = dataset
        self.decompose_type = decompose_type
        self.root = root
        self.mask_node = mask_node
        self.mask_edge = mask_edge
        self.mask_node_ratio = mask_node_ratio
        self.mask_edge_ratio = mask_edge_ratio
        self.fix_ratio = fix_ratio

        if not mask_node and not mask_edge:
            print("Not masking node and edge")
        elif not mask_node and mask_edge:
            print(
                "Masking edge with ratio at",
                mask_edge_ratio,
                "and fix state is",
                fix_ratio,
            )
        elif mask_node and not mask_edge:
            print(
                "Masking node with ratio at",
                mask_node_ratio,
                "and fix state is",
                fix_ratio,
            )
        else:
            print(
                "Masking node with ratio at",
                mask_node_ratio,
                "and masking edge with ratio at",
                mask_edge_ratio,
                "and fix state is",
                fix_ratio,
            )

        super(MoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.transform, self.pre_transform, self.pre_filter = (
            transform,
            pre_transform,
            pre_filter,
        )

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return "kgg_data_processed.pt"

    def download(self):
        raise NotImplementedError(
            "Must indicate valid location of raw data. No download allowed"
        )

    def process(self):
        data_smiles_list = []
        data_list = []
        print("Decompose type:", self.decompose_type)

        if self.dataset == "tox21":
            smiles_list, mols_list, labels = load_tox21_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == "bace":
            smiles_list, mols_list, folds, labels = load_bace_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data.fold = torch.tensor([folds[idx]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == "bbbp":
            smiles_list, mols_list, labels = load_bbbp_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == "clintox":
            smiles_list, mols_list, labels = load_clintox_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == "sider":
            smiles_list, mols_list, labels = load_sider_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == "toxcast":
            smiles_list, mols_list, labels = load_toxcast_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(
                delayed(feature)(
                    mol,
                    self.decompose_type,
                    self.mask_node,
                    self.mask_edge,
                    self.mask_node_ratio,
                    self.mask_edge_ratio,
                    self.fix_ratio,
                )
                for mol in tqdm(mols_list)
            )
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor(
                    [idx]
                )  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        else:
            raise ValueError(f"Dataset {self.dataset} is not supported")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(
            os.path.join(self.processed_dir, "smiles.csv"), index=False, header=False
        )

        if data_list:  # Ensure data_list is not empty
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = MoleculeDataset(
        "./Data/classification/tox21/",
        dataset="tox21",
        decompose_type="motif",
        mask_node=True,
        mask_edge=True,
        mask_node_ratio=0.25,
        mask_edge_ratio=0.1,
        fix_ratio=False,
    )
    print(dataset)
    print(dataset[0])
