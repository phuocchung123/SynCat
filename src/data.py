import torch
import numpy as np
from torch_geometric.data import Data
from typing import Any, Tuple


class GraphDataset:
    """
    Dataset for chemical reaction graph classification.
    """

    def __init__(self, save_path: str) -> None:
        """
        Initialize GraphDataset and load data.

        Parameters
        ----------
        save_path : str
            Path to the saved .npz data file.
        """
        self.save_path = save_path
        self.load()

    def load(self) -> None:
        """
        Load and process reactant, product, and reaction data from file.
        """
        rmol_dict = np.load(self.save_path, allow_pickle=True)["rmol"]
        pmol_dict = np.load(self.save_path, allow_pickle=True)["pmol"]
        reaction_dict = np.load(self.save_path, allow_pickle=True)["reaction"].item()

        self.rmol_max_cnt = len(rmol_dict)
        self.pmol_max_cnt = len(pmol_dict)

        # reactant
        self.rmol_n_node = [rmol_dict[j]["n_node"] for j in range(self.rmol_max_cnt)]
        self.rmol_n_edge = [rmol_dict[j]["n_edge"] for j in range(self.rmol_max_cnt)]
        self.rmol_node_attr = [
            rmol_dict[j]["node_attr"] for j in range(self.rmol_max_cnt)
        ]
        self.rmol_edge_attr = [
            rmol_dict[j]["edge_attr"] for j in range(self.rmol_max_cnt)
        ]
        self.rmol_src = [rmol_dict[j]["src"] for j in range(self.rmol_max_cnt)]
        self.rmol_dst = [rmol_dict[j]["dst"] for j in range(self.rmol_max_cnt)]
        self.r_dummy = [rmol_dict[j]["dummy"] for j in range(self.rmol_max_cnt)]

        # product
        self.pmol_n_node = [pmol_dict[j]["n_node"] for j in range(self.pmol_max_cnt)]
        self.pmol_n_edge = [pmol_dict[j]["n_edge"] for j in range(self.pmol_max_cnt)]
        self.pmol_node_attr = [
            pmol_dict[j]["node_attr"] for j in range(self.pmol_max_cnt)
        ]
        self.pmol_edge_attr = [
            pmol_dict[j]["edge_attr"] for j in range(self.pmol_max_cnt)
        ]
        self.pmol_src = [pmol_dict[j]["src"] for j in range(self.pmol_max_cnt)]
        self.pmol_dst = [pmol_dict[j]["dst"] for j in range(self.pmol_max_cnt)]
        self.p_dummy = [pmol_dict[j]["dummy"] for j in range(self.pmol_max_cnt)]

        self.y = reaction_dict["y"]
        self.rsmi = reaction_dict["rsmi"]

        # add csum reactant
        self.rmol_n_csum = [
            np.concatenate([[0], np.cumsum(self.rmol_n_node[j])])
            for j in range(self.rmol_max_cnt)
        ]
        self.rmol_e_csum = [
            np.concatenate([[0], np.cumsum(self.rmol_n_edge[j])])
            for j in range(self.rmol_max_cnt)
        ]

        # add csum product
        self.pmol_n_csum = [
            np.concatenate([[0], np.cumsum(self.pmol_n_node[j])])
            for j in range(self.pmol_max_cnt)
        ]
        self.pmol_e_csum = [
            np.concatenate([[0], np.cumsum(self.pmol_n_edge[j])])
            for j in range(self.pmol_max_cnt)
        ]

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Get graph data for a specific reaction sample.

        Parameters
        ----------
        idx : int
            Index of the reaction sample.

        Returns
        -------
        tuple
            Tuple containing reactant Data objects, product Data objects,
            reactant dummies, product dummies, label, and rsmi string.
        """
        data_r_lst = []
        for j in range(self.rmol_max_cnt):
            r_src = self.rmol_src[j][
                self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
            ]
            r_dst = self.rmol_dst[j][
                self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
            ]

            r_edge_index = torch.tensor([r_src, r_dst], dtype=torch.long)
            r_edge_index = torch.reshape(r_edge_index, (2, -1))

            r_edge_attr = torch.from_numpy(
                self.rmol_edge_attr[j][
                    self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
                ]
            ).float()

            r_node_attr = torch.from_numpy(
                self.rmol_node_attr[j][
                    self.rmol_n_csum[j][idx]: self.rmol_n_csum[j][idx + 1]
                ]
            ).float()

            data_r = Data(x=r_node_attr, edge_index=r_edge_index, edge_attr=r_edge_attr)

            data_r_lst.append(data_r)

        data_p_lst = []
        for j in range(self.pmol_max_cnt):

            p_src = self.pmol_src[j][
                self.pmol_e_csum[j][idx]: self.pmol_e_csum[j][idx + 1]
            ]
            p_dst = self.pmol_dst[j][
                self.pmol_e_csum[j][idx]: self.pmol_e_csum[j][idx + 1]
            ]

            p_edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)
            p_edge_index = torch.reshape(p_edge_index, (2, -1))

            p_edge_attr = torch.from_numpy(
                self.pmol_edge_attr[j][
                    self.pmol_e_csum[j][idx]: self.pmol_e_csum[j][idx + 1]
                ]
            ).float()

            p_node_attr = torch.from_numpy(
                self.pmol_node_attr[j][
                    self.pmol_n_csum[j][idx]: self.pmol_n_csum[j][idx + 1]
                ]
            ).float()

            data_p = Data(x=p_node_attr, edge_index=p_edge_index, edge_attr=p_edge_attr)
            data_p_lst.append(data_p)

        label = self.y[idx]
        rsmi = self.rsmi[idx]
        r_dummy = [i[idx] for i in self.r_dummy]
        p_dummy = [j[idx] for j in self.p_dummy]

        return *data_r_lst, *data_p_lst, r_dummy, p_dummy, label, rsmi

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of reaction samples.
        """
        return self.y.shape[0]
