import torch
import numpy as np
from torch_geometric.data import Data


class GraphDataset:
    def __init__(self, save_path, reagent_option=False):
        self.save_path = save_path
        self.rg_option = reagent_option
        self.load()

    def load(self):
        rmol_dict = np.load(self.save_path, allow_pickle=True)["rmol"]
        pmol_dict = np.load(self.save_path, allow_pickle=True)["pmol"]
        # rgmol_dict= np.load(self.save_path, allow_pickle=True)['rgmol']
        reaction_dict = np.load(self.save_path, allow_pickle=True)["reaction"].item()
        # load_link = np.load(self.save_path, allow_pickle=True)

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
        # print('p_dummy_shape: ',self.p_dummy[0][0])

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
        if self.rg_option:
            rgmol_dict = np.load(self.save_path, allow_pickle=True)["rgmol"]
            self.rgmol_max_cnt = len(rgmol_dict)

            # reagent
            self.rgmol_n_node = [
                rgmol_dict[j]["n_node"] for j in range(self.rgmol_max_cnt)
            ]  # have just added
            self.rgmol_n_edge = [
                rgmol_dict[j]["n_edge"] for j in range(self.rgmol_max_cnt)
            ]  # have just added
            self.rgmol_node_attr = [
                rgmol_dict[j]["node_attr"] for j in range(self.rgmol_max_cnt)
            ]  # have just added
            self.rgmol_edge_attr = [
                rgmol_dict[j]["edge_attr"] for j in range(self.rgmol_max_cnt)
            ]  # have just added
            self.rgmol_src = [
                rgmol_dict[j]["src"] for j in range(self.rgmol_max_cnt)
            ]  # have just added
            self.rgmol_dst = [
                rgmol_dict[j]["dst"] for j in range(self.rgmol_max_cnt)
            ]  # have just added

            # add csum reagent
            self.rgmol_n_csum = [
                np.concatenate([[0], np.cumsum(self.rgmol_n_node[j])])
                for j in range(self.rgmol_max_cnt)
            ]
            self.rgmol_e_csum = [
                np.concatenate([[0], np.cumsum(self.rgmol_n_edge[j])])
                for j in range(self.rgmol_max_cnt)
            ]

    def __getitem__(self, idx):
        data_r_lst = []
        for j in range(self.rmol_max_cnt):
            # fmt: off
            r_src = self.rmol_src[j][
                self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
            ]
            r_dst = self.rmol_dst[j][
                self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
            ]
            # fmt: on
            r_edge_index = torch.tensor([r_src, r_dst], dtype=torch.long)
            r_edge_index = torch.reshape(r_edge_index, (2, -1))

            # fmt: off
            r_edge_attr = torch.from_numpy(
                self.rmol_edge_attr[j][
                    self.rmol_e_csum[j][idx]: self.rmol_e_csum[j][idx + 1]
                ]
            ).float()
            if self.rmol_e_csum[j][idx + 1] - self.rmol_e_csum[j][idx] ==1:
                r_edge_attr = torch.tensor(r_edge_attr,requires_grad=False)
            r_node_attr = torch.from_numpy(
                self.rmol_node_attr[j][
                    self.rmol_n_csum[j][idx]: self.rmol_n_csum[j][idx + 1]
                ]
            ).float()
            if self.rmol_n_csum[j][idx + 1] - self.rmol_n_csum[j][idx] ==1:
                r_node_attr = torch.tensor(r_node_attr,requires_grad=False)
            # fmt: on
            data_r = Data(x=r_node_attr, edge_index=r_edge_index, edge_attr=r_edge_attr)

            data_r_lst.append(data_r)

        data_p_lst = []
        for j in range(self.pmol_max_cnt):
            # fmt: off
            p_src = self.pmol_src[j][
                self.pmol_e_csum[j][idx]: self.pmol_e_csum[j][idx + 1]
            ]
            p_dst = self.pmol_dst[j][
                self.pmol_e_csum[j][idx]: self.pmol_e_csum[j][idx + 1]
            ]
            # fmt: on
            p_edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)
            p_edge_index = torch.reshape(p_edge_index, (2, -1))
            # fmt: off
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
            # fmt: on
            data_p = Data(x=p_node_attr, edge_index=p_edge_index, edge_attr=p_edge_attr)
            data_p_lst.append(data_p)
        label = self.y[idx]
        rsmi = self.rsmi[idx]
        r_dummy = [i[idx] for i in self.r_dummy]
        p_dummy = [j[idx] for j in self.p_dummy]

        if self.rg_option:
            data_rg_lst = []
            for j in range(self.rgmol_max_cnt):
                # fmt: off
                rg_src = self.rgmol_src[j][
                    self.rgmol_e_csum[j][idx]: self.rgmol_e_csum[j][idx + 1]
                ]
                rg_dst = self.rgmol_dst[j][
                    self.rgmol_e_csum[j][idx]: self.rgmol_e_csum[j][idx + 1]
                ]
                # fmt: on
                rg_edge_index = torch.tensor([rg_src, rg_dst], dtype=torch.long)
                rg_edge_index = torch.reshape(rg_edge_index, (2, -1))

                # fmt: off
                rg_edge_attr = torch.from_numpy(
                    self.rgmol_edge_attr[j][
                        self.rgmol_e_csum[j][idx]: self.rgmol_e_csum[j][idx + 1]
                    ]
                ).float()

                rg_node_attr = torch.from_numpy(
                    self.rgmol_node_attr[j][
                        self.rgmol_n_csum[j][idx]: self.rgmol_n_csum[j][idx + 1]
                    ]
                ).float()
                # fmt: on
                data_rg = Data(
                    x=rg_node_attr, edge_index=rg_edge_index, edge_attr=rg_edge_attr
                )
                data_rg_lst.append(data_rg)
            return *data_r_lst, *data_p_lst, *data_rg_lst, label
        else:
            return *data_r_lst, *data_p_lst, r_dummy, p_dummy, label,rsmi

    def __len__(self):
        return self.y.shape[0]
