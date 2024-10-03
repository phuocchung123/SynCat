import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# from sklearn.metrics import roc_auc_score, average_precision_score

MAX_BOND_TYPE = 5
MAX_ATOM_TYPE = 119


def create_var(tensor, device, requires_grad=None):
    """Create a PyTorch Variable tensor on the specified device."""
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)


class Model_decoder(nn.Module):
    def __init__(self, hidden_size, device, dropout=0.2):
        super(Model_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.bond_if_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.bond_if_s = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

        # bond type
        # self.bond_type_s = nn.Sequential(
        #     nn.Linear(2 * hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, MAX_BOND_TYPE),
        # )

        # bond type features
        self.bond_type_s_sigma = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.bond_type_s_pi = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.bond_type_s_conjugate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # atom type
        # self.atom_type_s = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, MAX_ATOM_TYPE),
        # )

        # hybridization features
        self.atom_hybri_s_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.atom_hybri_s_p = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4)
        )
        self.atom_hybri_s_d = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3)
        )
        self.atom_hybri_s_a = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 7)
        )
        self.atom_hybri_s_lonepair = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7),
        )

        self.atom_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Softplus(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.bond_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Softplus(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.bond_pred_loss = nn.BCEWithLogitsLoss()
        # self.bond_type_pred_loss = nn.CrossEntropyLoss()
        # bond type features
        self.bond_type_sigma_pred_loss = nn.BCEWithLogitsLoss()
        self.bond_type_pi_pred_loss = nn.SmoothL1Loss()
        self.bond_type_conjugate_pred_loss = nn.BCEWithLogitsLoss()

        # self.atom_type_pred_loss = nn.CrossEntropyLoss()
        # hybridization features
        self.atom_hybri_s_pred_loss = nn.BCEWithLogitsLoss()
        self.atom_hybri_p_pred_loss = nn.CrossEntropyLoss()
        self.atom_hybri_d_pred_loss = nn.CrossEntropyLoss()
        self.atom_hybri_a_pred_loss = nn.CrossEntropyLoss()
        self.atom_hybri_lonepair_pred_loss = nn.CrossEntropyLoss()

        self.atom_num_pred_loss = nn.SmoothL1Loss()
        self.bond_num_pred_loss = nn.SmoothL1Loss()

    def super_node_rep(self, mol_batch, node_rep):
        super_group = []
        for mol_index, mol in enumerate(mol_batch):
            super_group.append(node_rep[mol_index][-1, :]).to(self.device)
        super_rep = torch.stack(super_group, dim=0)
        return super_rep

    def topo_pred(self, mol_batch, node_rep, super_node_rep):
        bond_if_loss = 0
        # bond_type_loss = 0
        (
            bond_type_sigma_loss,
            bond_type_pi_loss,
            bond_type_conjugate_loss,
        ) = (0, 0, 0)
        # atom_type_loss = 0
        (
            atom_hybri_s_loss,
            atom_hybri_p_loss,
            atom_hybri_d_loss,
            atom_hybri_a_loss,
            atom_hybri_lonepair_loss,
        ) = (
            0,
            0,
            0,
            0,
            0,
        )
        atom_num_loss, bond_num_loss = 0, 0

        atom_num_target, bond_num_target = [], []
        for mol in mol_batch:
            num_atoms = mol.size_atom
            atom_num_target.append(num_atoms)
            num_bonds = mol.size_bond
            bond_num_target.append(num_bonds)

        # predict atom number, bond number
        super_rep = torch.stack(super_node_rep, dim=0).to(self.device)
        atom_num_pred = self.atom_num_s(super_rep).squeeze(-1)
        atom_num_target = (
            torch.tensor(np.array(atom_num_target))
            .to(atom_num_pred.dtype)
            .to(self.device)
        )
        atom_num_loss += self.atom_num_pred_loss(atom_num_pred, atom_num_target) / len(
            mol_batch
        )
        # atom_num_rmse = torch.sqrt(torch.sum((atom_num_pred - atom_num_target) ** 2)).item() / len(mol_batch)

        super_rep = torch.stack(super_node_rep, dim=0).to(self.device)
        bond_num_pred = self.bond_num_s(super_rep).squeeze(-1)
        bond_num_target = (
            torch.tensor(np.array(bond_num_target))
            .to(bond_num_pred.dtype)
            .to(self.device)
        )
        bond_num_loss += self.bond_num_pred_loss(bond_num_pred, bond_num_target) / len(
            mol_batch
        )
        # bond_num_rmse = torch.sqrt(torch.sum((bond_num_pred - bond_num_target) ** 2)).item() / len(mol_batch)

        # predict atom type, bond type
        mol_num = len(mol_batch)
        for mol_index, mol in enumerate(mol_batch):
            num_atoms = mol.size_atom
            num_bonds = mol.size_bond
            if num_bonds < 1:
                mol_num -= 1
            else:
                # bond link
                mol_rep = node_rep[mol_index].to(self.device)
                mol_atom_rep_proj = self.feat_drop(self.bond_if_proj(mol_rep))

                bond_if_input = torch.cat(
                    [
                        mol_atom_rep_proj.repeat(1, num_atoms).view(
                            num_atoms * num_atoms, -1
                        ),
                        mol_atom_rep_proj.repeat(num_atoms, 1),
                    ],
                    dim=1,
                ).view(num_atoms, num_atoms, -1)
                bond_if_pred = self.bond_if_s(bond_if_input).squeeze(-1)
                a = torch.zeros(num_atoms, num_atoms)
                bond_if_target = a.index_put(
                    indices=[
                        mol.edge_index_nosuper[0, :],
                        mol.edge_index_nosuper[1, :],
                    ],
                    values=torch.tensor(1.0),
                ).to(self.device)
                bond_if_loss += self.bond_pred_loss(bond_if_pred, bond_if_target)
                # bond_if_auc += roc_auc_score(bond_if_target.flatten().cpu().detach(), torch.sigmoid(bond_if_pred.flatten().cpu().detach()))
                # bond_if_ap += average_precision_score(bond_if_target.cpu().detach(), torch.sigmoid(bond_if_pred.cpu().detach()))

                # bond type
                start_rep = mol_atom_rep_proj.index_select(
                    0, mol.edge_index_nosuper[0, :].to(self.device)
                )
                end_rep = mol_atom_rep_proj.index_select(
                    0, mol.edge_index_nosuper[1, :].to(self.device)
                )

                bond_type_input = torch.cat([start_rep, end_rep], dim=1)
                # bond_type_pred = self.bond_type_s(bond_type_input)
                bond_type_sigma_pred = self.bond_type_s_sigma(bond_type_input).squeeze(
                    -1
                )
                bond_type_pi_pred = self.bond_type_s_pi(bond_type_input)
                bond_type_conjugate_pred = self.bond_type_s_conjugate(
                    bond_type_input
                ).squeeze(-1)

                # bond_type_target = mol.edge_attr_nosuper[:, 0].to(self.device).long()
                bond_type_sigma_target = mol.edge_attr_nosuper[:, 2].to(self.device)
                bond_type_pi_target = mol.edge_attr_nosuper[:, 3].to(self.device)
                bond_type_conjugate_target = mol.edge_attr_nosuper[:, 4].to(self.device)

                # bond_type_loss += self.bond_type_pred_loss(
                #     bond_type_pred, bond_type_target
                # )
                bond_type_sigma_loss += self.bond_type_sigma_pred_loss(
                    bond_type_sigma_pred, bond_type_sigma_target
                )
                bond_type_pi_loss += self.bond_type_pi_pred_loss(
                    bond_type_pi_pred, bond_type_pi_target
                )
                bond_type_conjugate_loss += self.bond_type_conjugate_pred_loss(
                    bond_type_conjugate_pred, bond_type_conjugate_target
                )

                # atom type
                mol_rep = node_rep[mol_index].to(self.device)
                # atom_type_pred = self.atom_type_s(mol_rep)

                # atom_type_target = mol.x_nosuper[:, 0].to(self.device).long()
                # atom_type_loss += self.atom_type_pred_loss(
                #     atom_type_pred, atom_type_target
                # )

                # _, preds = torch.max(atom_type_pred, dim=1)
                # pred_acc = torch.eq(preds, atom_type_target).float()
                # atom_type_acc += (torch.sum(pred_acc) / atom_type_target.nelement())

                # atom hybridization
                atom_hybri_s_pred = self.atom_hybri_s_s(mol_rep).squeeze(-1)
                atom_hybri_p_pred = self.atom_hybri_s_p(mol_rep)
                atom_hybri_d_pred = self.atom_hybri_s_d(mol_rep)
                atom_hybri_a_pred = self.atom_hybri_s_a(mol_rep)
                atom_hybri_lonepair_pred = self.atom_hybri_s_lonepair(mol_rep)

                atom_hybri_s_target = mol.x_nosuper[:, 2].to(self.device)
                atom_hybri_p_target = mol.x_nosuper[:, 3].to(self.device).long()
                atom_hybri_d_target = mol.x_nosuper[:, 4].to(self.device).long()
                atom_hybri_a_target = mol.x_nosuper[:, 5].to(self.device).long()
                atom_hybri_lonepair_target = mol.x_nosuper[:, 6].to(self.device).long()

                atom_hybri_s_loss += self.atom_hybri_s_pred_loss(
                    atom_hybri_s_pred, atom_hybri_s_target
                )
                atom_hybri_p_loss += self.atom_hybri_p_pred_loss(
                    atom_hybri_p_pred, atom_hybri_p_target
                )
                atom_hybri_d_loss += self.atom_hybri_d_pred_loss(
                    atom_hybri_d_pred, atom_hybri_d_target
                )
                atom_hybri_a_loss += self.atom_hybri_a_pred_loss(
                    atom_hybri_a_pred, atom_hybri_a_target
                )
                atom_hybri_lonepair_loss += self.atom_hybri_lonepair_pred_loss(
                    atom_hybri_lonepair_pred, atom_hybri_lonepair_target
                )

        loss_tur = [
            bond_if_loss / mol_num,
            bond_type_sigma_loss / mol_num,
            bond_type_pi_loss / mol_num,
            bond_type_conjugate_loss / mol_num,
            # bond_type_loss / mol_num,
            # atom_type_loss / mol_num,
            atom_hybri_s_loss / mol_num,
            atom_hybri_p_loss / mol_num,
            atom_hybri_d_loss / mol_num,
            atom_hybri_a_loss / mol_num,
            atom_hybri_lonepair_loss / mol_num,
            atom_num_loss,
            bond_num_loss,
        ]
        # results = [bond_if_auc/mol_num, bond_if_ap/mol_num, atom_type_acc/mol_num, atom_num_rmse, bond_num_rmse]

        return loss_tur

    def forward(self, mol_batch, node_rep, super_node_rep):
        loss_tur = self.topo_pred(mol_batch, node_rep, super_node_rep)
        loss = 0
        # loss_weight = create_var(torch.rand(10), self.device, requires_grad=True)
        # loss_wei = torch.softmax(loss_weight, dim=-1)
        for index in range(len(loss_tur)):
            # loss += loss_tur[index] * loss_wei[index]
            loss += loss_tur[index]
        return loss
