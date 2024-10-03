from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data
import sys
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGEncode.edge_feature import edge_feature
from KGGraph.KGGEncode.x_feature import x_feature
from KGGraph.KGGChem.atom_utils import get_mol


class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_file,
        decompose_type,
        mask_node,
        mask_edge,
        mask_node_ratio,
        mask_edge_ratio,
        fix_ratio,
    ):
        self.decompose_type = decompose_type
        self.mask_node = mask_node
        self.mask_edge = mask_edge
        self.mask_node_ratio = mask_node_ratio
        self.mask_edge_ratio = mask_edge_ratio
        self.fix_ratio = fix_ratio
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

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

        print("Decompose type", decompose_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_graph = MolGraph(
            smiles,
            self.decompose_type,
            self.mask_node,
            self.mask_edge,
            self.mask_node_ratio,
            self.mask_edge_ratio,
            self.fix_ratio,
        )
        return mol_graph


class MolGraph(object):
    def __init__(
        self,
        smiles,
        decompose_type,
        mask_node,
        mask_edge,
        mask_node_ratio,
        mask_edge_ratio,
        fix_ratio,
    ):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self._x_nosuper, self._x, self._num_part = x_feature(
            self.mol, decompose_type, mask_node, mask_node_ratio, fix_ratio
        )
        (
            self._edge_attr_nosuper,
            self._edge_index_nosuper,
            self._edge_index,
            self._edge_attr,
        ) = edge_feature(
            self.mol, decompose_type, mask_edge, mask_edge_ratio, fix_ratio
        )

    @property
    def x(self):
        return self._x

    @property
    def x_nosuper(self):
        return self._x_nosuper

    @property
    def num_part(self):
        return self._num_part

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def edge_attr(self):
        return self._edge_attr

    @property
    def edge_index_nosuper(self):
        return self._edge_index_nosuper

    @property
    def edge_attr_nosuper(self):
        return self._edge_attr_nosuper

    @property
    def size_node(self):
        return self._x.size()[0]

    @property
    def size_edge(self):
        return self._edge_attr.size()[0]

    @property
    def size_atom(self):
        return self._x_nosuper.size()[0]

    @property
    def size_bond(self):
        return self._edge_attr_nosuper.size()[0]


def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(
            x=mol.x,
            edge_index=mol.edge_index,
            edge_attr=mol.edge_attr,
            num_part=mol.num_part,
        )
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch
