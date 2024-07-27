import numpy as np
import torch
from torch_geometric.data import Batch

def collate_reaction_graphs(batch):
    batchdata = list(map(list, zip(*batch)))
    datas=[Batch.from_data_list(d) for d in batchdata[:-1]]
    labels = torch.stack([torch.argmax(y) for y in torch.Tensor(batchdata[-1])], axis=0) 

    return *datas, labels