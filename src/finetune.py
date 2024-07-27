import numpy as np
import torch
import csv, os
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from scipy import stats

from model import recat, train, inference
from data import GraphDataset
from util import collate_reaction_graphs

def finetune(args):
    batch_size = args.batch_size
    model_path = args.model_path
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = GraphDataset(args.npz_folder + args.train_set)
    train_loader= DataLoader(
        dataset=train_set,
        batch_size= int(np.min([batch_size, len(train_set)])),
        shuffle= True,
        collate_fn=collate_reaction_graphs,
        num_workers=0,
        drop_last=True
    )

    test_set = GraphDataset(args.npz_folder + args.test_set)
    test_loader= DataLoader(
        dataset=test_set,
        batch_size= int(np.min([batch_size, len(test_set)])),
        shuffle= False,
        collate_fn=collate_reaction_graphs,
        num_workers=0,
        drop_last=False
    )

    val_set = GraphDataset(args.npz_folder + args.val_set)
    val_loader= DataLoader(
        dataset=val_set,
        batch_size= int(np.min([batch_size, len(val_set)])),
        shuffle= False,
        collate_fn=collate_reaction_graphs,
        num_workers=0,
        drop_last=False
    )


    print("-- CONFIGURATIONS")
    print("--- train/valid/test: %d/%d/%d" % (len(train_set),len(val_set), len(test_set)))
    print("--- max no. reactants_train, valid, test respectively:", train_set.rmol_max_cnt, val_set.rmol_max_cnt, test_set.rmol_max_cnt)
    print("--- max no. products_train, valid, test respectively:", train_set.pmol_max_cnt, val_set.pmol_max_cnt, test_set.pmol_max_cnt)
    print("--- model_path:", model_path)

    #training
    train_y= train_loader.dataset.y

    assert len(train_y) == len(train_set)
    node_dim= train_set.rmol_node_attr[0].shape[1]
    edge_dim= train_set.rmol_edge_attr[0].shape[1]
    net=recat(node_dim, edge_dim).to(device)
    print("-- TRAINING")
    net = train(net, train_loader,val_loader, model_path)

    #test
    test_y= test_loader.dataset.y
    test_y=torch.argmax(torch.Tensor(test_y),dim=1).tolist()
    net=recat(node_dim, edge_dim).to(device)
    net.load_state_dict(torch.load(model_path))
    acc,mcc =inference(net, test_loader)
    print("-- RESULT")
    print("--- test size: %d" % (len(test_y)))
    print(
        "--- Accuracy: %.3f, Mattews Correlation: %.3f,"
        % (acc, mcc)
    )
