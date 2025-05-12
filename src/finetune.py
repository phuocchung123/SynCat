import os
import json
import torch
import numpy as np
import pandas as pd
from data import GraphDataset
from torch.utils.data import DataLoader
from model import recat, inference
from utils import collate_reaction_graphs


def finetune(args):
    batch_size = args.batch_size
    model_path = args.model_path + args.model_name
    epochs = 0
    data = pd.read_csv(args.Data_folder + args.data_csv)
    out_dim = data[args.y_column].nunique()
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is\t", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    inference_set = GraphDataset(
        args.Data_folder + args.npz_inference + "/inference.npz"
    )
    inference_loader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=False,
    )


    print("-- CONFIGURATIONS")
    print(
        "--- data: %d" % (len(inference_set))
    )
    print(
        "--- max no. reactants_inference:",
        inference_set.rmol_max_cnt,
    )
    print(
        "--- max no. products_inference:",
        inference_set.pmol_max_cnt,
    )
    print("--- model_path:", model_path)

    node_dim = inference_set.rmol_node_attr[0].shape[1]
    edge_dim = inference_set.rmol_edge_attr[0].shape[1]

    # test
    test_y = inference_loader.dataset.y
    test_y = torch.argmax(torch.Tensor(test_y), dim=1).tolist()
    net = recat(node_dim, edge_dim, out_dim).to(device)
    checkpoint = torch.load(model_path,map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    _, _, atts_reactant, atts_product, rsmi, _, _, emb= inference(args, net, inference_loader, device)
    print("-- RESULT")
    print("--- test size: %d" % (len(test_y)))
    
    dict_att = {
        "Name": "Attention",
        "rsmi": rsmi,
        "Attention reactant": atts_reactant,
        "Attention product":atts_product,
        "emb": emb
    }
    

    with open('../Data/monitor/attention_inference.json','w') as f:
        json.dump(dict_att,f)
