import os
import json
import torch
import numpy as np
import pandas as pd
from data import GraphDataset
from torch.utils.data import DataLoader
from model import recat, train, inference
from utils import collate_reaction_graphs


def finetune(args):
    batch_size = args.batch_size
    model_path = args.model_path + args.model_name
    monitor_path = args.monitor_folder + args.monitor_name
    epochs = args.epochs
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

    train_set = GraphDataset(
        args.Data_folder + args.npz_folder + "/" + args.train_set,
        reagent_option=args.reagent_option,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(np.min([batch_size, len(train_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=True,
    )

    test_set = GraphDataset(
        args.Data_folder + args.npz_folder + "/" + args.test_set,
        reagent_option=args.reagent_option,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int(np.min([batch_size, len(test_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=False,
    )

    val_set = GraphDataset(
        args.Data_folder + args.npz_folder + "/" + args.val_set,
        reagent_option=args.reagent_option,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=int(np.min([batch_size, len(val_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=False,
    )

    print("-- CONFIGURATIONS")
    print(
        "--- train/valid/test: %d/%d/%d" % (len(train_set), len(val_set), len(test_set))
    )
    print(
        "--- max no. reactants_train, valid, test respectively:",
        train_set.rmol_max_cnt,
        val_set.rmol_max_cnt,
        test_set.rmol_max_cnt,
    )
    print(
        "--- max no. products_train, valid, test respectively:",
        train_set.pmol_max_cnt,
        val_set.pmol_max_cnt,
        test_set.pmol_max_cnt,
    )
    print("--- model_path:", model_path)

    # training
    train_y = train_loader.dataset.y

    assert len(train_y) == len(train_set)
    node_dim = train_set.rmol_node_attr[0].shape[1]
    edge_dim = train_set.rmol_edge_attr[0].shape[1]
    if not os.path.exists(model_path):
        net = recat(node_dim, edge_dim, out_dim).to(device)
        print("-- TRAINING")
        net = train(
            args, net, train_loader, val_loader, model_path, device, epochs=epochs
        )
    else:
        net = recat(node_dim, edge_dim, out_dim).to(device)
        checkpoint = torch.load(model_path,map_location=device,weights_only=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        current_epoch = checkpoint["epoch"]
        epochs = epochs - current_epoch
        net = train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            epochs=epochs,
            current_epoch=current_epoch,
            best_val_loss=checkpoint["val_loss"],
        )

    # test
    test_y = test_loader.dataset.y
    # test_y = torch.argmax(torch.Tensor(test_y), dim=1).tolist()
    net = recat(node_dim, edge_dim, out_dim).to(device)
    checkpoint = torch.load(model_path,map_location=device,weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    r2, rmse, mae, atts_reactant, atts_product, rsmi, targets, preds= inference(args, net, test_loader, device)
    print("-- RESULT")
    print("--- test size: %d" % (len(test_y)))
    print("--- R2: %.3f, RMSE: %.3f, MAE: %.3f" % (r2, rmse, mae))
    dict = {
        "Name": "Test",
        "test_r2": np.round(r2, decimals=3),
        "test_rmse": np.round(rmse,decimals=3),
        "test_mae": np.round(mae,decimals=3),
    }
    
    dict_att = {
        "Name": "Attention",
        "rsmi": rsmi,
        "Attention reactant": atts_reactant,
        "Attention product":atts_product
    }
    
    with open(monitor_path, "a") as f:
        f.write(json.dumps(dict) + "\n")
    with open('../Data/monitor/attention2.json','w') as f:
        json.dump(dict_att,f)
    np.savez_compressed('../Data/monitor/target_pred.npz',targets=targets, preds= preds)
