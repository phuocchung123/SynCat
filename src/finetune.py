import os
import json
import torch
import numpy as np
import pandas as pd
from data import GraphDataset
from torch.utils.data import DataLoader
from model import model
from training import train
from validation import validation
from utils import collate_reaction_graphs, setup_logging


def finetune(args) -> None:
    """
    Fine-tune a graph neural network on chemical reaction data.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing all required settings and paths.

    Returns
    -------
    None
    """
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")
    model_path = args.model_path + args.model_name
    data = pd.read_csv(args.Data_folder + args.data_csv, compression="gzip")
    out_dim = data[args.y_column].nunique()
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("device is\t", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "train.npz")
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(np.min([args.batch_size, len(train_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=True,
    )

    test_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "test.npz")
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int(np.min([args.batch_size, len(test_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=False,
    )

    val_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "valid.npz")
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=int(np.min([args.batch_size, len(val_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=2,
        drop_last=False,
    )

    logger.info("-- CONFIGURATIONS")
    logger.info(
        "--- train/valid/test: %d/%d/%d" % (len(train_set), len(val_set), len(test_set))
    )
    logger.info(
        "--- max no. reactants_train, valid, test respectively:",
        train_set.rmol_max_cnt,
        val_set.rmol_max_cnt,
        test_set.rmol_max_cnt,
    )
    logger.info(
        "--- max no. products_train, valid, test respectively:",
        train_set.pmol_max_cnt,
        val_set.pmol_max_cnt,
        test_set.pmol_max_cnt,
    )
    logger.info("--- model_path:", model_path)

    # training

    node_dim = train_set.rmol_node_attr[0].shape[1]
    edge_dim = train_set.rmol_edge_attr[0].shape[1]
    net = model(node_dim, edge_dim, out_dim, args.layer, args.emb_dim, args.dropout).to(
        device
    )
    if not os.path.exists(model_path):
        logger.info("-- TRAINING")
        net = train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
        )
    else:
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        current_epoch = checkpoint["epoch"]
        epochs = args.epochs - current_epoch
        net = train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            epochs,
            args.lr,
            args.weight_decay,
            current_epoch=current_epoch,
            best_val_loss=checkpoint["val_loss"],
        )

    # test
    test_y = test_loader.dataset.y
    net = model(node_dim, edge_dim, out_dim, args.layer, args.emb_dim, args.dropout).to(
        device
    )
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    acc, mcc, att_r, att_p, rsmis, _, _, emb = validation(
        args, net, test_loader, device
    )
    logger.info("-- RESULT")
    logger.info("--- test size: %d" % (len(test_y)))
    logger.info("--- Accuracy: %.3f, Mattews Correlation: %.3f," % (acc, mcc))

    dict_att = {
        "Name": "Attention",
        "rsmis": rsmis,
        "att_r": att_r,
        "att_p": att_p,
        "emb": emb,
    }
    with open(args.monitor_folder + "attention.json", "w") as f:
        json.dump(dict_att, f)
