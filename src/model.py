import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from gin import GIN
from utils import setup_logging
from sklearn.metrics import accuracy_score, matthews_corrcoef

logger = setup_logging()


class recat(nn.Module):
    def __init__(
        self,
        node_in_feats=155,
        edge_in_feats=9,
        out_dim=4,
        num_layer=3,
        node_hid_feats=384,
        readout_feats=1024,
        predict_hidden_feats=512,
        readout_option=False,
        drop_ratio=0.1,
    ):
        super(recat, self).__init__()
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            depth=num_layer,
            node_hid_feats=node_hid_feats,
            readout_feats=readout_feats,
            readout_option=readout_option,
        )
        if readout_option:
            emb_dim = readout_feats
        else:
            emb_dim = node_hid_feats

        # self.predict = nn.Sequential(
        #     torch.nn.Linear(emb_dim, predict_hidden_feats),
        #     torch.nn.PReLU(),
        #     torch.nn.Dropout(drop_ratio),
        #     torch.nn.Linear(predict_hidden_feats, predict_hidden_feats),
        #     torch.nn.PReLU(),
        #     torch.nn.Dropout(drop_ratio),
        #     torch.nn.Linear(predict_hidden_feats, out_dim),
        # )
        self.predict = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, rmols, pmols, rgmols=None):
        r_graph_feats = torch.sum(
            torch.stack([self.gnn(rmol) for rmol in rmols]), dim=0
        )
        p_graph_feats = torch.sum(
            torch.stack([self.gnn(pmol) for pmol in pmols]), dim=0
        )

        react_graph_feats = torch.sub(r_graph_feats, p_graph_feats)
        if rgmols is not None:
            rg_graph_feats = torch.sum(
                torch.stack([self.gnn(rgmol) for rgmol in rgmols]), dim=0
            )
            react_graph_feats = react_graph_feats * 0.7 + rg_graph_feats * 0.3
        out = self.predict(react_graph_feats)
        return out


def train(
    args,
    net,
    train_loader,
    val_loader,
    model_path,
    device,
    epochs=20,
    current_epoch=0,
    best_val_loss=1e10,
):
    # train_size = train_loader.dataset.__len__()
    # batch_size = train_loader.batch_size
    monitor_path = args.monitor_folder + args.monitor_name
    n_epochs = epochs

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
        if args.reagent_option:
            rgmol_max_cnt = train_loader.dataset.dataset.rgmol_max_cnt

    except Exception as e:
        logger.error(e)
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt
        if args.reagent_option:
            rgmol_max_cnt = train_loader.dataset.rgmol_max_cnt

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []
        targets = []
        preds = []

        for batchdata in tqdm(train_loader, desc="Training"):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            if args.reagent_option:
                inputs_rgmol = [
                    b.to(device)
                    for b in batchdata[
                        rmol_max_cnt
                        + pmol_max_cnt: rmol_max_cnt
                        + pmol_max_cnt
                        + rgmol_max_cnt
                    ]
                ]
            # fmt: on
                pred = net(inputs_rmol, inputs_pmol, inputs_rgmol)
            else:
                pred = net(inputs_rmol, inputs_pmol)
            labels = batchdata[-1]
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(torch.argmax(pred, dim=1).tolist())
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        acc = accuracy_score(targets, preds)
        mcc = matthews_corrcoef(targets, preds)
        print(
            "--- training epoch %d, loss %.3f, acc %.3f, mcc %.3f, time elapsed(min) %.2f---"
            % (
                epoch,
                np.mean(train_loss_list),
                acc,
                mcc,
                (time.time() - start_time) / 60,
            )
        )

        # validation
        net.eval()
        val_acc, val_mcc, val_loss = inference(args, net, val_loader, device, loss_fn)

        print(
            "--- validation at epoch %d, val_loss %.3f, val_acc %.3f, val_mcc %.3f ---"
            % (epoch, val_loss, val_acc, val_mcc)
        )
        print("\n" + "*" * 100)

        dict = {
            "epoch": epoch + current_epoch,
            "train_loss": np.mean(train_loss_list),
            "val_loss": val_loss,
            "train_acc": acc,
            "val_acc": val_acc,
        }
        with open(monitor_path, "a") as f:
            f.write(json.dumps(dict) + "\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + current_epoch,
                    "model_state_dict": net.state_dict(),
                    "val_loss": best_val_loss,
                },
                model_path,
            )


def inference(args, net, test_loader, device, loss_fn=None):
    # batch_size = test_loader.batch_size

    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
        if args.reagent_option:
            rgmol_max_cnt = test_loader.dataset.dataset.rgmol_max_cnt

    except Exception as e:
        logger.error(e)
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt
        if args.reagent_option:
            rgmol_max_cnt = test_loader.dataset.rgmol_max_cnt

    net.eval()
    inference_loss_list = []
    preds = []
    targets = []

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc="Testing"):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            if args.reagent_option:
                inputs_rgmol = [
                    b.to(device)
                    for b in batchdata[
                        rmol_max_cnt
                        + pmol_max_cnt: rmol_max_cnt
                        + pmol_max_cnt
                        + rgmol_max_cnt
                    ]
                ]
            # fmt: on
                pred = net(inputs_rmol, inputs_pmol, inputs_rgmol)
            else:
                pred = net(inputs_rmol, inputs_pmol)
            labels = batchdata[-1]
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(torch.argmax(pred, dim=1).tolist())

            if loss_fn is not None:
                inference_loss = loss_fn(pred, labels)
                inference_loss_list.append(inference_loss.item())

    acc = accuracy_score(targets, preds)
    mcc = matthews_corrcoef(targets, preds)

    if loss_fn is None:
        return acc, mcc, preds
    else:
        return acc, mcc, np.mean(inference_loss_list)
