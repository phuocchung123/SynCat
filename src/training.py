import time
import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from validation import validation
from utils import setup_logging
from sklearn.metrics import accuracy_score, matthews_corrcoef



def train(
    args,
    net,
    train_loader,
    val_loader,
    model_path,
    device,
    epochs,
    learning_rate,
    weight_decay,
    current_epoch=0,
    best_val_loss=1e10,
):
    logger = setup_logging(log_filename=args.monitor_folder+'monitor.log')

    rmol_max_cnt = train_loader.dataset.rmol_max_cnt
    pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []
        labels = []
        preds = []

        for batchdata in tqdm(train_loader, desc="Training"):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            r_dummy=batchdata[-4]
            p_dummy=batchdata[-3]
            

            pred,_,_,_ = net(inputs_rmol, inputs_pmol,r_dummy,p_dummy,device)
            label = batchdata[-2]
            label = label.to(device)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels.extend(label.tolist())
            preds.extend(torch.argmax(pred, dim=1).tolist())
            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        logger.info(
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
        val_acc, val_mcc, val_loss = validation(args, net, val_loader, device,loss_fn)

        logger.info(
            "--- validation at epoch %d, val_loss %.3f, val_acc %.3f, val_mcc %.3f ---"
            % (epoch, val_loss, val_acc, val_mcc)
        )
        logger.info("\n" + "*" * 100)


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

