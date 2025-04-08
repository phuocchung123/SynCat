import time
import itertools
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from gin import GIN
from attention import EncoderLayer
from utils import setup_logging
from sklearn.metrics import accuracy_score, matthews_corrcoef

logger = setup_logging()


class recat(nn.Module):
    def __init__(
        self,
        node_in_feats=155,
        edge_in_feats=9,
        out_dim=4,
        num_layer=2,
        node_hid_feats=256,
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
        self.predict= torch.nn.Linear(emb_dim,out_dim)
        self.attention=EncoderLayer()
        self.atts_reactant=[]
        self.atts_product=[]

    def forward(self, rmols, pmols, r_dummy=None, p_dummy=None,device='cuda:0'):
        r_graph_feats = torch.stack([self.gnn(rmol) for rmol in rmols])
        p_graph_feats = torch.stack([self.gnn(pmol) for pmol in pmols])

        reaction_vectors=torch.tensor([]).to(device)
        
        for batch in range(r_graph_feats.shape[1]):
            ### reactant and product vector correspoding each reaction
            r_graph_feats_1=r_graph_feats[:,batch,:][r_dummy[batch]].to(device)
            new_rows_r=[r_graph_feats_1[i] for i in range(r_graph_feats_1.size(0))]
            for i, j in itertools.combinations(range(r_graph_feats_1.size(0)), 2):
                new_rows_r.append(r_graph_feats_1[i] + r_graph_feats_1[j]) 
            r_graph_feats_1=torch.stack(new_rows_r).to(device)


            p_graph_feats_1=p_graph_feats[:,batch,:][p_dummy[batch]].to(device)
            new_rows_p=[p_graph_feats_1[i] for i in range(p_graph_feats_1.size(0))]
            for i, j in itertools.combinations(range(p_graph_feats_1.size(0)), 2):
                new_rows_p.append(p_graph_feats_1[i] + p_graph_feats_1[j]) 
            p_graph_feats_1=torch.stack(new_rows_p).to(device)



            #attention of product on each reactant
            # p_graph_feats_1 = p_graph_feats_1.view(-1,r_graph_feats_1.shape[1])
            att_p_r=self.attention(p_graph_feats_1,r_graph_feats_1)
            att_p_r=att_p_r.squeeze(0,1)
            att_reactant = torch.sum(att_p_r,dim=0)/att_p_r.shape[0]
            att_reactant=att_reactant.view(-1).to(device)
            att_reactant_max= torch.max(att_reactant)


            #attention of reactant on each product
            att_r_p = self.attention(r_graph_feats_1, p_graph_feats_1)
            att_r_p = att_r_p.squeeze(0,1)
            att_procduct=torch.sum(att_r_p,dim=0)/att_r_p.shape[0]
            att_procduct=att_procduct.view(-1).to(device)

            ##reactant vector = sum(attention*each reactant vetor)
            reactant_tensor=torch.zeros(1,r_graph_feats_1.shape[1]).to(device)
            for idx in range(r_graph_feats_1.shape[0]):
                reactant_tensor+=att_reactant[idx]*r_graph_feats_1[idx]

            ##product vector = sum(attention*each product vector)
            product_tensor=torch.zeros(1,p_graph_feats_1.shape[1]).to(device)
            for idx in range(p_graph_feats_1.shape[0]):
                product_tensor+=att_procduct[idx]*p_graph_feats_1[idx]
            
                
            ## each reaction vector
            reaction_tensor=torch.sub(reactant_tensor,product_tensor)
            reaction_vectors=torch.cat((reaction_vectors,reaction_tensor),dim=0)
            self.atts_reactant.append(att_reactant.tolist())
            self.atts_product.append(att_procduct.tolist())



        out = self.predict(reaction_vectors)
        return out, self.atts_reactant, self.atts_product


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

    except Exception as e:
        logger.error(e)
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []
        targets = []
        preds = []
        dem=0
        # check=0
        for batchdata in tqdm(train_loader, desc="Training"):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            rsmi = batchdata[-1]
            # if dem==0:
            #     print(rsmi)
            #     dem+=1
            r_dummy=[batchdata[-3]][0]
            p_dummy=[batchdata[-4]][0]
            pred,_,_ = net(inputs_rmol, inputs_pmol, r_dummy=r_dummy,p_dummy=p_dummy,device=device)
            labels = batchdata[-2]
            

            targets.extend(labels.tolist())
            labels = labels.to(device)
            # print('labels: ',labels.shape)

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
        val_acc, val_mcc, val_loss,_,_,_ = inference(args, net, val_loader, device, loss_fn)

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
        rsmis=[]
        # atts_reactant,atts_product=[],[]
        for batchdata in tqdm(test_loader, desc="Testing"):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            r_dummy=[batchdata[-3]][0]
            p_dummy=[batchdata[-4]][0]
            # fmt: on
            pred,atts_reactant,atts_product = net(inputs_rmol, inputs_pmol, r_dummy=r_dummy,p_dummy=p_dummy,device=device)
            labels = batchdata[-2]
            rsmi=batchdata[-1]
            rsmis.append(rsmi)
            # atts_reactant.append(att_reactant)
            # atts_product.append(att_product)
            
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(torch.argmax(pred, dim=1).tolist())

            if loss_fn is not None:
                inference_loss = loss_fn(pred, labels)
                inference_loss_list.append(inference_loss.item())

    acc = accuracy_score(targets, preds)
    mcc = matthews_corrcoef(targets, preds)

    if loss_fn is None:
        return acc, mcc,atts_reactant,atts_product,rsmis, targets, preds
    else:
        return acc, mcc, np.mean(inference_loss_list),atts_reactant,atts_product,rsmis
