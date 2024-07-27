import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from gin import GIN

class recat(nn.Module):
    def __init__(self, node_in_feats=155, edge_in_feats=9, num_layer=3, emb_dim=1024, predict_hidden_feats=512, out_dim=4, JK ='sum',drop_ratio=0.1,gnn_type='gin'):
        super(recat, self).__init__()
        self.gnn = GIN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            torch.nn.Linear(emb_dim, predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, out_dim),
        )

    
    def forward(self, rmols, pmols):
        r_graph_feats = torch.sum(torch.stack([self.gnn(rmol) for rmol in rmols]), dim=0)
        p_graph_feats = torch.sum(torch.stack([self.gnn(pmol) for pmol in pmols]), dim=0)


        react_graph_feats= torch.sub(r_graph_feats, p_graph_feats)
        out = self.predict(react_graph_feats)
        return out
def train(net, train_loader, val_loader, model_path, n_epochs=20):
    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt

    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    best_val_loss=1e10

    for epoch in range(n_epochs):
        #training
        net.train()
        start_time= time.time()

        train_loss_list=[]
        targets=[]
        preds=[]

        for batchdata in tqdm(train_loader, desc='Training'):
            inputs_rmol= [b.to(device) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol=[b.to(device) for b in batchdata[rmol_max_cnt: rmol_max_cnt+pmol_max_cnt]]

            labels=batchdata[-1]
            targets.extend(labels.tolist())
            labels=labels.to(device)

            pred=net(inputs_rmol,inputs_pmol)
            preds.extend(torch.argmax(pred, dim=1).tolist())
            loss=loss_fn(pred,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss=loss.detach().item()
            train_loss_list.append(train_loss)
        
        acc=accuracy_score(targets, preds)
        mcc=matthews_corrcoef(targets, preds)
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

        #validation
        net.eval()
        val_acc, val_mcc, val_loss= inference(net, val_loader, loss_fn)

        print(
            "--- validation at epoch %d, val_loss %.3f, val_acc %.3f, val_mcc %.3f ---"
            % (epoch, val_loss,val_acc,val_mcc)
        )
        print('\n'+'*'*100)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), model_path)



def inference(net, test_loader,loss_fn=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=test_loader.batch_size

    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    net.eval()
    inference_loss_list=[]
    preds=[]
    targets=[]

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc='Testing'):
            inputs_rmols = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            inputs_pmols = [b.to(device) for b in batchdata[rmol_max_cnt: rmol_max_cnt+pmol_max_cnt]]

            labels=batchdata[-1]
            targets.extend(labels.tolist())
            labels=labels.to(device)

            pred=net(inputs_rmols, inputs_pmols)
            preds.extend(torch.argmax(pred, dim=1).tolist())

            if loss_fn is not None:
                inference_loss=loss_fn(pred, labels)
                inference_loss_list.append(inference_loss.item())

    acc=accuracy_score(targets, preds)
    mcc=matthews_corrcoef(targets, preds)

    if loss_fn is None:
        return acc, mcc
    else:
        return acc, mcc, np.mean(inference_loss_list)


            

    

