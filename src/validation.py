import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef

def validation(args, net, test_loader, device,loss_fn=None):

    rmol_max_cnt = test_loader.dataset.rmol_max_cnt
    pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    net.eval()
    inference_loss_list = []
    preds = []
    labels = []
    rsmis=[]
    if loss_fn is None:
        name_process='External_validation'
    else:
        name_process='Internal_validation'

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc=name_process):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            r_dummy=batchdata[-4]
            p_dummy=batchdata[-3]

            pred,att_r, att_p, emb = net(inputs_rmol, inputs_pmol,r_dummy,p_dummy,device)
            label = batchdata[-2]
            label = label.to(device)
          
            if loss_fn is not None:
                inference_loss = loss_fn(pred, label)
                inference_loss_list.append(inference_loss.item())

            labels.extend(label.tolist())
            preds.extend(torch.argmax(pred, dim=1).tolist())
            rsmis.append(batchdata[-1])



    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    if loss_fn is None:
        return acc, mcc, att_r, att_p, rsmis, labels, preds, emb
    else:
        return acc, mcc, np.mean(inference_loss_list)