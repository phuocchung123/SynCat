import numpy as np
import torch
from tqdm import tqdm
# from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def validation(args, net, test_loader, device, loss_fn=None):
    """
    Runs model inference on the test set, computes metrics, and optionally returns attention and embeddings.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing configuration parameters.
    net : torch.nn.Module
        The trained model to evaluate.
    test_loader : DataLoader
        DataLoader for the test or validation set.
    device : torch.device
        Device to run the computations.
    loss_fn : callable, optional
        Loss function for evaluating inference loss (default is None, for external validation).

    Returns
    -------
    tuple
        If loss_fn is None (external validation), returns:
            (accuracy, mcc, att_r, att_p, rsmis, labels, preds, emb)
        If loss_fn is given (internal validation), returns:
            (accuracy, mcc, mean_inference_loss)
    """

    rmol_max_cnt = test_loader.dataset.rmol_max_cnt
    pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    net.eval()
    inference_loss_list = []
    preds = []
    labels = []
    rsmis = []
    if loss_fn is None:
        name_process = "External_validation"
    else:
        name_process = "Internal_validation"

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc=name_process):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            # fmt: off
            inputs_pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            r_dummy = batchdata[-4]
            p_dummy = batchdata[-3]

            pred, att_r, att_p, emb = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
            pred = pred.float()
            label = batchdata[-2].float()

            if loss_fn is not None:
                inference_loss = loss_fn(pred, label.to(device))
                inference_loss_list.append(inference_loss.item())

            preds.extend(pred.detach().cpu().tolist())
            labels.extend(label.detach().cpu().tolist())

            rsmis.append(batchdata[-1])

    # acc = accuracy_score(labels, preds)
    # mcc = matthews_corrcoef(labels, preds)

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = mean_squared_error(labels, preds) ** 0.5
    r2 = r2_score(labels, preds)

    if loss_fn is None:
        return mse, mae, rmse, r2, att_r, att_p, rsmis, labels, preds, emb
    else:
        return mse, mae, rmse, r2, np.mean(inference_loss_list)
