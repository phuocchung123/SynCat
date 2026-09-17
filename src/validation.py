import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def compute_regression_metrics(labels: list, preds: list) -> dict:
    """
    Computes regression metrics between ground-truth yields and predicted yields.

    Handles constant or very small evaluation sets safely: R-squared and Pearson
    correlation are undefined when there are fewer than 2 samples or when the
    ground-truth values are constant (zero variance), in which case they are
    reported as `None` instead of raising an error.

    Parameters
    ----------
    labels : list
        Ground-truth yield values.
    preds : list
        Predicted yield values.

    Returns
    -------
    dict
        Dictionary with keys "mae", "rmse", "r2", and "pearson". "r2" and
        "pearson" may be `None` when undefined for the given data.
    """
    labels_arr = np.asarray(labels, dtype=float)
    preds_arr = np.asarray(preds, dtype=float)

    mae = mean_absolute_error(labels_arr, preds_arr)
    rmse = mean_squared_error(labels_arr, preds_arr) ** 0.5

    if len(labels_arr) < 2:
        r2 = None
        pearson = None
    elif np.isclose(np.std(labels_arr), 0.0):
        # R2 and Pearson correlation are undefined when the target has no variance.
        r2 = None
        pearson = None
    else:
        r2 = r2_score(labels_arr, preds_arr)
        pearson, _ = pearsonr(labels_arr, preds_arr)

    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}


def validation(args, net, test_loader, device, loss_fn=None):
    """
    Runs model inference on the test set, computes regression metrics, and
    optionally returns the reaction embeddings.

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
            (metrics, rsmis, labels, preds, emb)
        If loss_fn is given (internal validation), returns:
            (metrics, mean_inference_loss)
        `metrics` is the dict returned by `compute_regression_metrics`.
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

            pred, emb = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
            label = batchdata[-2]
            label = label.to(device).float()
            if loss_fn is not None:
                inference_loss = loss_fn(pred, label)
                inference_loss_list.append(inference_loss.item())

            labels.extend(label.tolist())
            preds.extend(pred.tolist())
            rsmis.append(batchdata[-1])

    metrics = compute_regression_metrics(labels, preds)

    if loss_fn is None:
        return metrics, rsmis, labels, preds, emb
    else:
        return metrics, np.mean(inference_loss_list)
