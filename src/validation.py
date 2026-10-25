import os
import numpy as np
import torch
from tqdm import tqdm

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except (ImportError, Exception):
    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def mean_squared_error(y_true, y_pred):
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return float("nan")
        return float(1.0 - (ss_res / ss_tot))

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


def validation(args, net, test_loader, device, loss_fn=None, y_mean=0.0, y_std=1.0):
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
    y_mean, y_std : float, optional
        Target standardizer of the training recipe. The loss is computed on
        standardized labels; predictions are mapped back to the original units
        before metrics, labels and predictions are returned (identity by default).

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
        disable_tqdm = bool(os.environ.get("TQDM_DISABLE"))
        for batchdata in tqdm(test_loader, desc=name_process, disable=disable_tqdm):
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
                scaled_label = (label - y_mean) / y_std
                inference_loss = loss_fn(pred, scaled_label)
                inference_loss_list.append(inference_loss.item())

            labels.extend(label.tolist())
            preds.extend((pred * y_std + y_mean).tolist())
            rsmis.append(batchdata[-1])

    metrics = compute_regression_metrics(labels, preds)

    if loss_fn is None:
        return metrics, rsmis, labels, preds, emb
    else:
        return metrics, np.mean(inference_loss_list)


def enable_mc_dropout(net) -> None:
    """
    Keeps every non-dropout module in eval mode and switches dropout modules
    back to train mode, as reaction_yield_nn's `MC_dropout` does.
    """
    net.eval()
    for module in net.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.train()


def predict_mc(
    args, net, test_loader, device, passes: int, y_mean: float = 0.0, y_std: float = 1.0
):
    """
    Mean of `passes` MC-dropout forward passes per batch.

    Returns ``(metrics, labels, preds)`` with predictions de-standardized to the
    original target units. The model is left in eval mode.
    """
    if passes <= 0:
        raise ValueError("predict_mc needs passes > 0")

    rmol_max_cnt = test_loader.dataset.rmol_max_cnt
    pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    labels = []
    preds = []
    enable_mc_dropout(net)
    try:
        with torch.no_grad():
            disable_tqdm = bool(os.environ.get("TQDM_DISABLE"))
            for batchdata in tqdm(
                test_loader, desc="MC_validation", disable=disable_tqdm
            ):
                inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
                # fmt: off
                inputs_pmol = [
                    b.to(device)
                    for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
                ]
                r_dummy = batchdata[-4]
                p_dummy = batchdata[-3]

                pass_preds = []
                for _ in range(passes):
                    pred, _ = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
                    pass_preds.append(pred)
                pred = torch.stack(pass_preds, dim=0).mean(dim=0)

                label = batchdata[-2].to(device).float()
                labels.extend(label.tolist())
                preds.extend((pred * y_std + y_mean).tolist())
    finally:
        # Always leave the model in full evaluation mode for later callers.
        net.eval()

    return compute_regression_metrics(labels, preds), labels, preds


def clip_yields(preds, y_max: float) -> list:
    """Clips predicted yields to the physical ``[0, y_max]`` range."""
    return np.clip(np.asarray(preds, dtype=float), 0.0, float(y_max)).tolist()


def evaluate_test_predictions(
    args,
    net,
    test_loader,
    device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    y_max: float = None,
):
    """
    Final test evaluation: one deterministic pass plus optional MC dropout and
    clipping.

    The deterministic pass is an inference pass (no loss), so `validation` is
    called in its external-validation mode and returns metrics plus predictions.
    Returns a dict with the headline ``metrics``/``preds`` (MC + clip when
    enabled), the deterministic ``det_metrics``/``det_preds``, ``labels``,
    ``rsmis`` and the reaction ``emb``eddings. When MC dropout and clipping are
    both off, the headline predictions are exactly the deterministic ones.
    """
    det_metrics, rsmis, labels, det_preds, emb = validation(
        args, net, test_loader, device, y_mean=y_mean, y_std=y_std
    )

    passes = int(getattr(args, "mc_dropout_passes", 0))
    if passes > 0:
        _, _, preds = predict_mc(
            args, net, test_loader, device, passes, y_mean, y_std
        )
    else:
        preds = list(det_preds)

    if getattr(args, "clip_predictions", False):
        if y_max is None:
            y_max = 100.0 if np.nanmax(np.abs(np.asarray(labels))) > 1.5 else 1.0
        preds = clip_yields(preds, y_max)

    metrics = compute_regression_metrics(labels, preds)
    return {
        "metrics": metrics,
        "preds": preds,
        "labels": labels,
        "det_metrics": det_metrics,
        "det_preds": list(det_preds),
        "rsmis": rsmis,
        "emb": emb,
    }
