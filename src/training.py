import logging
import os
import time
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from validation import validation, compute_regression_metrics
from utils import setup_logging
from multi_gpu import broadcast_flag, gather_lists, is_main_process, unwrap
from visualization import (
    init_history,
    record_epoch,
    save_history,
    plot_training_history,
    set_loss_label,
)
from recipe import (
    build_loss_fn,
    build_scheduler,
    step_scheduler,
    target_scaler,
    validate_recipe_args,
)


def _train_epoch(
    net,
    train_loader,
    optimizer,
    loss_fn,
    device,
    rmol_max_cnt,
    pmol_max_cnt,
    y_mean,
    y_std,
    disable_tqdm=False,
):
    """Runs one training epoch; returns labels/preds/prediction losses."""
    net.train()
    loss_list = []
    labels = []
    preds = []

    for batchdata in tqdm(train_loader, desc="Training", disable=disable_tqdm):
        inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
        # fmt: off
        inputs_pmol = [
            b.to(device)
            for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
        ]
        r_dummy = batchdata[-4]
        p_dummy = batchdata[-3]

        pred, _ = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
        label = batchdata[-2].to(device).float()
        loss = loss_fn(pred, (label - y_mean) / y_std)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        labels.extend(label.tolist())
        preds.extend((pred.detach() * y_std + y_mean).tolist())
        loss_list.append(loss.detach().item())

    return labels, preds, loss_list


def _save_checkpoint(
    model_path,
    plain_net,
    epoch,
    val_loss,
    scheduler,
    scheduler_type,
    y_mean,
    y_std,
):
    """Writes a checkpoint with the architecture, scheduler and target scaler."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": plain_net.state_dict(),
            "model_config": plain_net.config,
            "val_loss": val_loss,
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scheduler_type": scheduler_type,
            "target_scaler": {"mean": y_mean, "std": y_std},
        },
        model_path,
    )


def _update_checkpoint(
    model_path,
    plain_net,
    epoch,
    val_loss,
    best_val_loss,
    epochs_without_improvement,
    checkpoint_select,
    scheduler,
    scheduler_kind,
    y_mean,
    y_std,
    has_validation,
):
    """Saves the checkpoint when selection requires it; updates early-stopping state."""
    improved = val_loss is not None and val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
    if checkpoint_select == "last" or improved:
        _save_checkpoint(
            model_path,
            plain_net,
            epoch,
            val_loss if val_loss is not None else best_val_loss,
            scheduler,
            scheduler_kind,
            y_mean,
            y_std,
        )
    if has_validation:
        epochs_without_improvement = (
            0 if improved else epochs_without_improvement + 1
        )
    return best_val_loss, epochs_without_improvement


def _record_history(history, epoch, train_loss, train_metrics, val=None, test=None):
    """Appends one epoch of train (and optional val/test) history."""
    history["epoch"].append(epoch)
    record_epoch(history, "train", train_loss, train_metrics)
    if val is not None:
        record_epoch(history, "val", val[0], val[1])
    if test is not None:
        record_epoch(history, "test", test[0], test[1])


def _log_validation(epoch, val_loss, val_metrics, learning_rate):
    """Logs one epoch's validation result."""
    val_r2_str = "n/a" if val_metrics["r2"] is None else "%.4f" % val_metrics["r2"]
    val_pearson_str = (
        "n/a" if val_metrics["pearson"] is None else "%.4f" % val_metrics["pearson"]
    )
    logger = logging.getLogger()
    logger.info(
        "--- validation at epoch %d, val_loss %.4f, val_mae %.4f, val_rmse %.4f, "
        "val_r2 %s, val_pearson %s, lr %.2e ---"
        % (
            epoch,
            val_loss,
            val_metrics["mae"],
            val_metrics["rmse"],
            val_r2_str,
            val_pearson_str,
            learning_rate,
        )
    )


def _log_test(epoch, test_loss, test_metrics):
    """Logs one epoch's monitoring-only test result."""
    logger = logging.getLogger()
    logger.info(
        "--- test at epoch %d, test_loss %.4f, test_mae %.4f, test_rmse %.4f, "
        "test_r2 %s, test_pearson %s ---"
        % (
            epoch,
            test_loss,
            test_metrics["mae"],
            test_metrics["rmse"],
            "n/a" if test_metrics["r2"] is None else "%.4f" % test_metrics["r2"],
            "n/a"
            if test_metrics["pearson"] is None
            else "%.4f" % test_metrics["pearson"],
        )
    )


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
    current_epoch: int = 0,
    best_val_loss: float = 1e10,
    scheduler_state: dict = None,
    test_loader=None,
    scheduler_type: str = None,
):
    """
    Trains a neural network for reaction-yield regression, monitors metrics, and saves the best model.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing configuration parameters.
    net : torch.nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set; None when the validation set was
        merged into training (no per-epoch validation).
    model_path : str
        Path to save the model checkpoint.
    device : torch.device
        Device to perform computation.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization).
    current_epoch : int, optional
        Starting epoch number, useful for resuming training (default is 0).
    best_val_loss : float, optional
        Best validation loss seen so far (default is 1e10).
    scheduler_state : dict, optional
        State of the learning-rate scheduler of an interrupted run, as stored in
        a checkpoint's "scheduler_state_dict" (default is None, i.e. a fresh one).
    test_loader : DataLoader, optional
        DataLoader for the test set; when given, test loss/metrics are tracked per epoch.
        They are only monitored and never used for checkpoint selection or early stopping.
    scheduler_type : str, optional
        Scheduler type of the checkpoint that supplied `scheduler_state`; its
        state is only restored into a scheduler of the same type.

    Returns
    -------
    dict
        Per-epoch loss and metrics for train/val/test (see `visualization.init_history`).
        Also saved to `<monitor_folder>/history.json`, with plots in `args.image_folder`.

    Notes
    -----
    When `args.patience` > 0, training stops early once the validation loss has
    not improved for `args.patience` consecutive epochs.

    With `args.lr_scheduler == "plateau"` (default) the learning rate is reduced
    by `ReduceLROnPlateau` whenever the validation loss stops improving; its
    patience is deliberately shorter than the early stopping one, so the run
    gets a smaller learning rate before it is stopped. With `"multistep"` it is
    stepped once per epoch without arguments (`MultiStepLR`), with `"none"` it
    stays constant.

    Under DistributedDataParallel every rank trains on its shard of the training
    set; only rank 0 (which may get `val_loader=None` on the other ranks) logs,
    validates and saves checkpoints, and it tells the others when to stop.
    """
    validate_recipe_args(args)
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")

    rmol_max_cnt = train_loader.dataset.rmol_max_cnt
    pmol_max_cnt = train_loader.dataset.pmol_max_cnt
    y_mean, y_std = target_scaler(args, train_loader.dataset)
    if y_std != 1.0 or y_mean != 0.0:
        logger.info(
            "--- loss is computed on z-scored targets (mean %.4f, std %.4f); "
            "MAE/RMSE/R2 stay in the original units" % (y_mean, y_std)
        )

    loss_fn = build_loss_fn(args)
    set_loss_label(getattr(args, "loss", "huber"))
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_kind = getattr(args, "lr_scheduler", "plateau")
    scheduler = build_scheduler(
        args, optimizer, scheduler_state=scheduler_state, checkpoint_type=scheduler_type
    )
    checkpoint_select = getattr(args, "checkpoint_select", "best_val")
    if checkpoint_select == "last" and scheduler_kind == "plateau":
        # Validation is monitoring only with last-epoch selection, so it must
        # not drive ReduceLROnPlateau either.
        logger.info(
            "--- --checkpoint_select last: validation is monitoring only, so "
            "the plateau scheduler is not stepped (the learning rate stays fixed)"
        )
    history = init_history()
    patience = getattr(args, "patience", 0)
    epochs_without_improvement = 0
    is_main = is_main_process()
    plain_net = unwrap(net)

    for epoch in range(epochs):
        start_time = time.time()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch + current_epoch)

        labels, preds, train_loss_list = _train_epoch(
            net,
            train_loader,
            optimizer,
            loss_fn,
            device,
            rmol_max_cnt,
            pmol_max_cnt,
            y_mean,
            y_std,
            disable_tqdm=(not is_main) or bool(os.environ.get("TQDM_DISABLE")),
        )

        # under DDP, score the whole training set rather than this rank's shard
        labels, preds, train_loss_list = gather_lists(labels, preds, train_loss_list)
        if not is_main:
            if broadcast_flag(False, device):
                break
            continue

        train_metrics = compute_regression_metrics(labels, preds)
        logger.info(
            "--- training epoch %d, loss %.4f, mae %.4f, rmse %.4f, time elapsed(min) %.2f---"
            % (
                epoch,
                np.mean(train_loss_list),
                train_metrics["mae"],
                train_metrics["rmse"],
                (time.time() - start_time) / 60,
            )
        )

        # validation: skipped when the validation set was merged into training
        net.eval()
        val_metrics, val_loss = None, None
        if val_loader is not None:
            val_metrics, val_loss = validation(
                args, plain_net, val_loader, device, loss_fn, y_mean, y_std
            )
            _log_validation(
                epoch, val_loss, val_metrics, optimizer.param_groups[0]["lr"]
            )

        test_metrics, test_loss = None, None
        if test_loader is not None:
            test_metrics, test_loss = validation(
                args, plain_net, test_loader, device, loss_fn, y_mean, y_std
            )
            _log_test(epoch, test_loss, test_metrics)
        logger.info("\n" + "*" * 100)

        # Once per epoch, on the same validation loss the early stopping watches.
        # With last-epoch selection, validation must not change the LR either.
        scheduler_watch_loss = None if checkpoint_select == "last" else val_loss
        step_scheduler(scheduler, scheduler_kind, scheduler_watch_loss)

        best_val_loss, epochs_without_improvement = _update_checkpoint(
            model_path,
            plain_net,
            epoch + current_epoch,
            val_loss,
            best_val_loss,
            epochs_without_improvement,
            checkpoint_select,
            scheduler,
            scheduler_kind,
            y_mean,
            y_std,
            val_loader is not None,
        )

        _record_history(
            history,
            epoch + current_epoch,
            np.mean(train_loss_list),
            train_metrics,
            val=None if val_metrics is None else (val_loss, val_metrics),
            test=None if test_metrics is None else (test_loss, test_metrics),
        )
        # Rewritten every epoch so curves are available even if training is interrupted.
        save_history(history, args.monitor_folder + "history.json")
        plot_training_history(history, args.image_folder)

        stop = patience > 0 and epochs_without_improvement >= patience
        if broadcast_flag(stop, device):
            logger.info(
                "--- early stopping at epoch %d: val_loss did not improve for %d epochs ---"
                % (epoch + current_epoch, patience)
            )
            break

    return history
