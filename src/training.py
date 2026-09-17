import time
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from validation import validation, compute_regression_metrics
from utils import setup_logging
from visualization import init_history, record_epoch, save_history, plot_training_history


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
    test_loader=None,
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
        DataLoader for the validation set.
    model_path : str
        Path to save the best model checkpoint.
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
    test_loader : DataLoader, optional
        DataLoader for the test set; when given, test loss/metrics are tracked per epoch.
        They are only monitored and never used for checkpoint selection or early stopping.

    Returns
    -------
    dict
        Per-epoch loss and metrics for train/val/test (see `visualization.init_history`).
        Also saved to `<monitor_folder>/history.json`, with plots in `args.image_folder`.

    Notes
    -----
    When `args.patience` > 0, training stops early once the validation loss has
    not improved for `args.patience` consecutive epochs.
    """
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")

    rmol_max_cnt = train_loader.dataset.rmol_max_cnt
    pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = torch.nn.HuberLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = init_history()
    patience = getattr(args, "patience", 0)
    epochs_without_improvement = 0

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
            r_dummy = batchdata[-4]
            p_dummy = batchdata[-3]

            pred, _ = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
            label = batchdata[-2]
            label = label.to(device).float()
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels.extend(label.tolist())
            preds.extend(pred.detach().tolist())
            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

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

        # validation
        net.eval()
        val_metrics, val_loss = validation(args, net, val_loader, device, loss_fn)

        val_r2_str = "n/a" if val_metrics["r2"] is None else "%.4f" % val_metrics["r2"]
        val_pearson_str = (
            "n/a" if val_metrics["pearson"] is None else "%.4f" % val_metrics["pearson"]
        )
        logger.info(
            "--- validation at epoch %d, val_loss %.4f, val_mae %.4f, val_rmse %.4f, "
            "val_r2 %s, val_pearson %s ---"
            % (
                epoch,
                val_loss,
                val_metrics["mae"],
                val_metrics["rmse"],
                val_r2_str,
                val_pearson_str,
            )
        )

        if test_loader is not None:
            test_metrics, test_loss = validation(args, net, test_loader, device, loss_fn)
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["epoch"].append(epoch + current_epoch)
        record_epoch(history, "train", np.mean(train_loss_list), train_metrics)
        record_epoch(history, "val", val_loss, val_metrics)
        if test_loader is not None:
            record_epoch(history, "test", test_loss, test_metrics)
        # Rewritten every epoch so curves are available even if training is interrupted.
        save_history(history, args.monitor_folder + "history.json")
        plot_training_history(history, args.image_folder)

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "--- early stopping at epoch %d: val_loss did not improve for %d epochs ---"
                % (epoch + current_epoch, patience)
            )
            break

    return history
