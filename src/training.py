import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from validation import validation
from utils import setup_logging
# from sklearn.metrics import accuracy_score, 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


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
):
    """
    Trains a neural network for reaction classification, monitors metrics, and saves the best model.

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

    Returns
    -------
    torch.nn.Module
        The trained neural network model.
    """
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")

    rmol_max_cnt = train_loader.dataset.rmol_max_cnt
    pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize and fit StandardScaler for labels
    scaler = StandardScaler()
    all_train_labels = []
    for batchdata in train_loader:
        all_train_labels.extend(batchdata[-2].tolist())
    scaler.fit(np.array(all_train_labels).reshape(-1, 1))

    # Initialize history dictionary to store metrics for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
        'train_mae': [], 'val_mae': [],
        'train_rmse': [], 'val_rmse': [],
        'train_r2': [], 'val_r2': []
    }

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

            pred, _, _, _ = net(inputs_rmol, inputs_pmol, r_dummy, p_dummy, device)
            pred = pred.float()
            label = batchdata[-2].float()

            # Scale labels for loss calculation
            scaled_label_np = scaler.transform(label.numpy().reshape(-1, 1)).flatten()
            scaled_label = torch.tensor(scaled_label_np, dtype=torch.float32, device=device)
            loss = loss_fn(pred, scaled_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Inverse transform predictions for metrics against original labels
            pred_inv = scaler.inverse_transform(pred.detach().cpu().numpy().reshape(-1, 1)).flatten()
            preds.extend(pred_inv.tolist())
            labels.extend(label.detach().cpu().tolist())
            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        # acc = accuracy_score(labels, preds)
        # mcc = matthews_corrcoef(labels, preds)

        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        rmse = mean_squared_error(labels, preds) ** 0.5
        r2 = r2_score(labels, preds)

        logger.info(
            "--- training epoch %d, loss %.3f, mse %.3f, mae %.3f, rmse %.3f, r2 %.3f, time elapsed(min) %.2f---"
            % (
                epoch,
                np.mean(train_loss_list),
                mse,
                mae,
                rmse,
                r2,
                (time.time() - start_time) / 60,
            )
        )

        # validation
        net.eval()
        val_mse, val_mae, val_rmse, val_r2, val_loss = validation(args, net, val_loader, device, loss_fn, scaler)

        # Record metrics for plotting
        history['train_loss'].append(np.mean(train_loss_list))
        history['train_mse'].append(mse)
        history['train_mae'].append(mae)
        history['train_rmse'].append(rmse)
        history['train_r2'].append(r2)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        logger.info(
            "--- validation at epoch %d, val_loss %.3f, val_mse %.3f, val_mae %.3f, val_rmse %.3f, val_r2 %.3f ---"
            % (epoch, val_loss, val_mse, val_mae, val_rmse, val_r2)
        )
        logger.info("\n" + "*" * 100)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + current_epoch,
                    "model_state_dict": net.state_dict(),
                    "val_loss": best_val_loss,
                    "scaler": scaler,
                },
                model_path,
            )

    # After training completes, plot and save the learning curves
    plot_dir = os.path.join(args.Data_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics_to_plot = [
        ('loss', 'Loss'),
        ('mse', 'MSE'),
        ('mae', 'MAE'),
        ('rmse', 'RMSE'),
        ('r2', 'R2 Score')
    ]

    for key, title in metrics_to_plot:
        plt.figure()
        plt.plot(history[f'train_{key}'], label=f'Train {title}')
        plt.plot(history[f'val_{key}'], label=f'Validation {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'{title} over Epochs')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plot_dir, f'{key}_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved {title} curve to {plot_path}")

    return net
