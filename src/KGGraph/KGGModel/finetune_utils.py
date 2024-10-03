import sys
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGModel.crawl_metrics import (
    create_test_round_df,
    create_train_round_df,
)
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    average_precision_score,
)
import pandas as pd


def get_task_type(args):
    """
    Determines the type of task (classification or regression) based on the dataset.

    Args:
    args: An argument parser object or a similar structure where args.dataset is the name of the dataset.

    Returns:
    str: The type of task associated with the dataset ('classification' or 'regression').
    """
    # List of datasets associated with classification tasks
    classification_datasets = ["tox21", "bace", "bbbp", "toxcast", "sider", "clintox"]

    if args.dataset in classification_datasets:
        return "classification"
    else:
        return "regression"


def get_num_task(args):
    """
    Retrieves the number of tasks associated with a specific dataset.

    Args:
    args: An argument parser object or a similar structure where args.dataset is the name of the dataset.

    Returns:
    int: The number of tasks associated with the dataset.
    """
    # Define the number of tasks for each dataset
    num_tasks_dict = {
        "tox21": 12,
        "bace": 1,
        "bbbp": 1,
        "toxcast": 617,
        "sider": 27,
        "clintox": 2,
        "esol": 1,
        "freesolv": 1,
        "lipophilicity": 1,
        "qm7": 1,
        "qm8": 12,
        "qm9": 12,
    }

    # Get the number of tasks based on the dataset
    num_tasks = num_tasks_dict.get(args.dataset)

    if num_tasks is None:
        raise ValueError("Invalid dataset name.")

    return num_tasks


def train(model, device, loader, optimizer, criterion):
    """
    Trains the model for one epoch over the provided data loader.

    Args:
    model (torch.nn.Module): The model to be trained.
    device (torch.device): The device to run the model on ('cuda' or 'cpu').
    loader (DataLoader): The data loader that provides batches of data.
    optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
    criterion (function): The loss function used for training.

    Returns:
    tuple: A tuple containing the average loss, ROC AUC, average precision, and F1 score for the training epoch.
    """
    model.train()
    y_true = []
    y_scores = []
    y_pred_labels = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y**2 > 0

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred.detach())
        y_pred_labels.append(pred.detach() >= 0.5)

        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
        )

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_pred_labels = torch.cat(y_pred_labels, dim=0).cpu().numpy()
    roc_list = []
    ap_list = []
    f1_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(
                roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i])
            )
            ap_list.append(
                average_precision_score(
                    (y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]
                )
            )
            f1_list.append(
                f1_score((y_true[is_valid, i] + 1) / 2, y_pred_labels[is_valid, i])
            )

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    train_roc = sum(roc_list) / len(roc_list)
    train_ap = sum(ap_list) / len(ap_list)
    train_f1 = sum(f1_list) / len(f1_list)

    return loss, train_roc, train_ap, train_f1


def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        if args.dataset in ["qm7", "qm8", "qm9"]:
            loss = torch.sum(torch.abs(pred - y)) / y.size(0)
        elif args.dataset in ["esol", "freesolv", "lipophilicity"]:
            loss = torch.sum((pred - y) ** 2) / y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(args, model, device, loader, task_type, criterion):
    """
    Evaluates the performance of a model on a given dataset.

    Args:
    - args: The command-line arguments passed to the program.
    - model: The model to be evaluated.
    - device: The device on which the model is running.
    - loader: The data loader for the dataset.
    - task_type: The type of task being performed.

    Returns:
    - eval_roc: The area under the ROC curve for the evaluation.
    - eval_ap: The average precision for the evaluation.
    - eval_f1: The F1 score for the evaluation.
    - loss: The average loss for the evaluation.
    - roc_list: A list of ROC scores for each target.
    - ap_list: A list of average precisions for each target.
    - f1_list: A list of F1 scores for each target.
    """
    model.eval()
    y_true = []
    y_scores = []
    y_pred_labels = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        y_pred_labels.append(pred >= 0.5)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_pred_labels = torch.cat(y_pred_labels, dim=0).cpu().numpy()
    # Whether y is non-null or not.
    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y**2 > 0
    # Loss matrix
    loss_mat = criterion(pred.double(), (y + 1) / 2)
    # loss matrix after removing null target
    loss_mat = torch.where(
        is_valid,
        loss_mat,
        torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
    )
    loss = torch.sum(loss_mat) / torch.sum(is_valid)

    roc_list = []
    ap_list = []
    f1_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(
                roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i])
            )
            ap_list.append(
                average_precision_score(
                    (y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]
                )
            )
            f1_list.append(
                f1_score((y_true[is_valid, i] + 1) / 2, y_pred_labels[is_valid, i])
            )

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    eval_roc = sum(roc_list) / len(roc_list)
    eval_ap = sum(ap_list) / len(ap_list)
    eval_f1 = sum(f1_list) / len(f1_list)

    return eval_roc, eval_ap, eval_f1, loss, roc_list, ap_list, f1_list


def eval_reg(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse = np.sqrt(mean_squared_error(y_true, y_scores))
    return mse, mae, rmse


def save_emb(model, device, loader, num_tasks, out_file):
    model.eval()

    emb, label = [], []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        graph_emb = (
            model.graph_emb(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            .cpu()
            .detach()
            .numpy()
        )
        y = batch.y.view(-1, num_tasks).cpu().detach().numpy()
        emb.append(graph_emb)
        label.append(y)
    output_emb = np.row_stack(emb)
    output_label = np.row_stack(label)

    np.savez(out_file, emb=output_emb, label=output_label)


def train_epoch_cls(
    args,
    model,
    device,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    task_type,
    training_round,
):
    """
    Trains a classification model for a specified number of epochs and returns the training metrics.

    Args:
        args (object): The arguments object containing the necessary parameters for training.
        model (object): The classification model to be trained.
        device (str): The device on which the training will be performed.
        train_loader (object): The data loader for the training dataset.
        val_loader (object): The data loader for the validation dataset.
        test_loader (object): The data loader for the test dataset.
        optimizer (object): The optimizer used for training.
        task_type (str): The type of task (e.g., binary classification, multi-class classification).
        training_rounds (int): The number of rounds of training to be performed.

    Returns:
        dict: A dictionary containing the training metrics for each epoch, including the training and validation loss, AUC, AP, and F1 scores.
    """
    columns = [
        "train_loss",
        "train_auc",
        "train_ap",
        "train_f1",
        "val_loss",
        "val_auc",
        "val_ap",
        "val_f1",
        "test_loss",
        "test_auc",
        "test_ap",
        "test_f1",
    ]
    train_df = pd.DataFrame(columns=columns, index=range(args.epochs))

    for epoch in range(1, args.epochs + 1):
        print("====epoch:", epoch)

        train_loss, train_auc, train_ap, train_f1 = train(
            model, device, train_loader, optimizer, criterion
        )

        print("====Evaluation")

        val_auc, val_ap, val_f1, val_loss, _, _, _ = evaluate(
            args, model, device, val_loader, task_type, criterion
        )
        test_auc, test_ap, test_f1, test_loss, roc_list, ap_list, f1_list = evaluate(
            args, model, device, test_loader, task_type, criterion
        )

        create_train_round_df(
            args,
            train_df,
            float(train_loss),
            train_auc,
            train_ap,
            train_f1,
            float(val_loss),
            val_auc,
            val_ap,
            val_f1,
            float(test_loss),
            test_auc,
            test_ap,
            test_f1,
            task_type,
            epoch,
            training_round,
        )

        create_test_round_df(
            args, roc_list, ap_list, f1_list, task_type, training_round
        )
        torch.save(
            model.state_dict(),
            f"{args.save_path+task_type}/{args.dataset}/{args.dataset}_{training_round}.pth",
        )

        print(
            "train_loss: %f val_loss: %f test_loss: %f"
            % (train_loss, val_loss, test_loss)
        )
        print("train_auc: %f val_auc: %f test_auc: %f" % (train_auc, val_auc, test_auc))
        print("train_ap: %f val_ap: %f test_ap: %f" % (train_ap, val_ap, test_ap))
        print("train_f1: %f val_f1: %f test_f1: %f" % (train_f1, val_f1, test_f1))


def train_epoch_reg(
    args,
    model,
    device,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    model_save_path,
):
    train_list, test_list = [], []
    for epoch in range(1, args.epochs + 1):
        print("====epoch:", epoch)

        train_reg(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_mse, train_mae, train_rmse = eval_reg(
                args, model, device, train_loader
            )
        else:
            print("omit the training accuracy computation")
            train_mse, train_mae, train_rmse = 0, 0, 0
        val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
        test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)

        if args.dataset in ["esol", "freesolv", "lipophilicity"]:
            test_list.append(float("{:.6f}".format(test_rmse)))
            train_list.append(float("{:.6f}".format(train_rmse)))
            torch.save(model.state_dict(), model_save_path)

        elif args.dataset in ["qm7", "qm8", "qm9"]:
            test_list.append(float("{:.6f}".format(test_mae)))
            train_list.append(float("{:.6f}".format(train_mae)))
            torch.save(model.state_dict(), model_save_path)

        print("train_mse: %f val_mse: %f test_mse: %f" % (train_mse, val_mse, test_mse))
        print("train_mae: %f val_mae: %f test_mae: %f" % (train_mae, val_mae, test_mae))
        print(
            "train_rmse: %f val_rmse: %f test_rmse: %f"
            % (train_rmse, val_rmse, test_rmse)
        )

    return test_list
