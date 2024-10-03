import pandas as pd
import os
import numpy as np


def create_test_round_df(args, roc_list, ap_list, f1_list, task_type, training_round):
    """
    Creates and saves a test metrics DataFrame for various datasets.

    Args:
    args: Argument parser or a similar object with attributes dataset, save_path, and task_type.
    roc_list (list): List of ROC AUC values for individual task.
    ap_list (list): List of average precision (AP) values for individual task.
    f1_list (list): List of F1 scores for individual task.
    task_type (str): The type of task for which metrics are being recorded.
    training_round (int): The identifier for the training round.
    """
    # Construct the file path using os.path.joi
    file_path = os.path.join(
        args.save_path,
        task_type,
        args.dataset,
        f"test_metrics_round_{training_round}.csv",
    )

    # Create a DataFrame and save it to CSV in one step
    test_metric_df = pd.DataFrame(
        {"AUC": [np.mean(roc_list)], "AP": [np.mean(ap_list)], "F1": [np.mean(f1_list)]}
    )
    test_metric_df.to_csv(file_path, index=False)


def create_train_round_df(
    args,
    train_df,
    train_loss,
    train_auc,
    train_ap,
    train_f1,
    val_loss,
    val_auc,
    val_ap,
    val_f1,
    test_loss,
    test_auc,
    test_ap,
    test_f1,
    task_type,
    epoch,
    training_round,
):
    """
    Updates and saves the training metrics DataFrame with new epoch data.

    Args:
    args: Argument parser or a similar object with attributes save_path, task_type, and dataset.
    train_df (DataFrame): The DataFrame containing the training metrics.
    train_loss, train_auc, train_ap, train_f1 (float): Training metrics for the current epoch.
    val_loss, val_auc, val_ap, val_f1 (float): Validation metrics for the current epoch.
    test_loss, test_auc, test_ap, test_f1 (float): Test metrics for the current epoch.
    task_type (str): The type of task for which metrics are being recorded.
    epoch (int): The current epoch number.
    training_round (int): Identifier for the training round.
    """
    # Update the DataFrame with the new metrics in a single operation
    new_data = {
        "train_loss": train_loss,
        "train_auc": train_auc,
        "train_ap": train_ap,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "val_f1": val_f1,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_f1": test_f1,
    }
    train_df.loc[epoch - 1] = new_data
    file_path = os.path.join(
        args.save_path,
        task_type,
        args.dataset,
        f"train_metrics_round_{training_round}.csv",
    )
    train_df.to_csv(file_path, index=False)


def average_train_metrics(args, task_type, remove=False):
    """
    Reads multiple training metrics CSV files, calculates their average, and optionally removes the original files.

    Args:
    args (object): An object containing configuration like save_path, dataset, and training_rounds.
    task_type (str): The type of task for which metrics are being aggregated.
    remove (bool): Whether to remove the original CSV files after processing.

    Returns:
    pd.DataFrame: A DataFrame containing the average of the metrics.
    """
    dataframes = []
    base_path = os.path.join(args.save_path, task_type, args.dataset)

    for i in range(1, args.training_rounds + 1):
        file_path = os.path.join(base_path, f"train_metrics_round_{i}.csv")
        try:
            round_metrics = pd.read_csv(file_path)
            dataframes.append(round_metrics)
            if remove:
                os.remove(file_path)  # Safely remove the file
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Concatenate all DataFrames at once
    df_concatenated = pd.concat(dataframes, axis=0)
    df_avg = df_concatenated.groupby(df_concatenated.index).mean()

    # Save the averaged metrics
    output_file = os.path.join(base_path, "train_metrics.csv")
    df_avg.to_csv(output_file, index=False)

    return df_avg


def average_test_metrics(args, task_type, remove=True):
    """
    Reads multiple test metrics CSV files, calculates their average and standard deviation,
    and optionally removes the original files.

    Args:
    args (object): An object containing configuration like save_path, dataset, and training_rounds.
    task_type (str): The type of task for which metrics are being aggregated.
    remove (bool): Whether to remove the original CSV files after processing.

    Returns:
    None: Outputs the AUC test metrics directly to the console.
    """
    dataframes = []
    base_path = os.path.join(args.save_path, task_type, args.dataset)

    for i in range(1, args.training_rounds + 1):
        file_path = os.path.join(base_path, f"test_metrics_round_{i}.csv")
        try:
            round_metrics = pd.read_csv(file_path)
            dataframes.append(round_metrics)
            if remove:
                os.remove(file_path)  # Safely remove the file
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Concatenate all DataFrames at once
    df_concatenated = pd.concat(dataframes, axis=0)
    df_avg = df_concatenated.groupby(df_concatenated.index).mean()
    df_std = df_concatenated.groupby(df_concatenated.index).std()

    # Save the average and standard deviation metrics
    df_avg.to_csv(os.path.join(base_path, "test_metrics_avg.csv"), index=False)
    df_std.to_csv(os.path.join(base_path, "test_metrics_std.csv"), index=False)

    # Print the results for AUC
    mean_auc = df_avg["AUC"].mean() * 100
    std_auc = df_std["AUC"].mean() * 100
    print(
        f"AUC test for {args.dataset} dataset over {args.training_rounds} training rounds: {mean_auc:.2f}±{std_auc:.2f}"
    )
