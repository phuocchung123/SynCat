import os
import matplotlib.pyplot as plt


def plot_metrics(args, df, task_type):
    """
    Plot the training, validation, and test loss, AUC, F1, and AP for each epoch.

    Args:
    args: Argument parser or a similar object with attributes like save_path and dataset.
    df (dict): Dataframe containing lists of training, validation, and test metrics.
    task_type (str): The type of task (e.g., 'classification', 'regression').

    The function saves the plot to a file and displays it.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot loss
    axs[0, 0].plot(df["train_loss"], label="Train loss")
    axs[0, 0].plot(df["val_loss"], label="Val loss")
    axs[0, 0].plot(df["test_loss"], label="Test loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].legend()

    # Plot AUC
    axs[0, 1].plot(df["train_auc"], label="Train AUC")
    axs[0, 1].plot(df["val_auc"], label="Val AUC")
    axs[0, 1].plot(df["test_auc"], label="Test AUC")
    axs[0, 1].set_title("AUC")
    axs[0, 1].legend()

    # Plot F1
    axs[1, 0].plot(df["train_f1"], label="Train F1")
    axs[1, 0].plot(df["val_f1"], label="Val F1")
    axs[1, 0].plot(df["test_f1"], label="Test F1")
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].legend()

    # Plot AP
    axs[1, 1].plot(df["train_ap"], label="Train AP")
    axs[1, 1].plot(df["val_ap"], label="Val AP")
    axs[1, 1].plot(df["test_ap"], label="Test AP")
    axs[1, 1].set_title("Average Precision")
    axs[1, 1].legend()

    # Setting labels for all subplots
    for ax in axs.flat:
        ax.set(xlabel="Epoch", ylabel="Value")

    # Adjust layout and save the plot
    plt.tight_layout()
    if not os.path.isdir(f"{args.save_path+task_type}/{args.dataset}/figures"):
        os.mkdir(f"{args.save_path+task_type}/{args.dataset}/figures")
    plt.savefig(
        f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png", dpi=600
    )
    plt.show()


def plot_pretrain_loss(pretrain_loss):
    plt.plot(pretrain_loss["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining Loss")
    plt.savefig(f"Data/pretraining.png", dpi=600)
    plt.show()
