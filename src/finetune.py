import os
import json
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from data import GraphDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import model
from training import train
from validation import validation
from multi_gpu import init_process_group, resolve_gpu_ids, set_launch_env
from utils import collate_reaction_graphs, set_seed, setup_logging
from visualization import plot_parity


def _load_checkpoint_safely(net, checkpoint, logger):
    """
    Loads a checkpoint's state dict into `net`, tolerating a mismatched
    regression head (e.g. when warm-starting from a classification checkpoint)
    while surfacing any other unexpected mismatch.

    Parameters
    ----------
    net : torch.nn.Module
        The model to load weights into.
    checkpoint : dict
        Checkpoint dictionary containing "model_state_dict".
    logger : logging.Logger
        Logger used to report skipped/missing keys.

    Returns
    -------
    None
    """
    result = net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    encoder_mismatches = [
        k
        for k in list(result.missing_keys) + list(result.unexpected_keys)
        if not k.startswith("regressor.")
    ]
    if encoder_mismatches:
        raise RuntimeError(
            "Checkpoint is incompatible with the current encoder "
            "architecture (mismatched keys: %s)" % encoder_mismatches
        )
    if result.missing_keys or result.unexpected_keys:
        logger.info(
            "--- loaded checkpoint with a mismatched head; missing=%s, unexpected=%s"
            % (result.missing_keys, result.unexpected_keys)
        )


def _build_model(args, node_dim, edge_dim):
    return model(
        node_dim,
        edge_dim,
        args.layer,
        args.emb_dim,
        args.dropout,
        num_attention_layer=getattr(args, "attention_layer", 1),
    )


def _ddp_train_worker(rank, args, gpus, model_path, resume):
    """
    One DistributedDataParallel training process, pinned to GPU `gpus[rank]`.

    Each rank trains on its own `1/len(gpus)` shard of the training set with a
    per-GPU batch of `args.batch_size // len(gpus)`; rank 0 validates, logs and
    writes the best checkpoint to `model_path`, as in the single-GPU run.
    """
    world_size = len(gpus)
    device = torch.device("cuda:%d" % gpus[rank])
    torch.cuda.set_device(device)
    init_process_group(rank, world_size)
    try:
        set_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        folder = args.Data_folder + args.npz_folder + "/"
        train_set = GraphDataset(folder + "train.npz")
        sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        # every rank must run the same number of steps, hence drop_last
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=max(1, min(args.batch_size // world_size, len(sampler))),
            sampler=sampler,
            collate_fn=collate_reaction_graphs,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader = test_loader = None
        if rank == 0:
            val_set = GraphDataset(folder + "valid.npz")
            val_loader = DataLoader(
                dataset=val_set,
                batch_size=int(np.min([args.batch_size, len(val_set)])),
                shuffle=False,
                collate_fn=collate_reaction_graphs,
                num_workers=args.num_workers,
            )
            if args.track_test_each_epoch:
                test_set = GraphDataset(folder + "test.npz")
                test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=int(np.min([args.batch_size, len(test_set)])),
                    shuffle=False,
                    collate_fn=collate_reaction_graphs,
                    num_workers=args.num_workers,
                )

        node_dim = train_set.rmol_node_attr[0].shape[1]
        edge_dim = train_set.rmol_edge_attr[0].shape[1]
        net = _build_model(args, node_dim, edge_dim).to(device)
        current_epoch, best_val_loss = 0, 1e10
        if resume:
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            _load_checkpoint_safely(net, checkpoint, setup_logging())
            current_epoch, best_val_loss = checkpoint["epoch"], checkpoint["val_loss"]
        net = DistributedDataParallel(net, device_ids=[device.index])
        train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            args.epochs - current_epoch,
            args.lr,
            args.weight_decay,
            current_epoch=current_epoch,
            best_val_loss=best_val_loss,
            test_loader=test_loader,
        )
    finally:
        dist.destroy_process_group()


def finetune(args, save_embedding: bool = True) -> dict:
    """
    Fine-tune a graph neural network on chemical reaction-yield data.

    With several GPUs (`args.gpus`), training runs as one DistributedDataParallel
    process per GPU; model selection and the final evaluation then run on the
    first of them, exactly as in a single-GPU run.

    The checkpoint with the lowest validation loss is selected, re-scored once on
    the validation set, and then evaluated exactly once on the test set. The test
    set is only scored during training when `args.track_test_each_epoch` is set,
    and even then it never influences checkpoint selection.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing all required settings and paths.
    save_embedding : bool, optional
        Whether to dump the reaction embeddings to
        `<monitor_folder>/embedding.json` (default is True).

    Returns
    -------
    dict
        Selected epoch ("best_epoch"), validation loss/metrics of the selected
        checkpoint ("val_loss", "val_metrics"), test metrics ("test_metrics"),
        per-sample test labels/predictions in test-set order ("test_labels",
        "test_preds"), subset sizes ("n_train", "n_valid", "n_test") and
        runtimes in seconds ("train_runtime_sec", "eval_runtime_sec").
    """
    logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")
    model_path = args.model_path + args.model_name
    gpus = resolve_gpu_ids(args)
    device = torch.device("cuda:%d" % gpus[0]) if gpus else torch.device("cpu")
    logger.info("device is\t%s" % device)
    if len(gpus) > 1:
        logger.info(
            "--- DistributedDataParallel on GPUs %s, per-GPU batch size %d"
            % (gpus, args.batch_size // len(gpus))
        )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "train.npz")
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(np.min([args.batch_size, len(train_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "test.npz")
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int(np.min([args.batch_size, len(test_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_set = GraphDataset(args.Data_folder + args.npz_folder + "/" + "valid.npz")
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=int(np.min([args.batch_size, len(val_set)])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        num_workers=args.num_workers,
        drop_last=False,
    )

    logger.info("-- CONFIGURATIONS")
    logger.info(
        "--- train/valid/test: %d/%d/%d" % (len(train_set), len(val_set), len(test_set))
    )
    logger.info(
        "--- max no. reactants_train, valid, test respectively: %d, %d, %d"
        % (train_set.rmol_max_cnt, val_set.rmol_max_cnt, test_set.rmol_max_cnt)
    )
    logger.info(
        "--- max no. products_train, valid, test respectively: %d, %d, %d"
        % (train_set.pmol_max_cnt, val_set.pmol_max_cnt, test_set.pmol_max_cnt)
    )
    logger.info("--- model_path: %s" % model_path)
    monitor_test_loader = test_loader if args.track_test_each_epoch else None

    # training

    node_dim = train_set.rmol_node_attr[0].shape[1]
    edge_dim = train_set.rmol_edge_attr[0].shape[1]
    train_start = time.time()
    resume = os.path.exists(model_path)
    if len(gpus) > 1:
        logger.info("-- TRAINING" + (" (resumed)" if resume else ""))
        set_launch_env()
        mp.spawn(
            _ddp_train_worker,
            args=(args, gpus, model_path, resume),
            nprocs=len(gpus),
            join=True,
        )
        # the workers reconfigured logging in their own processes only
        logger = setup_logging(log_filename=args.monitor_folder + "monitor.log")
    else:
        net = _build_model(args, node_dim, edge_dim).to(device)
        current_epoch, best_val_loss = 0, 1e10
        if not resume:
            logger.info("-- TRAINING")
        else:
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            _load_checkpoint_safely(net, checkpoint, logger)
            current_epoch, best_val_loss = checkpoint["epoch"], checkpoint["val_loss"]
        train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            args.epochs - current_epoch,
            args.lr,
            args.weight_decay,
            current_epoch=current_epoch,
            best_val_loss=best_val_loss,
            test_loader=monitor_test_loader,
        )

    train_runtime = time.time() - train_start
    if not os.path.exists(model_path):
        raise RuntimeError(
            "Training finished without saving a checkpoint to %s (the validation "
            "loss never improved, e.g. because it was NaN)" % model_path
        )

    # model selection: reload the best-validation checkpoint
    test_y = test_loader.dataset.y
    net = _build_model(args, node_dim, edge_dim).to(device)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_loss = validation(
        args, net, val_loader, device, torch.nn.HuberLoss()
    )
    logger.info(
        "--- selected checkpoint from epoch %d, val_loss %.4f, val_mae %.4f, val_rmse %.4f"
        % (checkpoint["epoch"], val_loss, val_metrics["mae"], val_metrics["rmse"])
    )

    # test: the selected model is evaluated exactly once
    eval_start = time.time()
    metrics, rsmis, test_labels, test_preds, emb = validation(
        args, net, test_loader, device
    )
    eval_runtime = time.time() - eval_start
    plot_parity(
        test_labels,
        test_preds,
        metrics,
        os.path.join(args.image_folder, "test_parity.png"),
    )
    logger.info("-- RESULT")
    logger.info("--- test size: %d" % (len(test_y)))
    logger.info(
        "--- MAE: %.4f (%.2f pp), RMSE: %.4f (%.2f pp), R2: %s, Pearson: %s"
        % (
            metrics["mae"],
            metrics["mae"] * 100,
            metrics["rmse"],
            metrics["rmse"] * 100,
            "n/a" if metrics["r2"] is None else "%.4f" % metrics["r2"],
            "n/a" if metrics["pearson"] is None else "%.4f" % metrics["pearson"],
        )
    )

    if save_embedding:
        dict_emb = {
            "Name": "Embedding",
            "rsmis": rsmis,
            "emb": emb,
        }
        with open(args.monitor_folder + "embedding.json", "w") as f:
            json.dump(dict_emb, f)

    return {
        "best_epoch": int(checkpoint["epoch"]),
        "val_loss": float(val_loss),
        "val_metrics": val_metrics,
        "test_metrics": metrics,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "n_train": len(train_set),
        "n_valid": len(val_set),
        "n_test": len(test_set),
        "train_runtime_sec": train_runtime,
        "eval_runtime_sec": eval_runtime,
    }
