"""
Helpers for multi-GPU training with DistributedDataParallel (DDP).

`--gpus 0 1` (or `--gpus all`) trains one process per GPU; `finetune` spawns
the processes itself, so no `torchrun` launcher is needed. Each process sees
`batch_size // n_gpus` samples per step, so the effective batch size, and
therefore the learning-rate setting, is the same as for a single-GPU run.
"""

import os
import socket
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def resolve_gpu_ids(args) -> list:
    """
    GPU ids to train on: `args.gpus` if given, otherwise `[args.device]`.

    Parameters
    ----------
    args : argparse.Namespace
        Needs `device`; `gpus` is optional (a list of ids as strings/ints, or ["all"]).

    Returns
    -------
    list
        Distinct GPU ids; an empty list when CUDA is unavailable.
    """
    gpus = getattr(args, "gpus", None)
    if not torch.cuda.is_available():
        if gpus and len(gpus) > 1:
            raise RuntimeError("--gpus %s requested but CUDA is not available" % gpus)
        return []
    n_available = torch.cuda.device_count()
    if not gpus:
        return [int(args.device)]
    if [str(g).lower() for g in gpus] == ["all"]:
        return list(range(n_available))
    ids = list(dict.fromkeys(int(g) for g in gpus))
    invalid = [g for g in ids if not 0 <= g < n_available]
    if invalid:
        raise ValueError(
            "GPU ids %s do not exist (%d CUDA device(s) visible)"
            % (invalid, n_available)
        )
    return ids


def is_main_process() -> bool:
    """True outside DDP and on rank 0 inside it."""
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(net: torch.nn.Module) -> torch.nn.Module:
    """The plain model inside a DDP wrapper (or `net` itself)."""
    return (
        net.module
        if isinstance(net, torch.nn.parallel.DistributedDataParallel)
        else net
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def set_launch_env() -> None:
    """Rendezvous settings for the spawned processes (single machine)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    if sys.platform == "win32":
        # Windows builds of torch ship without libuv for the TCP store
        os.environ["USE_LIBUV"] = "0"


def init_process_group(rank: int, world_size: int) -> None:
    """NCCL where available (Linux), gloo otherwise (Windows)."""
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    # rank 0 validates alone at the end of every epoch while the others wait
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(hours=1)
    )


def gather_lists(*values: list) -> tuple:
    """Concatenates per-rank python lists on every rank (no-op outside DDP)."""
    if not dist.is_initialized():
        return values
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, values)
    return tuple([x for part in gathered for x in part[i]] for i in range(len(values)))


def broadcast_flag(flag: bool, device: torch.device) -> bool:
    """Rank 0's value of `flag` on every rank (no-op outside DDP)."""
    if not dist.is_initialized():
        return flag
    tensor = torch.tensor([int(flag)], device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())
