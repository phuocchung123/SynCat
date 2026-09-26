"""
Chooses the device to train on: a CUDA GPU, a TPU (through torch_xla) or the CPU.

`--accelerator auto` (the default) takes a GPU when CUDA is available, then a
TPU when torch_xla finds one, and falls back to the CPU otherwise;
`--accelerator gpu|tpu|cpu` forces one and fails loudly if it is missing.

torch_xla is only imported when a TPU is actually requested or probed, so the
GPU and CPU paths do not need it installed.

TPU notes. XLA compiles one graph per distinct tensor shape. Molecular graphs
make every batch a different shape, so the first epoch compiles one graph per
batch; the training loader is not shuffled, so later epochs repeat the same
shapes and reuse the compiled graphs. Only one TPU core is used.
"""

import torch

ACCELERATORS = ("auto", "gpu", "tpu", "cpu")


def tpu_available() -> bool:
    """
    Whether torch_xla is installed and its runtime is backed by a TPU.

    Returns
    -------
    bool
        True when a TPU can be used.
    """
    try:
        import torch_xla.runtime as xr
    except ImportError:
        return False
    try:
        return xr.device_type() == "TPU"
    except Exception:  # the runtime raises when no device can be initialised
        return False


def resolve_accelerator(args) -> str:
    """
    The accelerator to train on, from `args.accelerator`.

    Parameters
    ----------
    args : argparse.Namespace
        `accelerator` is one of `ACCELERATORS` (missing means "auto").

    Returns
    -------
    str
        "gpu", "tpu" or "cpu".

    Raises
    ------
    RuntimeError
        If a GPU or TPU is explicitly requested but not available.
    """
    choice = getattr(args, "accelerator", "auto") or "auto"
    if choice not in ACCELERATORS:
        raise ValueError(
            "accelerator must be one of %s, got %r" % (list(ACCELERATORS), choice)
        )
    if choice == "auto":
        if torch.cuda.is_available():
            return "gpu"
        return "tpu" if tpu_available() else "cpu"
    if choice == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu requested but CUDA is not available")
    if choice == "tpu" and not tpu_available():
        raise RuntimeError(
            "--accelerator tpu requested but no TPU was found: install torch_xla "
            "matching your torch version on a TPU machine (e.g. a Cloud TPU VM "
            "or a Colab/Kaggle TPU runtime), or set PJRT_DEVICE=TPU"
        )
    return choice


def get_device(accelerator: str, gpu_id: int = 0) -> torch.device:
    """
    The torch device of an accelerator.

    Parameters
    ----------
    accelerator : str
        "gpu", "tpu" or "cpu", as returned by `resolve_accelerator`.
    gpu_id : int, optional
        CUDA device index, used for "gpu" only (default is 0).

    Returns
    -------
    torch.device
        The device to place the model and batches on.
    """
    if accelerator == "gpu":
        return torch.device("cuda:%d" % gpu_id)
    if accelerator == "tpu":
        import torch_xla

        if hasattr(torch_xla, "device"):
            return torch_xla.device()
        import torch_xla.core.xla_model as xm  # torch_xla < 2.5

        return xm.xla_device()
    return torch.device("cpu")


def is_xla(device: torch.device) -> bool:
    """Whether `device` is an XLA (TPU) device."""
    return device.type == "xla"


def seed_device(device: torch.device, seed: int) -> None:
    """
    Seeds the random generator of the device itself where it has its own.

    CUDA is seeded by the caller; XLA keeps a separate generator for ops that
    run on the TPU, such as dropout.

    Parameters
    ----------
    device : torch.device
        The training device.
    seed : int
        The random seed.
    """
    if is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed, device=device)


def sync(device: torch.device) -> None:
    """
    Executes the operations recorded so far on an XLA device; no-op elsewhere.

    XLA traces operations lazily and only runs them at a synchronisation point,
    so a training step has to end with one.

    Parameters
    ----------
    device : torch.device
        The training device.
    """
    if not is_xla(device):
        return
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        import torch_xla.core.xla_model as xm  # torch_xla < 2.5

        xm.mark_step()


def cpu_state_dict(net: torch.nn.Module, device: torch.device) -> dict:
    """
    The model's state dict, moved to the CPU when it lives on an XLA device.

    A checkpoint holding XLA tensors cannot be loaded on a machine without a
    TPU, so TPU checkpoints are always written with CPU tensors. The GPU and
    CPU paths return the state dict unchanged.

    Parameters
    ----------
    net : torch.nn.Module
        The (unwrapped) model.
    device : torch.device
        The training device.

    Returns
    -------
    dict
        The state dict to save.
    """
    state = net.state_dict()
    if is_xla(device):
        return {k: v.detach().cpu() for k, v in state.items()}
    return state


def load_location(device: torch.device) -> torch.device:
    """
    Where `torch.load` should put a checkpoint's tensors.

    Checkpoints are loaded to the CPU for an XLA device and copied into the
    model by `load_state_dict`; other devices load directly.

    Parameters
    ----------
    device : torch.device
        The training device.

    Returns
    -------
    torch.device
        The `map_location` to use.
    """
    return torch.device("cpu") if is_xla(device) else device
