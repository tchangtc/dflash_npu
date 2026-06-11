"""Device abstraction layer for DFlash — supports CUDA and Ascend NPU."""

import torch


def get_device_type() -> str:
    """Return ``'npu'``, ``'cuda'`` or ``'cpu'`` based on available hardware."""
    try:
        if torch.npu.is_available():
            return "npu"
    except AttributeError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device(index: int = 0) -> torch.device:
    """Return the default compute device for the current hardware."""
    dtype = get_device_type()
    if dtype == "cpu":
        return torch.device("cpu")
    return torch.device(f"{dtype}:{index}")


def synchronize() -> None:
    """Synchronize all operations on the current device."""
    dtype = get_device_type()
    if dtype == "npu":
        torch.npu.synchronize()
    elif dtype == "cuda":
        torch.cuda.synchronize()


def manual_seed_all(seed: int) -> None:
    """Set the random seed for all devices of the current type."""
    dtype = get_device_type()
    if dtype == "npu":
        torch.npu.manual_seed_all(seed)
    elif dtype == "cuda":
        torch.cuda.manual_seed_all(seed)


def set_device(index: int) -> None:
    """Set the current device to *index*."""
    dtype = get_device_type()
    if dtype == "npu":
        torch.npu.set_device(index)
    elif dtype == "cuda":
        torch.cuda.set_device(index)


def get_dist_backend() -> str:
    """Return the distributed backend name for the current hardware."""
    dtype = get_device_type()
    if dtype == "npu":
        return "hccl"
    return "nccl"
