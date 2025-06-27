"""
Torch distribution utils.
"""
import functools
import os
from typing import Union, Optional, Callable

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ZERO_RANK = (-1, 0)


def rank_zero_only(func: Callable) -> Callable:
    """Decorator that ensures the function runs only on rank 0.

    If the current process is rank 0 or if the distributed process group is
    not initialized, the decorated function is executed. Otherwise, the function
    is skipped.

    Args:
        func: The function to be decorated.

    Returns:
        wrapper: The wrapped function that will only execute on rank 0.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional[Callable]:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))

        if local_rank in ZERO_RANK:
            return func(*args, **kwargs)
        return None

    return wrapper


def is_zero_rank() -> bool:
    """
    Check if worker has rank zero. This is always True for single-node single-gpu training.

    Returns:
        True if worker has rank zero.
    """
    return int(os.environ.get('LOCAL_RANK', -1)) in ZERO_RANK


def is_global_zero_rank() -> bool:
    """
    Check if worker has global rank zero. This is always True for single-node single-gpu training.

    Returns:
        True if worker has global rank zero.
    """
    return int(os.environ.get('RANK', -1)) in ZERO_RANK


def rank_zero_tqdm(data_loader: DataLoader, *args, **kwargs) -> Union[DataLoader, tqdm]:
    """
    Wraps dataloader with a tqdm (only rank zero).

    Args:
        data_loader: Dataloader
        *args:
        **kwargs:

    Returns:
        Dataloader with a progress bar
    """
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    if local_rank in [-1, 0]:
        return tqdm(data_loader, total=len(data_loader), *args, **kwargs)

    return data_loader


def get_model(model: Union[nn.Module, DDP]) -> nn.Module:
    """
    Get model module.

    Args:
        model: Module

    Returns:
        Module
    """
    if isinstance(model, DDP):
        return model.module
    else:
        return model


def dist_barrier() -> None:
    """
    Multi-gpu: Blocks workers from continuing until all workers are gathered.
    Single-gpu: Does nothing.
    """
    if dist.is_initialized():
        dist.barrier()
