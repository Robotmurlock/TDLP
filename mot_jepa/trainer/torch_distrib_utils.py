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


"""
DDP Sampler support.
"""
from operator import itemgetter
from typing import Optional

from torch.utils.data import DistributedSampler, Dataset, Sampler


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        # noinspection PyTypeChecker
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Source: https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py#L522

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super().__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        # noinspection PyTypeChecker
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
