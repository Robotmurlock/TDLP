"""
PyTorch's extension with simple functions
"""
from pathlib import Path
from typing import Union, List, Tuple, Dict, Hashable, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.optim import Optimizer

TorchDevice = Union[str, int, torch.device]
TensorBasedObject = Union[
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor, ...],
    Dict[Hashable, torch.Tensor],
    bool, bytes, float, int, str, type(None)
]
PrimitiveTypeTuple = (bool, bytes, float, int, str, type(None))


def get_optim_lr(optimizer: 'Optimizer') -> float:
    """
    Gets current optimizer learning rate

    Args:
        optimizer: Torch optimizer

    Returns:
        Learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optim_lr(optimizer: 'Optimizer', lr: float) -> None:
    """
    Sets current optimizer learning rate

    Args:
        optimizer: Optimizer
        lr: learning rate to be set
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model: nn.Module, path: str) -> None:
    """
    Saves model to given path. Creates directory if it doesn't exist.

    Args:
        model: PyTorch model
        path: Checkpoint path
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path)


def to_device(item: TensorBasedObject, device: TorchDevice) -> TensorBasedObject:
    """
    Moves tensor based object to device recursively.

    Args:
        item: Tensor object or object containing tensors
        device: Target device

    Returns:
        Object moved to device
    """
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, list):
        return [to_device(v, device) for v in item]
    if isinstance(item, tuple):
        return tuple(to_device(v, device) for v in item)
    if isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}

    raise TypeError(f'Unsupported item type: {type(item)}')


def to_cpu(item: TensorBasedObject) -> TensorBasedObject:
    """
    Moves tensor based object to CPU recursively.

    Args:
        item: Tensor object or object containing tensors

    Returns:
        Object moved to CPU
    """
    if isinstance(item, torch.Tensor):
        return item.cpu()
    if isinstance(item, list):
        return [to_cpu(v) for v in item]
    if isinstance(item, tuple):
        return tuple(to_cpu(v) for v in item)
    if isinstance(item, dict):
        return {k: to_cpu(v) for k, v in item.items()}
    if isinstance(item, PrimitiveTypeTuple):
        return item  # skip primitive types

    raise TypeError(f'Unsupported item type: {type(item)}')


def detach(item: TensorBasedObject) -> TensorBasedObject:
    """
    Detaches tensor based object recursively.

    Args:
        item: Tensor object or object containing tensors

    Returns:
        Object with detached tensor gradients
    """
    if isinstance(item, torch.Tensor):
        return item.detach()
    if isinstance(item, list):
        return [detach(v) for v in item]
    if isinstance(item, tuple):
        return tuple(detach(v) for v in item)
    if isinstance(item, dict):
        return {k: detach(v) for k, v in item.items()}
    if isinstance(item, PrimitiveTypeTuple):
        return item  # skip primitive types

    raise TypeError(f'Unsupported item type: {type(item)}')


# noinspection PyProtectedMember
def optimizer_to(optim: 'Optimizer', device: TorchDevice) -> None:
    """
    Moves optimizer object to device recursively.

    Args:
        optim: Optimizer
        device: Device (CPU/GPU)
    """
    # pylint: disable=protected-access
    for param in optim.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
