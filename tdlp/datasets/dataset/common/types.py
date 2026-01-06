"""Shared tensor-related type aliases for dataset utilities."""
from typing import Collection, Union

import torch

TensorCollection = Union[torch.Tensor, Collection[torch.Tensor], dict, None]
