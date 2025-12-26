from typing import Union, Collection

import torch

TensorCollection = Union[torch.Tensor, Collection[torch.Tensor], dict, None]
