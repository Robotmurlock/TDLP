"""Utilities for merging architecture configs."""
from typing import Union

from omegaconf import DictConfig, OmegaConf

Cfg = Union[DictConfig, dict]


def merge_configs(lhs_cfg: Cfg, rhs_cfg: Cfg) -> Cfg:
    """
    Merges two configuration dictionaries.

    Args:
        lhs_cfg: Left-hand side configuration dictionary.
        rhs_cfg: Right-hand side configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    assert type(lhs_cfg) == type(rhs_cfg)
    if isinstance(lhs_cfg, DictConfig):
        return OmegaConf.merge(lhs_cfg, rhs_cfg)
    if isinstance(lhs_cfg, dict):
        return lhs_cfg | rhs_cfg

    # noinspection PyUnreachableCode
    raise TypeError(f'Unsupported Type {type(lhs_cfg)}!')
