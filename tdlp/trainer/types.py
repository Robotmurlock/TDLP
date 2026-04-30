"""Trainer type definitions."""
from enum import Enum


class TrainerType(str, Enum):
    DEFAULT = 'default'
    END_TO_END = 'end_to_end'
