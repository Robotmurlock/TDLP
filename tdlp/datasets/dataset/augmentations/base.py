"""
Implementation of trajectory augmentations.
"""
import random
from abc import abstractmethod, ABC
from typing import List

from tdlp.datasets.dataset.common.data import VideoClipData


class Augmentation(ABC):
    """
    Abstract augmentation - defines interface
    """
    @abstractmethod
    def apply(self, data: VideoClipData) -> VideoClipData:
        """
        Args:
            data: Data to augment

        Returns:
            augmented data
        """
        pass

    def __call__(self, data: VideoClipData) -> VideoClipData:
        return self.apply(data)


class IdentityAugmentation(Augmentation):
    """
    Performs no transformations (identity).
    """
    def apply(self, data: VideoClipData) -> VideoClipData:
        return data


class CompositionAugmentation(Augmentation):
    """
    Composition of multiple augmentations.
    """
    def __init__(self, augmentations: List[Augmentation]):
        """
        Args:
            augmentations: List of augmentations
        """
        self._augmentations = augmentations

    def apply(self, data: VideoClipData) -> VideoClipData:
        for aug in self._augmentations:
            data = aug.apply(data)
        return data


class NonDeterministicAugmentation(Augmentation, ABC):
    """
    Non-deterministic augmentation.
    """
    def __init__(self, proba: float):
        """
        Args:
            proba: Probability to apply augmentation.
        """
        self._proba = proba

    def apply(self, data: VideoClipData) -> VideoClipData:
        r = random.uniform(0, 1)
        if r > self._proba:
            # Skip augmentation
            return data

        return self._apply(data)

    @abstractmethod
    def _apply(self, data: VideoClipData) -> VideoClipData:
        """
        Args:
            data: Data to augment

        Returns:
            augmented data
        """
