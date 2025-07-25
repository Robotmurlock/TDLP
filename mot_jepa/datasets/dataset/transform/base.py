"""
Implementations of data transformations.
"""
from abc import ABC, abstractmethod
from typing import List

from mot_jepa.datasets.dataset.common.data import VideoClipData


class Transform(ABC):
    """
    Maps data with implemented transformation.
    """
    def __init__(self, name: str):
        """
        Args:
            name: Transformation name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            Transformation name
        """
        return self._name

    @abstractmethod
    def apply(self, data: VideoClipData) -> VideoClipData:
        """
        Perform transformation on given raw data.

        Args:
            data: Raw data

        Returns:
            Transformed data
        """
        pass

    def __call__(self, data: VideoClipData) -> VideoClipData:
        return self.apply(data)


class IdentityTransform(Transform):
    """
    Transformation neutral operator.
    """
    def __init__(self):
        super().__init__(name='identity')

    def apply(self, data: VideoClipData) -> VideoClipData:
        return data


class ComposeTransform(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__(name='compose')
        self._transforms = transforms

    def apply(self, data: VideoClipData) -> VideoClipData:
        for t in self._transforms:
            data = t(data)
        return data
