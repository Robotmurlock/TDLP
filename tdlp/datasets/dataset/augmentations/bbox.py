"""Bounding box augmentation utilities."""
import torch

from tdlp.datasets.dataset.augmentations.base import NonDeterministicAugmentation
from tdlp.datasets.dataset.common.data import VideoClipData


class BBoxGaussianNoiseAugmentation(NonDeterministicAugmentation):
    """Add Gaussian noise to bbox and keypoint features."""
    SUPPORTED_FEATURES = ['bbox', 'keypoints']

    """
    Add Gaussian noise based on the bbox width and height.
    """
    def __init__(self, sigma: float = 0.05, proba: float = 0.5, unobs_noise: bool = True):
        """
        Args:
            sigma: Noise multiplier
            proba: Probability to apply this augmentation
            unobs_noise: Apply noise to unobserved part of the trajectory
        """
        super().__init__(proba=proba)
        self._sigma = sigma
        self._unobs_noise = unobs_noise

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds gaussian vector to bboxes vector x.

        Args:
            x:

        Returns:
            Input vector with noise
        """
        x_noise = self._sigma * torch.randn_like(x)
        x_noise[..., :-1][..., 0::2] *= x[..., 2].unsqueeze(-1)  # `x/h` noise is proportional to the `h`
        x_noise[..., :-1][..., 1::2] *= x[..., 3].unsqueeze(-1)  # `y/w` noise is proportional to the `w`
        x_noise[..., -1] *= 0.1
        return x + x_noise

    def _apply(self, data: VideoClipData) -> VideoClipData:
        for feature_name in self.SUPPORTED_FEATURES:
            if feature_name in data.observed.features:
                data.observed.features[feature_name] = self._add_noise(data.observed.features[feature_name])
            if self._unobs_noise and feature_name in data.unobserved.features:
                data.unobserved.features[feature_name] = self._add_noise(data.unobserved.features[feature_name])

        return data
