import torch

from mot_jepa.datasets.dataset.augmentations.base import NonDeterministicAugmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData


class BBoxGaussianNoiseAugmentation(NonDeterministicAugmentation):
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
        x_noise[..., 0] *= x[..., 2]  # `x` noise is proportional to the `h`
        x_noise[..., 2] *= x[..., 2]  # `h` noise is proportional to the `h`
        x_noise[..., 1] *= x[..., 3]  # `y` noise is proportional to the `w`
        x_noise[..., 3] *= x[..., 3]  # `w` noise is proportional to the `w`
        return x + x_noise

    def _apply(self, data: VideoClipData) -> VideoClipData:
        for feature_name in self.SUPPORTED_FEATURES:
            if feature_name in data.observed.features:
                data.observed.features[feature_name] = self._add_noise(data.observed.features[feature_name])
            if self._unobs_noise and feature_name in data.unobserved.features:
                data.unobserved.features[feature_name] = self._add_noise(data.unobserved.features[feature_name])

        return data
