import torch

from mot_jepa.datasets.dataset.augmentations.base import Augmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData


class AppearanceNoiseAugmentation(Augmentation):
    """
    Source: https://github.com/TrackingLaboratory/CAMELTrack/blob/main/cameltrack/train/transforms/batch.py

    Apply Gaussian noise to the track feature embeddings in the batch. The amount of noise added
    is controlled by the `alpha` parameter, which scales the noise by the standard deviation of
    the embeddings along the last dimension.
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self._alpha = alpha

    def apply(self, data: VideoClipData) -> VideoClipData:
        if 'appearance' not in data.observed.features:
            return data

        for attr in ['observed', 'unobserved']:
            observed_emb = getattr(data, attr).features['appearance'][..., :-1]  # Do not jitter the visibility
            observed_emb_std = observed_emb.std(dim=-1, keepdim=True)
            observed_emb_noise = torch.randn_like(observed_emb) * observed_emb_std
            getattr(data, attr).features['appearance'][..., :-1] = observed_emb + self._alpha * observed_emb_noise

        return data