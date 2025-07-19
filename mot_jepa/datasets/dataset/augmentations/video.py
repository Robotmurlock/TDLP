import math
import random

import torch

from mot_jepa.datasets.dataset.augmentations.base import Augmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData


class OcclusionAugmentations(Augmentation):
    """
    Remove parts of observed track (simulate occlusions)
    """
    def __init__(self, drop_ratio: float):
        super().__init__()
        self._drop_ratio = drop_ratio

    def apply(self, data: VideoClipData) -> VideoClipData:
        n_tracks, clip_length, _ = data.observed_temporal_mask.shape

        observed_remove_masks = torch.zeros((n_tracks, clip_length), dtype=torch.bool)

        for n in range(n_tracks):
            if random.random() < self._drop_ratio:
                observed_begin = random.randint(0, clip_length)
                observed_max_interval = clip_length - observed_begin
                observed_end = observed_begin + math.ceil(observed_max_interval * random.random())
                observed_remove_masks[n, observed_begin:observed_end] = True
        data.observed_temporal_mask = data.observed_temporal_mask | observed_remove_masks

        return data


class IdentitySwitchAugmentation(Augmentation):
    def __init__(self, switch_ratio: float):
        """
        Partially switch track identities (simulate identity switches).

        Args:
            switch_ratio:
        """
        super().__init__()
        self._switch_ratio = switch_ratio

    def apply(self, data: VideoClipData) -> VideoClipData:
        n_tracks, clip_length, _ = data.observed_temporal_mask.shape

        for t in range(0, clip_length):
            switch_proba = torch.ones((n_tracks,), dtype=torch.float) * self._switch_ratio
            switch_proba = (1 - data.observed_temporal_mask.all(dim=-1).float()) * switch_proba
            switch_map = torch.bernoulli(switch_proba)
            switch_indices = torch.nonzero(switch_map)  # objects to switch
            switch_indices = switch_indices.reshape((switch_indices.shape[0],))
            if len(switch_indices) == 1 and n_tracks > 1:
                # Only one object can be switched, but we have more than one object.
                # So we need to randomly select another object to switch.
                switch_indices = torch.as_tensor(
                    [switch_indices[0].item(), random.randint(0, n_tracks - 1)], dtype=torch.long
                )
            if len(switch_indices) > 1:
                # Switch the trajectory features, boxes and masks:
                shuffle_switch_indices = switch_indices[torch.randperm(len(switch_indices))]
                data.observed_bboxes[switch_indices, t, :] = data.observed_bboxes[shuffle_switch_indices, t, :]
                data.observed_temporal_mask[switch_indices, t] = data.observed_temporal_mask[shuffle_switch_indices, t]
            else:
                continue  # no object to switch

        return data
