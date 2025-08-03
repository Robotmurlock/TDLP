import math
import random
from typing import Tuple

import torch

from mot_jepa.datasets.dataset.augmentations.base import Augmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData


class PointOcclusionAugmentations(Augmentation):
    """
    Remove points of observed track (simulate occlusions)
    """
    def __init__(self, drop_ratio: float, occlude_unobs: bool = False):
        super().__init__()
        self._drop_ratio = drop_ratio
        self._occlude_unobs = occlude_unobs

    def apply(self, data: VideoClipData) -> VideoClipData:
        n_tracks, clip_length = data.observed_temporal_mask.shape

        observed_remove_masks = torch.zeros((n_tracks, clip_length), dtype=torch.bool)

        for n in range(n_tracks):
            if random.random() < self._drop_ratio:
                observed_begin = random.randint(0, clip_length)
                observed_max_interval = clip_length - observed_begin
                observed_end = observed_begin + math.ceil(observed_max_interval * random.random())
                observed_remove_masks[n, observed_begin:observed_end] = True
        data.observed_temporal_mask = data.observed_temporal_mask | observed_remove_masks

        if self._occlude_unobs:
            n_detections = data.unobserved_temporal_mask.shape[0]
            for n in range(n_detections):
                if random.random() < self._drop_ratio / clip_length:
                    data.unobserved_temporal_mask[n] = True

        return data


class LeftOrRightOcclusionAugmentations(Augmentation):
    """
    Remove the left or the right part of the observed track (simulate occlusions)
    """
    def __init__(self, drop_ratio: float, min_length: int = 1):
        super().__init__()
        self._drop_ratio = drop_ratio
        self._min_length = min_length

    def apply(self, data: VideoClipData) -> VideoClipData:
        traj_length = data.observed_temporal_mask.shape[0]
        if traj_length < self._min_length:
            return data

        if random.random() > self._drop_ratio:
            return data

        occlusion_point = random.randrange(self._min_length, traj_length - self._min_length)
        if random.random() > 0.5:
            data.observed_bboxes[occlusion_point:, :] = 0
            data.observed_temporal_mask[occlusion_point:] = True
        else:
            data.observed_bboxes[:occlusion_point, :] = 0
            data.observed_temporal_mask[:occlusion_point] = True

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
        n_tracks, clip_length = data.observed_temporal_mask.shape

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


class SmartIdentitySwitchAugmentation(Augmentation):
    def __init__(self, switch_ratio: float, iou_threshold: float = 0.1, max_switch_ratio: float = 1.0):
        """
        Partially switch track identities, simulating realistic identity switches.

        Args:
            switch_ratio: Probability of attempting a switch.
            iou_threshold: IoU threshold to determine intersection.
        """
        super().__init__()
        self._switch_ratio = switch_ratio
        self._iou_threshold = iou_threshold
        self._max_switch_ratio = max_switch_ratio

    def apply(self, data: VideoClipData) -> VideoClipData:
        n_tracks, _ = data.observed_temporal_mask.shape

        candidate_matrix, candidate_pair_matrix = self._compute_switch_candidates(data)

        switch_pairs = torch.nonzero(candidate_pair_matrix)
        switch_pairs = switch_pairs[torch.randperm(len(switch_pairs))]
        max_switches = int(self._max_switch_ratio * n_tracks)
        switch_pairs = switch_pairs[:max_switches]

        for idx_a, idx_b in switch_pairs:
            if random.random() > self._switch_ratio:
                continue

            switchable_timesteps = self._get_switchable_timesteps(
                data.observed_temporal_mask, idx_a, idx_b, candidate_matrix[idx_a, idx_b]
            )

            if len(switchable_timesteps) == 0:
                continue

            switch_time = random.choice(switchable_timesteps)

            # Perform the switch after the selected timestep
            self._switch(data, idx_a, idx_b, switch_time)

        return data

    def _compute_switch_candidates(self, data: VideoClipData) -> Tuple[torch.Tensor, torch.Tensor]:
        n_tracks, clip_length = data.observed_temporal_mask.shape
        candidate_points_matrix = torch.zeros((n_tracks, n_tracks, clip_length), dtype=torch.bool)

        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                ious = self._calculate_iou_trajectory(
                    data.observed_bboxes[i], data.observed_bboxes[j],
                    data.observed_temporal_mask[i], data.observed_temporal_mask[j]
                )
                candidate_points_matrix[i, j] = ious >= self._iou_threshold
                candidate_points_matrix[j, i] = candidate_points_matrix[i, j]

        candidate_pair_matrix = candidate_points_matrix.any(dim=-1)
        return candidate_points_matrix, candidate_pair_matrix

    def _calculate_iou_trajectory(self, boxes_a, boxes_b, mask_a, mask_b):
        valid = (~mask_a & ~mask_b)
        ious = torch.zeros(boxes_a.shape[0])
        valid_indices = torch.nonzero(valid).flatten()

        if len(valid_indices) == 0:
            return ious

        for t in valid_indices:
            ious[t] = self._calculate_iou(boxes_a[t], boxes_b[t])

        return ious

    @staticmethod
    def _calculate_iou(box_a, box_b):
        xa1, ya1, wa, ha, _ = box_a
        xb1, yb1, wb, hb, _ = box_b

        xa2, ya2 = xa1 + wa, ya1 + ha
        xb2, yb2 = xb1 + wb, yb1 + hb

        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)

        intersection_w = max(0, xi2 - xi1)
        intersection_h = max(0, yi2 - yi1)
        intersection_area = intersection_w * intersection_h

        area_a = wa * ha
        area_b = wb * hb
        union_area = area_a + area_b - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _get_switchable_timesteps(mask, idx_a, idx_b, candidate_vector):
        combined_mask = ~mask[idx_a] & ~mask[idx_b]
        valid_timesteps = torch.nonzero(candidate_vector & combined_mask).flatten()
        return valid_timesteps.tolist()

    @staticmethod
    def _switch(data, idx_a, idx_b, start_time):
        _, clip_length = data.observed_temporal_mask.shape
        switch_max_len = clip_length - start_time - 1
        assert switch_max_len >= 0
        if switch_max_len == 0:
            return
        swap_index = start_time + random.randint(1, switch_max_len)

        data.observed_bboxes[[idx_a, idx_b], swap_index] = data.observed_bboxes[[idx_b, idx_a], swap_index]
        data.observed_temporal_mask[[idx_a, idx_b], swap_index] = data.observed_temporal_mask[[idx_b, idx_a], swap_index]


def test_identity_switch_augmentation():
    augmentation = SmartIdentitySwitchAugmentation(switch_ratio=1.0, iou_threshold=0.1)

    observed_bboxes = torch.tensor([
        [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]],
        [[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0]],
        [[4, 4, 5, 5], [4, 4, 5, 5], [4, 4, 5, 5]]
    ], dtype=torch.float)

    observed_temporal_mask = torch.tensor([
        [False, False, False],
        [False, False, True],
        [False, False, False]
    ])

    data = VideoClipData(
        observed_bboxes=observed_bboxes,
        observed_temporal_mask=observed_temporal_mask,
        observed_ts=None,
        unobserved_bboxes=None,
        unobserved_ts=None,
        unobserved_temporal_mask=None
    )

    augmented_data = augmentation.apply(data)

    print("Original BBoxes:", data.observed_bboxes)
    print("Augmented BBoxes:", augmented_data.observed_bboxes)


if __name__ == '__main__':
    test_identity_switch_augmentation()

