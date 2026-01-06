"""Bounding-box related transforms for dataset clips."""
from typing import Dict, List

import torch

from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.transform.base import Transform
from tdlp.datasets.dataset.transform.utils import expand_pattern


class BBoxXYWHtoXYXY(Transform):
    """Convert bbox format from (x, y, w, h) to (x1, y1, x2, y2)."""
    def __init__(self, keep_wh: bool = False):
        """
        Args:
            keep_wh: Whether to keep the width and height.
        """
        super().__init__(name='bbox_xywh_to_xyxy')
        self._keep_wh = keep_wh

    def apply(self, data: VideoClipData) -> VideoClipData:
        if 'bbox' not in data.observed.features:
            return data

        observed_bottom_xy = data.observed.features['bbox'][..., :2] + data.observed.features['bbox'][..., 2:4]
        unobserved_bottom_xy = data.unobserved.features['bbox'][..., :2] + data.unobserved.features['bbox'][..., 2:4]

        if not self._keep_wh:
            data.observed.features['bbox'][..., 2:4] = observed_bottom_xy
            data.unobserved.features['bbox'][..., 2:4] = unobserved_bottom_xy
        else:
            data.observed.features['bbox'] = torch.cat([
                data.observed.features['bbox'][..., :2],
                observed_bottom_xy,
                data.observed.features['bbox'][..., 2:]
            ], dim=-1)
            data.unobserved.features['bbox'] = torch.cat([
                data.unobserved.features['bbox'][..., :2],
                unobserved_bottom_xy,
                data.unobserved.features['bbox'][..., 2:]
            ], dim=-1)

        return data


class BBoxStandardization(Transform):
    """Standardize bbox coordinates with provided mean/std."""
    def __init__(
        self,
        coord_mean: Dict[str, List[float]],
        coord_std: Dict[str, List[float]],
    ):
        """
        Args:
            coord_mean: Mean of the coordinates.
            coord_std: Standard deviation of the coordinates.
        """
        super().__init__(name='feature_standardization')

        assert set(coord_mean.keys()) == set(coord_std.keys())

        self._feature_names = list(coord_mean.keys())
        self._coord_mean = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_mean.items()}
        self._coord_std = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_std.items()}

    def apply(self, data: VideoClipData) -> VideoClipData:
        for feature_name in self._feature_names:
            if feature_name not in data.observed.features:
                continue

            data.observed.features[feature_name] = \
                (data.observed.features[feature_name] - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.unobserved.features[feature_name] = \
                (data.unobserved.features[feature_name] - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.observed.features[feature_name][data.observed.mask] = 0
            data.unobserved.features[feature_name][data.unobserved.mask] = 0

        return data


class FeatureFODStandardization(Transform):
    """Standardize bbox features and augment with finite differences."""
    def __init__(
        self,
        coord_mean: Dict[str, List[float]],
        coord_std: Dict[str, List[float]],
        fod_mean: Dict[str, List[float]],
        fod_std: Dict[str, List[float]],
        fod_time_scaled: bool = False
    ):
        """
        Args:
            coord_mean: Mean of the coordinates.
            coord_std: Standard deviation of the coordinates.
            fod_mean: Mean of the finite differences.
            fod_std: Standard deviation of the finite differences.
            fod_time_scaled: Whether to scale the finite differences by the time.
        """
        super().__init__(name='feature_fod_standardization')

        assert set(coord_mean.keys()) == set(coord_std.keys()) == set(fod_mean.keys()) == set(fod_std.keys())

        self._feature_names = list(coord_mean.keys())
        self._coord_mean = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_mean.items()}
        self._coord_std = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_std.items()}
        self._fod_mean = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in fod_mean.items()}
        self._fod_std = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in fod_std.items()}
        self._fod_time_scaled = fod_time_scaled

    def apply(self, data: VideoClipData) -> VideoClipData:
        for feature_name in self._feature_names:
            if feature_name not in data.observed.features:
                continue

            features = data.observed.features[feature_name]
            mask = data.observed.mask

            if self._fod_time_scaled:
                fod = torch.zeros_like(features)
                ts = data.observed.ts.unsqueeze(-1).repeat(1, 1, features.shape[-1])
                for n in range(features.shape[0]):
                    ts_n = ts[n][~mask[n]]
                    features_n = features[n][~mask[n]]
                    if features_n.shape[0] == 0:
                        continue

                    ts_diff = torch.clamp(ts_n[1:, :] - ts_n[:-1, :], min=1)
                    features_n[1:, :] = (features_n[1:, :] - features_n[:-1, :]) / (ts_diff)
                    features_n[0, :] = 0
                    fod[n][~mask[n]] = features_n
            else:
                fod = torch.zeros_like(features)
                fod[:, 1:, :] = features[:, 1:, :] - features[:, :-1, :]

            # FoD standardization
            fod = (fod - self._fod_mean[feature_name]) / self._fod_std[feature_name]
            extended_mask = mask[:, :-1].unsqueeze(-1).repeat(1, 1, features.shape[-1]).float()
            fod[:, 1:, :] = fod[:, 1:, :] * (1 - extended_mask)

            features = (features - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.observed.features[feature_name] = torch.cat([features, fod], dim=-1)  # Add FOD
            data.observed.features[feature_name][data.observed.mask] = 0  # Cleanup

            data.unobserved.features[feature_name] = \
                (data.unobserved.features[feature_name] - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.unobserved.features[feature_name][data.unobserved.mask] = 0  # Cleanup

        return data


class BBoxMinMaxScaling(Transform):
    """Scale bbox/keypoint coordinates to [0, 1] range."""
    def __init__(self):
        super().__init__(name='bbox_min_max_scaling')

    def apply(self, data: VideoClipData) -> VideoClipData:
        if 'bbox' not in data.observed.features and 'keypoints' not in data.observed.features:
            return data

        bboxes_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        # Collect coordinates and masks
        if 'bbox' in data.observed.features:
            N, T, _ = data.observed.features['bbox'].shape
            bboxes_list.append(data.observed.features['bbox'][:, :, :4].reshape(N, 2 * T, 2))  # (N, 2 * T, 2)
            masks_list.append(data.observed.mask.repeat(1, 2))
            bboxes_list.append(data.unobserved.features['bbox'][:, :4].unsqueeze(1).reshape(N, 2 * 1, 2))  # (N, 2, 2)
            masks_list.append(data.unobserved.mask.unsqueeze(1).repeat(1, 2))
        if 'keypoints' in data.observed.features:
            N, T, D = data.observed.features['keypoints'].shape
            assert (D - 1) % 2 == 0
            n_coords = (D - 1) // 2
            bboxes_list.append(data.observed.features['keypoints'][:, :, :-1].reshape(N, T * n_coords, 2))  # (N, T * n_coords, 2)
            masks_list.append(data.observed.mask.repeat(1, n_coords))
            bboxes_list.append(data.unobserved.features['keypoints'][:, :-1].unsqueeze(1).reshape(N, n_coords, 2))  # (N, n_coords, 2)
            masks_list.append(data.unobserved.mask.unsqueeze(1).repeat(1, n_coords))

        # Concatenate all bboxes and masks
        all_bboxes = torch.cat(bboxes_list, dim=1)  # (N, X, 2)
        all_masks = torch.cat(masks_list, dim=1)  # (N, X)

        # Compute min and max values
        valid_bboxes = all_bboxes[~all_masks].view(-1, 2)
        if valid_bboxes.shape[0] == 0:
            return data
        min_val = valid_bboxes.min(dim=0).values
        max_val = valid_bboxes.max(dim=0).values
        scale = max_val - min_val
        scale[scale == 0] = 1  # prevent division by zero

        if 'bbox' in data.observed.features:
            # Update bboxes
            data.observed.features['bbox'][:, :, :2] = (data.observed.features['bbox'][:, :, :2] - min_val) / scale
            data.unobserved.features['bbox'][:, :2] = (data.unobserved.features['bbox'][:, :2] - min_val) / scale
            data.observed.features['bbox'][:, :, 2:4] = (data.observed.features['bbox'][:, :, 2:4] - min_val) / scale
            data.unobserved.features['bbox'][:, 2:4] = (data.unobserved.features['bbox'][:, 2:4] - min_val) / scale

            # Zero-out masked entries
            data.observed.features['bbox'][data.observed.mask] = 0
            data.unobserved.features['bbox'][data.unobserved.mask] = 0

        if 'keypoints' in data.observed.features:
            N, T, D = data.observed.features['keypoints'].shape
            assert (D - 1) % 2 == 0
            n_coords = (D - 1) // 2

            # Rescale keypoints
            observed_keypoints = data.observed.features['keypoints'][:, :, :-1].reshape(N, T * n_coords, 2)
            unobserved_keypoints = data.unobserved.features['keypoints'][:, :-1].unsqueeze(1).reshape(N, n_coords, 2)
            observed_keypoints = (observed_keypoints - min_val) / scale
            unobserved_keypoints = (unobserved_keypoints - min_val) / scale

            # Update keypoints
            data.observed.features['keypoints'][:, :, :-1] = observed_keypoints.reshape(N, T, n_coords * 2)
            data.unobserved.features['keypoints'][:, :-1] = unobserved_keypoints.reshape(N, n_coords * 2)

            # Zero-out masked entries
            data.observed.features['keypoints'][data.observed.mask] = 0
            data.unobserved.features['keypoints'][data.unobserved.mask] = 0

        return data


def test_bbox_min_scaling():
    # Setup: 2 tracks, 2 time steps
    observed_bboxes = torch.tensor([
        [[0.1, 0.1, 0.2, 0.2, 0.9], [0.1, 0.1, 0.2, 0.4, 0.8]],
        [[0.2, 0.2, 0.3, 0.3, 0.7], [0., 0., 0, 0, 0]]
    ])  # shape: (2, 2, 5)

    unobserved_bboxes = torch.tensor([
        [0.2, 0.2, 0.4, 0.4, 0.85],
        [1., 1., 1., 1., 0.0]
    ])  # shape: (2, 2, 5)

    observed_mask = torch.tensor([
        [False, False],
        [False, True]
    ])  # all valid

    unobserved_mask = torch.tensor([
        False,
        True
    ])  # second time step masked

    data = VideoClipData(
        observed=VideoClipPart(
            ids=None,
            ts=None,
            mask=observed_mask,
            features={
                'bbox': observed_bboxes
            }
        ),
        unobserved=VideoClipPart(
            ids=None,
            ts=None,
            mask=unobserved_mask,
            features={
                'bbox': unobserved_bboxes
            }
        )
    )

    transform = BBoxMinMaxScaling()
    transformed_data = transform(data)
    print(transformed_data)


if __name__ == '__main__':
    test_bbox_min_scaling()
