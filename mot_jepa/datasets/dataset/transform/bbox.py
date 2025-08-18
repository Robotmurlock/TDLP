from typing import List, Dict

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipData, VideoClipPart
from mot_jepa.datasets.dataset.transform.base import Transform
from mot_jepa.datasets.dataset.transform.utils import expand_pattern


class BBoxXYWHtoXYXY(Transform):
    def __init__(self):
        super().__init__(name='bbox_xywh_to_xyxy')

    def apply(self, data: VideoClipData) -> VideoClipData:
        data.observed.features['bbox'][..., 2:4] = data.observed.features['bbox'][..., :2] + data.observed.features['bbox'][..., 2:4]
        data.unobserved.features['bbox'][..., 2:4] = data.unobserved.features['bbox'][..., :2] + data.unobserved.features['bbox'][..., 2:4]
        return data


class BBoxStandardization(Transform):
    def __init__(
        self,
        coord_mean: Dict[str, List[float]],
        coord_std: Dict[str, List[float]],
    ):
        super().__init__(name='feature_standardization')

        assert set(coord_mean.keys()) == set(coord_std.keys())

        self._feature_names = list(coord_mean.keys())
        self._coord_mean = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_mean.items()}
        self._coord_std = {k: torch.tensor(expand_pattern(v), dtype=torch.float32) for k, v in coord_std.items()}

    def apply(self, data: VideoClipData) -> VideoClipData:
        for feature_name in self._feature_names:
            data.observed.features[feature_name] = \
                (data.observed.features[feature_name] - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.unobserved.features[feature_name] = \
                (data.unobserved.features[feature_name] - self._coord_mean[feature_name]) / self._coord_std[feature_name]  # Centralize
            data.observed.features[feature_name][data.observed.mask] = 0
            data.unobserved.features[feature_name][data.unobserved.mask] = 0

        return data


class FeatureFODStandardization(Transform):
    def __init__(
        self,
        coord_mean: Dict[str, List[float]],
        coord_std: Dict[str, List[float]],
        fod_mean: Dict[str, List[float]],
        fod_std: Dict[str, List[float]],
        fod_time_scaled: bool = False
    ):
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

                    features_n[1:, :] = (features_n[1:, :] - features_n[:-1, :]) / (ts_n[1:, :] - ts_n[:-1, :])
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
    def __init__(self):
        super().__init__(name='bbox_min_max_scaling')

    def apply(self, data: VideoClipData) -> VideoClipData:
        # Concatenate all bboxes and masks
        all_bboxes = torch.cat([
            data.observed.features['bbox'],                # (N, T, 5)
            data.unobserved.features['bbox'].unsqueeze(1)  # (N, 1, 5)
        ], dim=1).view(-1, data.observed.features['bbox'].shape[-1])  # (N * (T + 1), 5)

        all_masks = torch.cat([
            data.observed.mask,                # (N, T)
            data.unobserved.mask.unsqueeze(1)  # (N, 1)
        ], dim=1).view(-1)  # (N * (T + 1))

        # Invert masks: 0 = valid, 1 = invalid
        valid_mask = ~all_masks

        valid_bboxes = all_bboxes[valid_mask][:, :4].reshape(-1, 2)
        min_val = valid_bboxes.min(dim=0).values
        max_val = valid_bboxes.max(dim=0).values
        scale = max_val - min_val
        scale[scale == 0] = 1  # prevent division by zero

        # Apply scaling
        data.observed.features['bbox'][:, :, :2] = (data.observed.features['bbox'][:, :, :2] - min_val) / scale
        data.unobserved.features['bbox'][:, :2] = (data.unobserved.features['bbox'][:, :2] - min_val) / scale
        data.observed.features['bbox'][:, :, 2:4] = (data.observed.features['bbox'][:, :, 2:4] - min_val) / scale
        data.unobserved.features['bbox'][:, 2:4] = (data.unobserved.features['bbox'][:, 2:4] - min_val) / scale

        # Zero-out masked entries
        data.observed.features['bbox'][data.observed.mask] = 0
        data.unobserved.features['bbox'][data.unobserved.mask] = 0

        if 'keypoints' in data.observed.features:
            data.observed.features['keypoints'][:, :, :2] = (data.observed.features['keypoints'][:, :, :2] - min_val) / scale
            data.unobserved.features['keypoints'][:, :2] = (data.unobserved.features['keypoints'][:, :2] - min_val) / scale

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
