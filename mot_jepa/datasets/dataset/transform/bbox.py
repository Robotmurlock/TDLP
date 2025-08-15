from typing import List

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipData, VideoClipPart
from mot_jepa.datasets.dataset.transform.base import Transform


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
        coord_mean: List[float],
        coord_std: List[float]
    ):
        super().__init__(name='bbox_standardization')

        # Validation
        assert len(coord_mean) == 5
        assert len(coord_std) == 5

        self._coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        self._coord_std = torch.tensor(coord_std, dtype=torch.float32)

    def apply(self, data: VideoClipData) -> VideoClipData:
        data.observed.features['bbox'] = (data.observed.features['bbox'] - self._coord_mean) / self._coord_std  # Centralize
        data.unobserved.features['bbox'] = (data.unobserved.features['bbox'] - self._coord_mean) / self._coord_std  # Centralize
        data.observed.features['bbox'][data.observed.mask] = 0
        data.unobserved.features['bbox'][data.unobserved.mask] = 0

        return data


class BBoxFODStandardization(Transform):
    def __init__(
        self,
        coord_mean: List[float],
        coord_std: List[float],
        fod_mean: List[float],
        fod_std: List[float],
        fod_time_scaled: bool = False
    ):
        super().__init__(name='bbox_fod_standardization')

        # Validation
        assert len(coord_mean) == 5
        assert len(coord_std) == 5
        assert len(fod_mean) == 5
        assert len(fod_std) == 5

        self._coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        self._coord_std = torch.tensor(coord_std, dtype=torch.float32)
        self._fod_mean = torch.tensor(fod_mean, dtype=torch.float32)
        self._fod_std = torch.tensor(fod_std, dtype=torch.float32)
        self._fod_time_scaled = fod_time_scaled

    def apply(self, data: VideoClipData) -> VideoClipData:
        bboxes = data.observed.features['bbox']
        mask = data.observed.mask

        if self._fod_time_scaled:
            fod = torch.zeros_like(bboxes)
            ts = data.observed.ts.unsqueeze(-1).repeat(1, 1, bboxes.shape[-1])
            for n in range(bboxes.shape[0]):
                ts_n = ts[n][~mask[n]]
                bboxes_n = bboxes[n][~mask[n]]
                if bboxes_n.shape[0] == 0:
                    continue

                bboxes_n[1:, :] = (bboxes_n[1:, :] - bboxes_n[:-1, :]) / (ts_n[1:, :] - ts_n[:-1, :])
                bboxes_n[0, :] = 0
                fod[n][~mask[n]] = bboxes_n
        else:
            fod = torch.zeros_like(bboxes)
            fod[:, 1:, :] = bboxes[:, 1:, :] - bboxes[:, :-1, :]

        # FoD standardization
        fod = (fod - self._fod_mean) / self._fod_std
        extended_mask = mask[:, :-1].unsqueeze(-1).repeat(1, 1, bboxes.shape[-1]).float()
        fod[:, 1:, :] = fod[:, 1:, :] * (1 - extended_mask)

        bboxes = (bboxes - self._coord_mean) / self._coord_std  # Centralize
        data.observed.features['bbox'] = torch.cat([bboxes, fod], dim=-1)  # Add FOD
        data.observed.features['bbox'][data.observed.mask] = 0  # Cleanup

        data.unobserved.features['bbox'] = (data.unobserved.features['bbox'] - self._coord_mean) / self._coord_std  # Centralize
        data.unobserved.features['bbox'][data.unobserved.mask] = 0  # Cleanup

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
