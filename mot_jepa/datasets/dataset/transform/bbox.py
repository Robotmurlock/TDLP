from typing import List

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipData
from mot_jepa.datasets.dataset.transform.base import Transform


class BBoxXYWHtoXYXY(Transform):
    def __init__(self):
        super().__init__(name='bbox_xywh_to_xyxy')

    def apply(self, data: VideoClipData) -> VideoClipData:
        data.observed_bboxes[..., 2:4] = data.observed_bboxes[..., :2] + data.observed_bboxes[..., 2:4]
        data.unobserved_bboxes[..., 2:4] = data.unobserved_bboxes[..., :2] + data.unobserved_bboxes[..., 2:4]
        return data


class BBoxStandardization(Transform):
    def __init__(
        self,
        coord_mean: List[float],
        coord_std: List[float]
    ):
        super().__init__(name='bbox_fod_standardization')

        # Validation
        assert len(coord_mean) == 5
        assert len(coord_std) == 5

        self._coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        self._coord_std = torch.tensor(coord_std, dtype=torch.float32)

    def apply(self, data: VideoClipData) -> VideoClipData:
        data.observed_bboxes = (data.observed_bboxes - self._coord_mean) / self._coord_std  # Centralize
        data.unobserved_bboxes = (data.unobserved_bboxes - self._coord_mean) / self._coord_std  # Centralize
        data.observed_bboxes[data.observed_temporal_mask] = 0
        data.unobserved_bboxes[data.unobserved_temporal_mask] = 0

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
        if self._fod_time_scaled:
            bboxes = data.observed_bboxes
            mask = data.observed_temporal_mask
            fod = torch.zeros_like(bboxes)
            ts = data.observed_ts.unsqueeze(-1).repeat(1, 1, bboxes.shape[-1])
            for n in range(bboxes.shape[0]):
                ts_n = ts[n][~mask[n]]
                bboxes_n = bboxes[n][~mask[n]]
                if bboxes_n.shape[0] == 0:
                    continue

                bboxes_n[1:, :] = (bboxes_n[1:, :] - bboxes_n[:-1, :]) / (ts_n[1:, :] - ts_n[:-1, :])
                bboxes_n[0, :] = 0
                fod[n][~mask[n]] = bboxes_n
        else:
            fod = torch.zeros_like(data.observed_bboxes)
            fod[:, 1:, :] = (data.observed_bboxes[:, 1:, :] - data.observed_bboxes[:, :-1, :] - self._fod_mean) / self._fod_std
            fod[:, 1:, :] = fod[:, 1:, :] * (1 - data.observed_temporal_mask[:, :-1].unsqueeze(-1).repeat(1, 1, data.observed_bboxes.shape[-1]).float())

        data.observed_bboxes = (data.observed_bboxes - self._coord_mean) / self._coord_std  # Centralize
        data.unobserved_bboxes = (data.unobserved_bboxes - self._coord_mean) / self._coord_std  # Centralize

        data.observed_bboxes = torch.cat([data.observed_bboxes, fod], dim=-1)
        data.observed_bboxes[data.observed_temporal_mask] = 0
        data.unobserved_bboxes[data.unobserved_temporal_mask] = 0

        return data


class BBoxMinMaxScaling(Transform):
    def __init__(self):
        super().__init__(name='bbox_min_max_scaling')

    def apply(self, data: VideoClipData) -> VideoClipData:
        # Concatenate all bboxes and masks
        all_bboxes = torch.cat([
            data.observed_bboxes,                # (N, T, 5)
            data.unobserved_bboxes.unsqueeze(1)  # (N, 1, 5)
        ], dim=1).view(-1, data.observed_bboxes.shape[-1])  # (N * (T + 1), 5)

        all_masks = torch.cat([
            data.observed_temporal_mask,                # (N, T)
            data.unobserved_temporal_mask.unsqueeze(1)  # (N, 1)
        ], dim=1).view(-1)  # (N * (T + 1))

        # Invert masks: 0 = valid, 1 = invalid
        valid_mask = ~all_masks

        valid_bboxes = all_bboxes[valid_mask][:, :4].reshape(-1, 2)
        min_val = valid_bboxes.min(dim=0).values
        max_val = valid_bboxes.max(dim=0).values
        scale = max_val - min_val
        scale[scale == 0] = 1  # prevent division by zero

        # Apply scaling
        data.observed_bboxes[:, :, :2] = (data.observed_bboxes[:, :, :2] - min_val) / scale
        data.unobserved_bboxes[:, :2] = (data.unobserved_bboxes[:, :2] - min_val) / scale
        data.observed_bboxes[:, :, 2:4] = (data.observed_bboxes[:, :, 2:4] - min_val) / scale
        data.unobserved_bboxes[:, 2:4] = (data.unobserved_bboxes[:, 2:4] - min_val) / scale

        # Zero-out masked entries
        data.observed_bboxes[data.observed_temporal_mask] = 0
        data.unobserved_bboxes[data.unobserved_temporal_mask] = 0

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
        observed_bboxes=observed_bboxes,
        observed_temporal_mask=observed_mask,
        observed_ts=None,
        unobserved_bboxes=unobserved_bboxes,
        unobserved_ts=None,
        unobserved_temporal_mask=unobserved_mask
    )

    transform = BBoxMinMaxScaling()
    transformed_data = transform(data)
    print(transformed_data)


if __name__ == '__main__':
    test_bbox_min_scaling()
