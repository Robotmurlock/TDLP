from typing import List

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipData
from mot_jepa.datasets.dataset.transform.base import Transform


class BBoxStandardization(Transform):
    def __init__(
        self,
        coord_mean: List[float],
        coord_std: List[float],
        xywh_to_xyxy: bool = True
    ):
        super().__init__(name='bbox_fod_standardization')

        # Validation
        assert len(coord_mean) == 5
        assert len(coord_std) == 5

        self._coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        self._coord_std = torch.tensor(coord_std, dtype=torch.float32)
        self._xywh_to_xyxy = xywh_to_xyxy

    def apply(self, data: VideoClipData) -> VideoClipData:
        if self._xywh_to_xyxy:
            data.observed_bboxes[..., 2:4] = data.observed_bboxes[..., :2] + data.observed_bboxes[..., 2:4]
            data.unobserved_bboxes[..., 2:4] = data.unobserved_bboxes[..., :2] + data.unobserved_bboxes[..., 2:4]

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
        xywh_to_xyxy: bool = True
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
        self._xywh_to_xyxy = xywh_to_xyxy

    def apply(self, data: VideoClipData) -> VideoClipData:
        if self._xywh_to_xyxy:
            data.observed_bboxes[..., 2:4] = data.observed_bboxes[..., :2] + data.observed_bboxes[..., 2:4]
            data.unobserved_bboxes[..., 2:4] = data.unobserved_bboxes[..., :2] + data.unobserved_bboxes[..., 2:4]

        fod = torch.zeros_like(data.observed_bboxes)
        fod[:, 1:, :] = (data.observed_bboxes[:, 1:, :] - data.observed_bboxes[:, :-1, :] - self._fod_mean) / self._fod_std
        fod[:, 1:, :] = fod[:, 1:, :] * (1 - data.observed_temporal_mask[:, :-1].unsqueeze(-1).repeat(1, 1, data.observed_bboxes.shape[-1]).float())

        data.observed_bboxes = (data.observed_bboxes - self._coord_mean) / self._coord_std  # Centralize
        data.unobserved_bboxes = (data.unobserved_bboxes - self._coord_mean) / self._coord_std  # Centralize

        data.observed_bboxes = torch.cat([data.observed_bboxes, fod], dim=-1)
        data.observed_bboxes[data.observed_temporal_mask] = 0
        data.unobserved_bboxes[data.unobserved_temporal_mask] = 0

        return data