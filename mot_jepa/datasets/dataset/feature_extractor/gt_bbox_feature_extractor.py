from typing import Dict, List

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipPart
from mot_jepa.datasets.dataset.feature_extractor.feature_extractor import FeatureExtractor
from mot_jepa.datasets.dataset.index.index import DatasetIndex


class GTBBoxFeatureExtractor(FeatureExtractor):
    BBOX_DIM = 5

    def __init__(
        self,
        index: DatasetIndex,
        object_id_mapping: Dict[str, int],
        n_tracks: int
    ):
        super().__init__(
            index=index,
            object_id_mapping=object_id_mapping,
            n_tracks=n_tracks
        )

    @staticmethod
    def bbox_to_tensor(bbox: List[float], score: float) -> torch.Tensor:
        """
        Convert BBox to torch tensor.

        Args:
            bbox: BBox
            score: Detection confidence score

        Returns:
            BBox tensor
        """
        return torch.tensor([*bbox, score], dtype=torch.float32)

    def _extract_extra_data(
        self,
        video_clip_part: VideoClipPart,
        scene_name: str,
        start_index: int,
        temporal_length: int,
        observed: bool
    ) -> VideoClipPart:
        _ = observed

        end_index = start_index + temporal_length
        object_ids = sorted(self._index.get_objects_present_in_scene_clip(scene_name, start_index, end_index))
        bboxes = torch.zeros(self._n_tracks, temporal_length, self.BBOX_DIM, dtype=torch.float32)

        for clip_index, frame_index in enumerate(range(start_index, end_index)):
            for object_index, object_id in enumerate(object_ids):
                data = self._index.get_object_data_label_by_frame_index(object_id, frame_index)
                if data is None:
                    continue

                assert not video_clip_part.mask[object_index, clip_index].item()
                bboxes[object_index, clip_index, :] = self.bbox_to_tensor(data.bbox, data.score)

        video_clip_part.features['bbox'] = bboxes
        return video_clip_part
