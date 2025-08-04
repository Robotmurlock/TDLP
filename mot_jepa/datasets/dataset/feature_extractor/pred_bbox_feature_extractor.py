import random
from typing import Dict, List, Optional

import torch

from mot_jepa.datasets.dataset.common.data import VideoClipPart
from mot_jepa.datasets.dataset.feature_extractor.feature_extractor import FeatureExtractor
from mot_jepa.datasets.dataset.index.index import DatasetIndex
from mot_jepa.utils.extra_features import ExtraFeaturesReader


class PredictionBBoxFeatureExtractor(FeatureExtractor):
    BBOX_DIM = 5

    def __init__(
        self,
        index: DatasetIndex,
        object_id_mapping: Dict[str, int],
        n_tracks: int,
        prediction_path: str,
        extra_false_positives: bool = True
    ):
        super().__init__(
            index=index,
            object_id_mapping=object_id_mapping,
            n_tracks=n_tracks
        )
        self._extra_features_reader = ExtraFeaturesReader(prediction_path)
        self._extra_false_positives = extra_false_positives

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
        end_index = start_index + temporal_length

        object_ids = sorted(self._index.get_objects_present_in_scene_clip(scene_name, start_index, end_index))
        n_object_ids = len(object_ids)
        bboxes = torch.zeros(self._n_tracks, temporal_length, self.BBOX_DIM, dtype=torch.float32)
        video_clip_part.mask.fill_(True)  # Override mask

        for clip_index, frame_index in enumerate(range(start_index, end_index)):
            extra_data = self._extra_features_reader.read(scene_name, frame_index)
            object_id_to_extra_data_lookup: Dict[str, dict] = {raw['object_id']: raw for raw in extra_data if raw['object_id'] is not None}

            for object_index, object_id in enumerate(object_ids):
                data = object_id_to_extra_data_lookup.get(object_id)
                if data is None:
                    continue

                bboxes[object_index, clip_index, :] = self.bbox_to_tensor(data['bbox_xywh'], data['bbox_conf'])
                video_clip_part.mask[object_index, clip_index] = False

            # Add extra false positives (specific augmentation type)
            if not observed and self._extra_false_positives:
                extra_false_positives = [raw for raw in extra_data if raw['object_id'] is None]
                random.shuffle(extra_false_positives)
                n_extra_positives = min(self._n_tracks - n_object_ids, len(extra_false_positives))
                extra_false_positives = extra_false_positives[:n_extra_positives]
                for data_index, object_index in enumerate(range(n_object_ids, n_object_ids + n_extra_positives)):
                    data = extra_false_positives[data_index]
                    bbox = self.bbox_to_tensor(data['bbox_xywh'], data['bbox_conf'])

                    bboxes[object_index, clip_index, :] = bbox
                    video_clip_part.mask[object_index, clip_index] = False

        video_clip_part.features['bbox'] = bboxes
        return video_clip_part
