import random
from typing import Dict, List, Set

import enum
import torch

from tdlp.datasets.dataset.common.data import VideoClipPart
from tdlp.datasets.dataset.feature_extractor.feature_extractor import FeatureExtractor
from tdlp.datasets.dataset.index.index import DatasetIndex
from tdlp.utils.extra_features import ExtraFeaturesReader


class SupportedFeatures(enum.Enum):
    BBOX = 'bbox'
    KEYPOINTS = 'keypoints'
    APPEARANCE = 'appearance'


class PredictionBBoxFeatureExtractor(FeatureExtractor):
    BBOX_DIM = 5  # XYWHC
    KEYPOINTS_DIM = 35  # 17x2 XYC (per part) + 1 C (global) = 52
    APPEARANCE_NUM_PARTS = 6
    APPEARANCE_PER_PART_DIM = 129 # 6x129 = 774
    SUPPORTED_FEATURES = {
        SupportedFeatures.BBOX, 
        SupportedFeatures.KEYPOINTS, 
        SupportedFeatures.APPEARANCE,
    }

    WORKER_ID_STEP = 1_000_000

    def __init__(
        self,
        index: DatasetIndex,
        object_id_mapping: Dict[str, int],
        n_tracks: int,
        prediction_path: str,
        feature_names: List[str],
        extra_false_positives: bool = True,
        random_appearance_jitter_ratio: float = 0.0,
        random_appearance_jitter_range: int = 0
    ):
        super().__init__(
            index=index,
            object_id_mapping=object_id_mapping,
            n_tracks=n_tracks
        )
        is_train = (index.split == 'train')
        feature_names = [SupportedFeatures(feature_name.lower()) for feature_name in feature_names]
        for feature_name in feature_names:
            assert feature_name in self.SUPPORTED_FEATURES, \
                f'Unsupported feature "{feature_name}". Supported features: {self.SUPPORTED_FEATURES}'

        self._feature_names = set(feature_names)
        self._extra_features_reader = ExtraFeaturesReader(prediction_path)

        # Augmentations
        self._extra_false_positives = extra_false_positives
        self._random_appearance_jitter_ratio = random_appearance_jitter_ratio if is_train else 0.0
        self._random_appearance_jitter_range = random_appearance_jitter_range if is_train else 0

        self._worker_id_counter = 0

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

    @staticmethod
    def initialize_features(feature_names: Set[SupportedFeatures], n_tracks: int, temporal_length: int, is_train: bool = False) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        if SupportedFeatures.BBOX in feature_names:
            features[SupportedFeatures.BBOX.value] = \
                torch.zeros(n_tracks, temporal_length, PredictionBBoxFeatureExtractor.BBOX_DIM, dtype=torch.float32)
        if SupportedFeatures.KEYPOINTS in feature_names:
            features[SupportedFeatures.KEYPOINTS.value] = \
                torch.zeros(n_tracks, temporal_length, PredictionBBoxFeatureExtractor.KEYPOINTS_DIM, dtype=torch.float32)
        if SupportedFeatures.APPEARANCE in feature_names:
            features[SupportedFeatures.APPEARANCE.value] = \
                torch.zeros(n_tracks, temporal_length, PredictionBBoxFeatureExtractor.APPEARANCE_NUM_PARTS, PredictionBBoxFeatureExtractor.APPEARANCE_PER_PART_DIM, dtype=torch.float32)

        return features

    @staticmethod
    def set_features(
        feature_names: Set[SupportedFeatures], 
        features: Dict[str, torch.Tensor], 
        object_index: int, 
        clip_index: int, 
        data: dict
    ) -> None:
        if SupportedFeatures.BBOX in feature_names:
            bbox = [*data['bbox_xywh'], data['bbox_conf']]
            features[SupportedFeatures.BBOX.value][object_index, clip_index, :] = torch.tensor(bbox, dtype=torch.float32)
        if SupportedFeatures.KEYPOINTS in feature_names:
            keypoints = sum([d[:2] for d in data['keypoints_xyc']], []) + [data['keypoints_conf']]
            features[SupportedFeatures.KEYPOINTS.value][object_index, clip_index, :] = torch.tensor(keypoints, dtype=torch.float32)
        if SupportedFeatures.APPEARANCE in feature_names:
            embs = [[*emb, visibility] for emb, visibility in zip(data['appearance_embeddings'], data['appearance_visibility'])]
            features[SupportedFeatures.APPEARANCE.value][object_index, clip_index, :] = torch.tensor(embs, dtype=torch.float32)

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
        video_clip_part.mask.fill_(True)  # Override mask

        features: Dict[str, torch.Tensor] = self.initialize_features(
            feature_names=self._feature_names,
            n_tracks=self._n_tracks,
            temporal_length=temporal_length
        )

        for clip_index, frame_index in enumerate(range(start_index, end_index)):
            extra_data = self._extra_features_reader.read(scene_name, frame_index)
            object_id_to_extra_data_lookup: Dict[str, dict] = {raw['object_id']: raw for raw in extra_data if raw['object_id'] is not None}

            for object_index, object_id in enumerate(object_ids):
                data = object_id_to_extra_data_lookup.get(object_id)
                if data is None:
                    continue

                self.set_features(
                    feature_names=self._feature_names,
                    features=features,
                    object_index=object_index,
                    clip_index=clip_index,
                    data=data
                )
                video_clip_part.mask[object_index, clip_index] = False

            # Add extra false positives (specific augmentation type)
            if not observed and self._extra_false_positives:
                extra_false_positives = [raw for raw in extra_data if raw['object_id'] is None]
                random.shuffle(extra_false_positives)
                n_extra_positives = min(self._n_tracks - n_object_ids, len(extra_false_positives))
                extra_false_positives = extra_false_positives[:n_extra_positives]
                for data_index, object_index in enumerate(range(n_object_ids, n_object_ids + n_extra_positives)):
                    data = extra_false_positives[data_index]
                    self.set_features(
                        feature_names=self._feature_names,
                        features=features,
                        object_index=object_index,
                        clip_index=clip_index,
                        data=data
                    )
                    video_clip_part.mask[object_index, clip_index] = False

                    # Setting ID is complex as it need has to be unique and not match any dataset ID
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else 0
                    self._worker_id_counter = (self._worker_id_counter + 1) % self.WORKER_ID_STEP
                    next_id = self.WORKER_ID_STEP * worker_id + self._worker_id_counter
                    video_clip_part.ids[object_index, clip_index] = next_id

            if self._random_appearance_jitter_ratio > 0 and SupportedFeatures.APPEARANCE in self._feature_names:
                scene_info = self._index.get_scene_info(scene_name)
                jitter = random.randint(-self._random_appearance_jitter_range, self._random_appearance_jitter_range)
                random_frame_index = max(0, min(scene_info.seqlength - 1, frame_index + jitter))
                aug_extra_data = self._extra_features_reader.read(scene_name, random_frame_index)
                aug_object_id_to_extra_data_lookup: Dict[str, dict] = {raw['object_id']: raw for raw in aug_extra_data if raw['object_id'] is not None}

                for object_index, object_id in enumerate(object_ids):
                    data = aug_object_id_to_extra_data_lookup.get(object_id)
                    if data is None:
                        continue

                    r = random.random()
                    if r < self._random_appearance_jitter_ratio:
                        self.set_features(
                            feature_names=[SupportedFeatures.APPEARANCE],
                            features=features,
                            object_index=object_index,
                            clip_index=clip_index,
                            data=data
                        )

        video_clip_part.features = features
        return video_clip_part
