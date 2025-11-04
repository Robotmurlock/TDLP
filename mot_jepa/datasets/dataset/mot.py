"""
Dataset structure:
- Dataset consists of scenes i.e., videos
- Each scene has multiple tracks
- Each track has multiple times, bounding boxes

Dataset sample should be a clip from some scene with tensors:
- BBox tensor (N, T, Be), N is number of tracks, T is number of frames, Be is bbox feature dimension
    - Be should be (x, y, w, h, s), i.e. bbox coordinates and score
- Time tensor (N, T) with clip times
- Mask tensor (N, T) of bools (missing tracks and missing track's times)
"""
import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from mot_jepa.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import SupportedFeatures
from torch.utils.data import Dataset

from mot_jepa.datasets.dataset.augmentations import Augmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData
from mot_jepa.datasets.dataset.feature_extractor import GTBBoxFeatureExtractor, feature_extractor_factory, FeatureExtractor
from mot_jepa.datasets.dataset.index.index import DatasetIndex
from mot_jepa.datasets.dataset.transform import Transform

logger = logging.getLogger('MOTClipDataset')


class MOTClipDataset(Dataset):
    """
    MOT video clip-based dataset. It samples random video clips based on the dataset frame index.
    """
    BBOX_DIM = 5

    def __init__(
        self,
        index: DatasetIndex,
        n_tracks: int,
        clip_length: int,
        transform: Transform,
        augmentations: Augmentation,
        min_clip_tracks: int = 1,
        clip_sampling_step: int = 1,
        feature_extractor_type: Optional[str] = None,
        feature_extractor_params: Optional[dict] = None
    ):
        """
        Args:
            index: Dataset index
            n_tracks: Number of tracks per clip (maximum number of IDs)
            transform: Data transform function
            augmentations: Data augmentations function
            clip_length: Clip length
            min_clip_tracks: Minimum number of tracks (IDs) in a clip
            clip_sampling_step: Clip sampling step (subsamples dataset)
        """
        self._index = index
        self._is_train = (index.split == 'train')

        # Track parameters
        max_tracks = index.get_max_tracks()
        logger.info(f'Maximum number of tracks over all scenes: {max_tracks}.')

        if n_tracks < max_tracks:
            logger.warning(f'Number of tracks ({n_tracks}) is lower than maximum '
                           f'number of tracks ({max_tracks}) in a scene.')
        self._n_tracks = n_tracks

        # Temporal parameters
        self._clip_length = clip_length

        # Create dataset index
        self._clip_index = self._create_clip_index(
            index=index,
            clip_length=clip_length,
            min_clip_tracks=min_clip_tracks,
            clip_sampling_step=clip_sampling_step
        )

        # Create mapping (object_id -> unique number)
        self._id_lookup = self._create_id_lookup(index)

        # Transforms
        self._transform = transform
        self._augmentations = augmentations

        # Features
        assert (feature_extractor_type is None) == (feature_extractor_params is None), \
            f'Either set both type and params for feature extractor or neither!'
        if feature_extractor_type is None:
            logger.warning(f'Feature extractor not set, using {GTBBoxFeatureExtractor.__name__} as the default one!')
            self._feature_extractor = GTBBoxFeatureExtractor(
                index=index,
                object_id_mapping=self._id_lookup,
                n_tracks=n_tracks
            )
        else:
            self._feature_extractor = feature_extractor_factory(
                extractor_type=feature_extractor_type,
                extractor_params=feature_extractor_params,
                index=index,
                object_id_mapping=self._id_lookup,
                n_tracks=n_tracks,
            )

        logger.info(f'Number of sampled clips: {len(self._clip_index)}.')

    @property
    def index(self) -> DatasetIndex:
        """
        Returns: Dataset index
        """
        return self._index

    @property
    def feature_extractor(self) -> FeatureExtractor:
        """
        Returns: Feature extractor
        """
        return self._feature_extractor

    @staticmethod
    def _create_clip_index(
        index: DatasetIndex,
        clip_length: int,
        min_clip_tracks: int = 1,
        clip_sampling_step: int = 1
    ) -> List[Tuple[str, int, int]]:
        """
        Creates dataset CLIP index. Each index element is a tuple containing:
            - Scene info (object)
            - Clip's start frame index
            - Clip's end frame index

        Args:
            index: Dataset index
            clip_length: Clip length
            min_clip_tracks: Minimum number of tracks (IDs) in a clip
            clip_sampling_step: Clip sampling step (subsamples dataset)

        Returns:
            Clip index
        """
        # Result
        clip_index: List[Tuple[str, int, int]] = []

        # Stats
        n_skipped = 0
        n_total = 0

        for scene in index.scenes:
            seqlength = index.get_scene_info(scene).seqlength

            clip_start_time_point_candidates = list(range(0, seqlength, clip_sampling_step))
            for start_index in clip_start_time_point_candidates:
                end_index = start_index + clip_length + 1
                if end_index > seqlength:  # Out of range
                    continue

                n_total += 1

                if len(index.get_objects_present_in_scene_clip(scene, start_index, end_index)) <= min_clip_tracks:
                    n_skipped += 1
                    continue
                clip_index.append((scene, start_index, start_index + clip_length + 1))

        logger.info(f'Sampled total number of {n_total} clips ')

        return clip_index

    @property
    def scene_names_per_frame(self) -> List[str]:
        return [scene_name for scene_name, _, _ in self._clip_index]

    @staticmethod
    def _create_id_lookup(index: DatasetIndex) -> Dict[str, int]:
        """
        Creates mapping (object_id -> unique number), as numbers are more convenient for training.

        Args:
            index: DatasetIndex

        Returns:
            Mapping (object_id -> unique number)
        """
        object_ids: List[str] = []
        for scene_name in index.scenes:
            object_ids.extend(index.get_objects_present_in_scene(scene_name))

        assert len(object_ids) == len(set(object_ids)), f'Found unexpected object duplicates!'
        object_ids = sorted(object_ids)
        return {object_id: i for i, object_id in enumerate(object_ids)}

    def __len__(self) -> int:
        return len(self._clip_index)

    def get_raw(self, index: int) -> VideoClipData:
        """
        Get raw clip data (without any transformations).

        Full data structure: {
            observed: # NOTE: This part (or the unobserved is extracted and returned)
                ids: ...
                ts: ...
                mask: ...
                features: {
                    bbox_xywh: ...
                    bbox_conf: ...
                    keypoints_xyc: ...
                    keypoints_conf: ...
                    appearance: ...
                }
            unobserved: ...
        }

        Notes:
            - if `use_extra_data=False`, then only X.features.bbox_xywh are available!

        Args:
            index: Dataset index

        Returns:
            Clip observed and unobserved bboxes, timestamps and temporal masks (6 elements)
        """
        scene_name, start_index, end_index = self._clip_index[index]

        observed_end_index = start_index + self._clip_length

        return self._feature_extractor(
            scene_name=scene_name,
            observed_start_index=start_index,
            observed_start_time=0,
            observed_temporal_length=self._clip_length,
            unobserved_start_index=observed_end_index,
            unobserved_start_time=self._clip_length,
            unobserved_temporal_length=1,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.get_raw(index)
        data = self._augmentations(data)
        data = self._transform(data)
        return data.serialize()

    def visualize_scene(self, index: int) -> np.ndarray:
        scene_name, start_index, end_index = self._clip_index[index]
        scene_image_path = self._index.get_scene_image_path(scene_name, end_index)
        raw = self.get_raw(index)

        # noinspection PyUnresolvedReferences
        image = cv2.imread(scene_image_path)
        h, w, _ = image.shape
        assert image is not None, f'Failed to load image "{scene_image_path}".'

        # Visualize detections
        unobserved_bboxes = raw.unobserved.features['bbox']
        unobserved_temporal_mask = raw.unobserved.mask
        for obj_index in range(unobserved_bboxes.shape[0]):
            if bool(unobserved_temporal_mask[obj_index].bool().item()):
                continue

            bbox = unobserved_bboxes[obj_index].numpy().tolist()

            bbox = [
                round(bbox[0] * w),
                round(bbox[1] * h),
                round((bbox[0] + bbox[2]) * w),
                round((bbox[1] + bbox[3]) * h)
            ]
            # noinspection PyUnresolvedReferences
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        # Visualize track history
        observed_bboxes = raw.observed.features['bbox']
        observed_temporal_mask = raw.observed.mask
        for obj_index in range(observed_bboxes.shape[0]):
            points = []
            for temporal_index in range(observed_bboxes.shape[1]):
                if bool(observed_temporal_mask[obj_index, temporal_index].bool().item()):
                    continue

                bbox = observed_bboxes[obj_index, temporal_index].numpy().tolist()
                center_point = [round((bbox[0] + bbox[2] / 2) * w), round((bbox[1] + bbox[3] / 2) * h)]
                points.append(center_point)

            points = np.array([points], dtype=np.int32)
            image = cv2.polylines(image, points, False, (0, 0, 255), 3)

        return image
