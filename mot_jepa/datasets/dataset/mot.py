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
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mot_jepa.datasets.dataset.augmentations import Augmentation
from mot_jepa.datasets.dataset.common.data import VideoClipData
from mot_jepa.datasets.dataset.index.index import DatasetIndex
from mot_jepa.datasets.dataset.transform import Transform

logger = logging.getLogger('MOTClipDataset')


class MOTClipDataset(Dataset):
    """
    MOT video clip-based dataset. It samples random video clips based on the dataset frame index.
    """
    BBOX_DIM = 5
    CROP_CHANNELS = 3

    def __init__(
        self,
        index: DatasetIndex,
        n_tracks: int,
        clip_length: int,
        transform: Transform,
        augmentations: Augmentation,
        min_clip_tracks: int = 1,
        clip_sampling_step: int = 1
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

        # Track parameters
        max_tracks = index.get_max_tracks()
        logger.info(f'Maximum number of tracks over all scenes: {max_tracks}.')

        if n_tracks <= max_tracks:
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

        # Transforms
        self._transform = transform
        self._augmentations = augmentations

        logger.info(f'Number of sampled clips: {len(self._clip_index)}.')

    @property
    def index(self) -> DatasetIndex:
        """
        Returns: Dataset index
        """
        return self._index

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

    def __len__(self) -> int:
        return len(self._clip_index)

    def _bbox_augmentation(self, bbox: torch.Tensor) -> torch.Tensor:
        # TODO: Generalize and make configurable
        # noinspection PyPep8Naming
        SIGMA = 0.05
        bbox_noise = SIGMA * torch.randn_like(bbox)
        w, h = bbox[..., 2], bbox[..., 3]
        bbox_noise[..., [0, 2]] *= w
        bbox_noise[..., [1, 3]] *= h
        bbox = bbox + bbox_noise
        bbox = torch.clamp(bbox, min=0.0, max=1.0)
        return bbox

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

    def _extract_scene_clip_data(
        self,
        scene_name: str,
        start_index: int,
        end_index: int,
        start_time: int,
        temporal_length: int,
        remove_temporal_dim: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts scene per-frame video clip data. It extracts bboxes for each object for each frame.

        Args:
            scene_name: Scene name
            start_index: Clip start index
            end_index: Clip end index
            start_time: Start time
            temporal_length: Temporal (clip) length
            remove_temporal_dim: Remove temporal dimension (only possible if clip length is equal to 1)

        Returns: torch tensors
            - BBoxes
            - Relative timestamps
            - Temporal mask (1 if an object is missing else 0)
        """
        object_ids = self._index.get_objects_present_in_scene_clip(scene_name, start_index, end_index)
        if len(object_ids) > self._n_tracks:
            logger.debug(f'Too many tracks: Maximum is {self._n_tracks} but got {len(object_ids)}. '
                         f'Removing at random...')
            random.shuffle(object_ids)
            object_ids = object_ids[:self._n_tracks]

        bboxes = torch.zeros(self._n_tracks, temporal_length, self.BBOX_DIM, dtype=torch.float32)
        times = torch.arange(start_time, start_time + temporal_length, dtype=torch.long) \
            .unsqueeze(0).repeat(self._n_tracks, 1)
        temporal_mask = torch.ones(self._n_tracks, temporal_length, dtype=torch.bool)

        for clip_index, frame_index in enumerate(range(start_index, end_index)):
            for object_index, object_id in enumerate(object_ids):
                data = self._index.get_object_data_label_by_frame_index(object_id, frame_index)
                if data is None:
                    continue

                # Mask
                temporal_mask[object_index, clip_index] = False

                # BBox
                bboxes[object_index, clip_index, :] = self.bbox_to_tensor(data.bbox, data.score)

        if remove_temporal_dim:
            assert temporal_length == 1, 'Can\'t remove temporal dim unless it has length of 1!'
            bboxes = bboxes[:, 0, :]
            times = times[:, 0]
            temporal_mask = temporal_mask[:, 0]

        return bboxes, times, temporal_mask

    def get_raw(self, index: int) -> VideoClipData:
        """
        Get raw clip data (without any transformations)

        Args:
            index: Dataset index

        Returns:
            Clip observed and unobserved bboxes, timestamps and temporal masks (6 elements)
        """
        scene_name, start_index, end_index = self._clip_index[index]

        observed_end_index = start_index + self._clip_length
        observed_bboxes, observed_ts, observed_temporal_mask = self._extract_scene_clip_data(
            scene_name=scene_name,
            start_index=start_index,
            end_index=observed_end_index,
            start_time=0,
            temporal_length=self._clip_length
        )

        unobserved_bboxes, unobserved_ts, unobserved_temporal_mask = self._extract_scene_clip_data(
            scene_name=scene_name,
            start_index=observed_end_index,
            end_index=observed_end_index + 1,
            start_time=self._clip_length,
            temporal_length=1,
            remove_temporal_dim=True
        )

        return VideoClipData(
            observed_bboxes=observed_bboxes,
            observed_ts=observed_ts,
            observed_temporal_mask=observed_temporal_mask,
            unobserved_bboxes=unobserved_bboxes,
            unobserved_ts=unobserved_ts,
            unobserved_temporal_mask=unobserved_temporal_mask
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.get_raw(index)
        data = self._transform(data)
        data = self._augmentations(data)
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
        unobserved_bboxes = raw.unobserved_bboxes
        unobserved_temporal_mask = raw.unobserved_temporal_mask
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
        observed_bboxes = raw.observed_bboxes
        observed_temporal_mask = raw.observed_temporal_mask
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
