"""
Dataset structure:
- Dataset consists of scenes i.e. videos
- Each scene has multiple tracks
- Each track has multiple times, bounding boxes

Dataset sample should be a clip from some scene with tensors:
- BBox tensor (N, T, Be), N is number of tracks, T is number of frames, Be is bbox feature dimension
    - Be should be (x, y, w, h, s), i.e. bbox coordinates and score
- Time tensor (N, T) with clip times
- Mask tensor (N, T) of bools (missing tracks and missing track's times)
"""
import logging
import math
import random
from typing import List, Tuple, Dict, Any
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset

from mot_jepa.datasets.dataset.index.index import DatasetIndex

logger = logging.getLogger('MOTClipDataset')


class MOTClipDataset(Dataset):
    BBOX_DIM = 5
    CROP_CHANNELS = 3

    def __init__(
        self,
        index: DatasetIndex,
        n_tracks: int,
        clip_length: int,
        min_clip_tracks: int = 1,
        clip_sampling_step: int = 1,
        use_augmentation: bool = False
    ):
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

        # Augmentation configuraion
        self._use_augmentation = False # use_augmentation

        logger.info(f'Number of sampled clips: {len(self._clip_index)}.')

    @property
    def index(self) -> DatasetIndex:
        return self._index

    @staticmethod
    def _create_clip_index(
        index: DatasetIndex,
        clip_length: int,
        min_clip_tracks: int = 1,
        clip_sampling_step: int = 1
    ) -> List[Tuple[str, int, int]]:
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
    def create_bbox(bbox: List[float], score: float) -> torch.Tensor:
        return torch.tensor([*bbox, score], dtype=torch.float32)

    def transform_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        if self._use_augmentation:
            bbox = self._bbox_augmentation(bbox)
        return bbox

    def _extract_scene_clip_data(
        self,
        scene_name: str,
        start_index: int,
        end_index: int,
        start_time: int,
        temporal_length: int,
        remove_temporal_dim: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                bbox = self.create_bbox(data.bbox, data.score)
                bboxes[object_index, clip_index, :] = self.transform_bbox(bbox)

        if remove_temporal_dim:
            assert temporal_length == 1, 'Can\'t remove temporal dim unless it has length of 1!'
            bboxes = bboxes[:, 0, :]
            times = times[:, 0]
            temporal_mask = temporal_mask[:, 0]

        return bboxes, times, temporal_mask

    def get_raw(self, index: int) -> Dict[str, Any]:
        scene_name, start_index, end_index = self._clip_index[index]

        observed_end_index = start_index + self._clip_length
        observed_bboxes, observed_times, observed_temporal_mask = self._extract_scene_clip_data(
            scene_name=scene_name,
            start_index=start_index,
            end_index=observed_end_index,
            start_time=0,
            temporal_length=self._clip_length
        )

        unobserved_bboxes, unobserved_times, unobserved_temporal_mask = self._extract_scene_clip_data(
            scene_name=scene_name,
            start_index=observed_end_index,
            end_index=observed_end_index + 1,
            start_time=self._clip_length,
            temporal_length=1,
            remove_temporal_dim=True
        )

        return {
            'observed_bboxes': observed_bboxes,
            'observed_ts': observed_times,
            'observed_temporal_mask': observed_temporal_mask,
            'unobserved_bboxes': unobserved_bboxes,
            'unobserved_ts': unobserved_times,
            'unobserved_temporal_mask': unobserved_temporal_mask
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.get_raw(index)
        observed_bboxes = raw['observed_bboxes']
        observed_times = raw['observed_ts']
        observed_temporal_mask = raw['observed_temporal_mask']
        unobserved_bboxes = raw['unobserved_bboxes']
        unobserved_times = raw['unobserved_ts']
        unobserved_temporal_mask = raw['unobserved_temporal_mask']

        # TODO: Refactor
        # Transform bboxes
        observed_bboxes[..., 2:4] = observed_bboxes[..., :2] + observed_bboxes[..., 2:4]
        # observed_bboxes[..., :2] = 3.5 * (observed_bboxes[..., :2] - 0.5)  # Centralize
        observed_bboxes = 3.5 * (observed_bboxes - 0.5)  # Centralize
        observed_bboxes_fod = torch.zeros_like(observed_bboxes)
        observed_bboxes_fod[:, 1:, :] = 30 * (observed_bboxes[:, 1:, :] - observed_bboxes[:, :-1, :])  # Scale
        observed_bboxes_fod[:, 1:, :] = observed_bboxes_fod[:, 1:, :] * (1 - observed_temporal_mask[:, :-1].unsqueeze(-1).repeat(1, 1, observed_bboxes.shape[-1]).float())
        observed_bboxes = torch.cat([observed_bboxes, observed_bboxes_fod], dim=-1)
        observed_bboxes[observed_temporal_mask] = 0

        unobserved_bboxes[..., 2:4] = unobserved_bboxes[..., :2] + unobserved_bboxes[..., 2:4]
        # unobserved_bboxes[..., :2] = 3.5 * (unobserved_bboxes[..., :2] - 0.5)
        unobserved_bboxes = 3.5 * (unobserved_bboxes - 0.5)  # Centralize
        unobserved_bboxes[unobserved_temporal_mask] = 0

        if self._use_augmentation:
            observed_bboxes, observed_times, observed_temporal_mask, \
                unobserved_bboxes, unobserved_times, unobserved_temporal_mask = \
                    self._augment_trajectories(
                        observed_bboxes=observed_bboxes,
                        observed_times=observed_times,
                        observed_temporal_mask=observed_temporal_mask,
                        unobserved_bboxes=unobserved_bboxes,
                        unobserved_times=unobserved_times,
                        unobserved_temporal_mask=unobserved_temporal_mask
                    )

        return {
            'observed_bboxes': observed_bboxes,
            'observed_ts': observed_times,
            'observed_temporal_mask': observed_temporal_mask,
            'unobserved_bboxes': unobserved_bboxes,
            'unobserved_ts': unobserved_times,
            'unobserved_temporal_mask': unobserved_temporal_mask
        }

    def visualize_scene(self, index: int) -> np.ndarray:
        scene_name, start_index, end_index = self._clip_index[index]
        scene_image_path = self._index.get_scene_image_path(scene_name, end_index)
        raw = self.get_raw(index)

        # noinspection PyUnresolvedReferences
        image = cv2.imread(scene_image_path)
        h, w, _ = image.shape
        assert image is not None, f'Failed to load image "{scene_image_path}".'

        # Visualize detections
        unobserved_bboxes = raw['unobserved_bboxes']
        unobserved_temporal_mask = raw['unobserved_temporal_mask']
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
        observed_bboxes = raw['observed_bboxes']
        observed_temporal_mask = raw['observed_temporal_mask']
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

    def _augment_trajectories(
        self,
        observed_bboxes: torch.Tensor,
        observed_times: torch.Tensor,
        observed_temporal_mask: torch.Tensor,
        unobserved_bboxes: torch.Tensor,
        unobserved_times: torch.Tensor,
        unobserved_temporal_mask: torch.Tensor
    ):
        traj_drop_ratio = 0.5  # self._augmentation_cfg.traj_drop_ratio

        # Record which token is removed during this process:
        observed_remove_masks = torch.zeros((self._n_tracks, self._clip_length), dtype=torch.bool)

        for n in range(self._n_tracks):
            if random.random() < traj_drop_ratio:
                observed_begin = random.randint(0, self._clip_length)
                observed_max_interval = self._clip_length - observed_begin
                observed_end = observed_begin + math.ceil(observed_max_interval * random.random())
                observed_remove_masks[n, observed_begin:observed_end] = True
        observed_temporal_mask = observed_temporal_mask | observed_remove_masks

        # Trajectory switch
        # TODO: Improve trajectory switch to be more meaningful
        traj_switch_ratio = 0.3  # self._augmentation_cfg.traj_switch_ratio
        if traj_switch_ratio > 0.0:
            for t in range(0, self._clip_length):
                switch_proba = torch.ones((self._n_tracks,), dtype=torch.float) * traj_switch_ratio
                switch_proba = (1 - observed_temporal_mask.all(dim=-1).float()) * switch_proba
                switch_map = torch.bernoulli(switch_proba)
                switch_indices = torch.nonzero(switch_map)  # objects to switch
                switch_indices = switch_indices.reshape((switch_indices.shape[0],))
                if len(switch_indices) == 1 and self._n_tracks > 1:
                    # Only one object can be switched, but we have more than one object.
                    # So we need to randomly select another object to switch.
                    switch_indices = torch.as_tensor(
                        [switch_indices[0].item(), random.randint(0, self._n_tracks - 1)], dtype=torch.long
                    )
                if len(switch_indices) > 1:
                    # Switch the trajectory features, boxes and masks:
                    shuffle_switch_indices = switch_indices[torch.randperm(len(switch_indices))]
                    observed_bboxes[switch_indices, t, :] = observed_bboxes[shuffle_switch_indices, t, :]
                    observed_temporal_mask[switch_indices, t] = observed_temporal_mask[shuffle_switch_indices, t]
                else:
                    continue  # no object to switch

        return observed_bboxes, observed_times, observed_temporal_mask,  \
                    unobserved_bboxes, unobserved_times, unobserved_temporal_mask
