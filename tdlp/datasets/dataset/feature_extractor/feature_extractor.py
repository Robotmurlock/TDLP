"""Abstract base class for feature extractors."""
from abc import ABC, abstractmethod
import logging
from typing import Dict

import torch

from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.index.index import DatasetIndex

logger = logging.getLogger('FeatureExtractor')


class FeatureExtractor(ABC):
    """Base interface for extracting clip features."""
    def __init__(
        self,
        index: DatasetIndex,
        object_id_mapping: Dict[str, int],
        n_tracks: int
    ):
        """
        Args:
            index: Dataset index.
            object_id_mapping: Object ID mapping.
            n_tracks: Number of tracks.
        """
        self._index = index
        self._object_id_mapping = object_id_mapping
        self._n_tracks = n_tracks

    def extract(
        self,
        scene_name: str,
        observed_start_index: int,
        observed_start_time: int,
        observed_temporal_length: int,
        unobserved_start_index: int,
        unobserved_start_time: int,
        unobserved_temporal_length: int,
    ) -> VideoClipData:
        """
        Extract the features for a given scene. This is performed in two steps: 
        - Extract the common part of the clip, which includes the IDs, timestamps, and masks.
        - Extract the extra data part of the clip, which includes the features.

        Args:
            scene_name: Name of the scene.
            observed_start_index: Start index of the observed part.
            observed_start_time: Start time of the observed part.
            observed_temporal_length: Temporal length of the observed part.
            unobserved_start_index: Start index of the unobserved part.
            unobserved_start_time: Start time of the unobserved part.
            unobserved_temporal_length: Temporal length of the unobserved part.
        """
        return VideoClipData(
            observed=self._extract_part(
                scene_name=scene_name,
                start_index=observed_start_index,
                start_time=observed_start_time,
                temporal_length=observed_temporal_length,
                observed=True
            ),
            unobserved=self._extract_part(
                scene_name=scene_name,
                start_index=unobserved_start_index,
                start_time=unobserved_start_time,
                temporal_length=unobserved_temporal_length,
                observed=False
            )
        )

    def __call__(
        self,
        scene_name: str,
        observed_start_index: int,
        observed_start_time: int,
        observed_temporal_length: int,
        unobserved_start_index: int,
        unobserved_start_time: int,
        unobserved_temporal_length: int,
    ) -> VideoClipData:
        """
        Extract the features for a given scene. This is performed in two steps: 
        - Extract the common part of the clip, which includes the IDs, timestamps, and masks.
        - Extract the extra data part of the clip, which includes the features.

        Args:
            scene_name: Name of the scene.
            observed_start_index: Start index of the observed part.
            observed_start_time: Start time of the observed part.
            observed_temporal_length: Temporal length of the observed part.
            unobserved_start_index: Start index of the unobserved part.
            unobserved_start_time: Start time of the unobserved part.
            unobserved_temporal_length: Temporal length of the unobserved part.
        """
        return self.extract(
            scene_name=scene_name,
            observed_start_index=observed_start_index,
            observed_start_time=observed_start_time,
            observed_temporal_length=observed_temporal_length,
            unobserved_start_index=unobserved_start_index,
            unobserved_start_time=unobserved_start_time,
            unobserved_temporal_length=unobserved_temporal_length
        )

    def _extract_part(
        self,
        scene_name: str,
        start_index: int,
        start_time: int,
        temporal_length: int,
        observed: bool
    ) -> VideoClipPart:
        """
        Extract the common part of the clip, which includes the IDs, timestamps, and masks.

        Args:
            scene_name: Name of the scene.
            start_index: Start index of the clip.
            start_time: Start time of the clip.
            temporal_length: Temporal length of the clip.
            observed: Whether the clip is observed.

        Returns:
            Video clip part.
        """
        video_clip_part = self._extract_common_part(
            scene_name=scene_name,
            start_index=start_index,
            start_time=start_time,
            temporal_length=temporal_length
        )

        video_clip_part = self._extract_extra_data(
            video_clip_part=video_clip_part,
            scene_name=scene_name,
            start_index=start_index,
            temporal_length=temporal_length,
            observed=observed
        )

        if not observed:
            video_clip_part.remove_temporal_dimension()

        return video_clip_part


    def _extract_common_part(
        self,
        scene_name: str,
        start_index: int,
        start_time: int,
        temporal_length: int,
    ) -> VideoClipPart:
        """
        Extract the common part of the clip, which includes the IDs, timestamps, and masks.
        Note: `start_time` is not used in this implementation (back-compatibility with old code).

        Args:
            scene_name: Name of the scene.
            start_index: Start index of the clip.
            start_time: Start time of the clip.
            temporal_length: Temporal length of the clip.

        Returns:
            Video clip part.
        """
        _ = start_time  # Unused
        end_index = start_index + temporal_length
        object_ids = sorted(self._index.get_objects_present_in_scene_clip(scene_name, start_index, end_index))

        n_object_ids = len(object_ids)
        if n_object_ids > self._n_tracks:
            logger.warning(f'Too many tracks for scene {scene_name}: Maximum is {self._n_tracks} but got {n_object_ids}. '
                           f'Removing at random...')
            object_ids = object_ids[:self._n_tracks]

        ts = torch.zeros(self._n_tracks, temporal_length, dtype=torch.long)
        ids = torch.full_like(ts, fill_value=-1)
        mask = torch.ones(self._n_tracks, temporal_length, dtype=torch.bool)

        for clip_index, frame_index in enumerate(range(start_index, end_index)):
            for object_index, object_id in enumerate(object_ids):
                ts[object_index, clip_index] = frame_index
                ids[object_index, clip_index] = self._object_id_mapping[object_id]
                mask[object_index, clip_index] = False

        return VideoClipPart(
            ids=ids,
            ts=ts,
            mask=mask,
            features={}
        )

    @abstractmethod
    def _extract_extra_data(
        self,
        video_clip_part: VideoClipPart,
        scene_name: str,
        start_index: int,
        temporal_length: int,
        observed: bool
    ) -> VideoClipPart:
        """
        Extract the extra data part of the clip, which includes the features.

        Args:
            video_clip_part: Video clip part.
            scene_name: Name of the scene.
            start_index: Start index of the clip.
            temporal_length: Temporal length of the clip.
            observed: Whether the clip is observed.
        """
        pass
