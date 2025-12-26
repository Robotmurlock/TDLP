"""
Any Dataset that uses `DatasetIndex` interface can be used for training and evaluation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tdlp.datasets.common import BasicSceneInfo


@dataclass
class FrameObjectData:
    scene: str
    object_id: str
    frame_index: int
    bbox: List[float]
    score: float
    category: str
    features: Optional[dict] = None

    @property
    def frame_id(self) -> int:
        return self.frame_index + 1


class DatasetIndex(ABC):
    """
    Defines interface for dataset index.
    """
    def __init__(
        self,
        split: str,
        sequence_list: Optional[List[str]] = None
    ):
        self._split = split
        self._sequence_list = sequence_list

    @property
    def split(self) -> str:
        """
        Returns:
            Get dataset split.
        """
        return self._split

    @property
    @abstractmethod
    def scenes(self) -> List[str]:
        """
        Returns:
            List of scenes in dataset.
        """

    @abstractmethod
    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        """
        Parses and validates object id.

        Object id convention is `{scene_name}_{scene_object_id}` and is unique over all scenes.

        For MOT {scene_name} represents one video sequence.
        For SOT {scene_name} does not need to be unique for the sequence
        but `{scene_name}_{scene_object_id}` is always unique

        Args:
            object_id: Object id

        Returns:
            scene name, scene object id
        """

    @abstractmethod
    def get_object_category(self, object_id: str) -> str:
        """
        Gets category for object.

        Args:
            object_id: Object id

        Returns:
            Object category
        """

    @abstractmethod
    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        """
        Gets object ids for given scene name

        Args:
            scene_name: Scene name

        Returns:
            Scene objects
        """

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        """
        Gets number of unique objects in the scene.

        Args:
            scene_name: Scene name

        Returns:
            Number of objects in the scene
        """
        return len(self.get_scene_object_ids(scene_name))

    @abstractmethod
    def get_object_data_length(self, object_id: str) -> int:
        """
        Gets total number of data points for given `object_id` for .

        Args:
            object_id: Object id

        Returns:
            Number of data points
        """

    @abstractmethod
    def get_object_data_label_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[FrameObjectData]:
        """
        Like `get_object_data_label` but data is relative to given frame_index.
        If object does not exist in given frame index then None is returned.

        Args:
            object_id: Object id
            frame_index: Frame Index
            relative_bbox_coords: Scale bbox coords to [0, 1]

        Returns:
            Data point.
        """

    @abstractmethod
    def get_scene_info(self, scene_name: str) -> BasicSceneInfo:
        """
        Get scene metadata by name.

        Args:
            scene_name: Scene name

        Returns:
            Scene metadata
        """

    @abstractmethod
    def get_scene_image_path(self, scene_name: str, frame_index: int) -> str:
        """
        Get image (frame) path for given scene and frame id.

        Args:
            scene_name: scene name
            frame_index: frame index

        Returns:
            Frame path
        """

    @abstractmethod
    def get_objects_present_in_scene_at_frame(self, scene_name: str, frame_index: int) -> List[str]:
        """
        Get all object ids present in the scene at the given frame index.

        Args:
            scene_name: Scene name
            frame_index: Frame index

        Returns:
            List of present objects in the scene.
        """

    def get_objects_present_in_scene_clip(
        self,
        scene_name: str,
        start_index: int,
        end_index: int
    ) -> List[str]:
        """
        Get all object ids present in the scene at the given frame index.

        Args:
            scene_name: Scene name
            start_index: Clip start index
            end_index: Clip end index

        Returns:
            List of present objects in clip.
        """
        all_object_ids: List[str] = []
        for frame_index in range(start_index, end_index):
            object_ids = self.get_objects_present_in_scene_at_frame(scene_name, frame_index)
            all_object_ids.extend(object_ids)

        return sorted(list(set(all_object_ids)))

    def get_objects_present_in_scene(self, scene_name: str) -> List[str]:
        """
        Get all objects present in the scene across the full video.
        Special case for the `get_objects_present_in_scene_clip` function.

        Args:
            scene_name: Scene name

        Returns:
            List of present objects in scene/video.
        """
        scene_info = self.get_scene_info(scene_name)
        return self.get_objects_present_in_scene_clip(scene_name, 0, scene_info.seqlength)

    def get_max_tracks(self) -> int:
        """
        Returns:
            Maximum number of tracks over all scenes.
        """
        n = 0
        scene_with_max_tracks = None
        for scene in self.scenes:
            n_tracks = len(self.get_scene_object_ids(scene))
            if n_tracks > n:
                n = n_tracks
                scene_with_max_tracks = scene
        return scene_with_max_tracks, n
