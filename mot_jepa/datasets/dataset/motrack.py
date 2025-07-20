from typing import Optional, List, Tuple, Union

import numpy as np
from motrack.datasets import BaseDataset
from motrack.datasets.base import BasicSceneInfo, ObjectFrameData

from mot_jepa.datasets.dataset.index.mot import DatasetIndex


# TODO: Move to Motrack?
class MotrackDatasetWrapper(BaseDataset):
    def __init__(
        self,
        index: DatasetIndex,
        test: bool = False,
        sequence_list: Optional[List[str]] = None,
        image_shape: Union[None, List[int], Tuple[int, int]] = None,
        image_bgr_to_rgb: bool = True
    ):
        super().__init__(
            test=test,
            sequence_list=sequence_list,
            image_shape=image_shape,
            image_bgr_to_rgb=image_bgr_to_rgb
        )
        self._index = index

    def load_scene_image_by_frame_index(self, scene_name: str, frame_index: int) -> np.ndarray:
        return self.load_image(self._index.get_scene_image_path(scene_name, frame_index))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def scenes(self) -> List[str]:
        return self._index.scenes

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        return self._index.parse_object_id(object_id)

    def get_object_category(self, object_id: str) -> str:
        return self._index.get_object_category(object_id)

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        return self._index.get_scene_object_ids(scene_name)

    def get_object_data_length(self, object_id: str) -> int:
        return self._index.get_object_data_length(object_id)

    def get_object_data(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> Optional[ObjectFrameData]:
        return self._index.get_object_data_label_by_frame_index(object_id, index, relative_bbox_coords=relative_bbox_coords)

    def get_object_data_by_frame_index(self, object_id: str, frame_index: int, relative_bbox_coords: bool = True) -> Optional[ObjectFrameData]:
        return self._index.get_object_data_label_by_frame_index(object_id, frame_index, relative_bbox_coords=relative_bbox_coords)

    def get_scene_info(self, scene_name: str) -> BasicSceneInfo:
        return self._index.get_scene_info(scene_name)

    def get_scene_image_path(self, scene_name: str, frame_index: int) -> str:
        return self._index.get_scene_image_path(scene_name, frame_index)
