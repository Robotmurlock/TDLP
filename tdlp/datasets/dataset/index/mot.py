"""
MOT Challenge Dataset support.
"""
import configparser
import copy
import enum
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from tdlp.datasets.common.scene_info import BasicSceneInfo
from tdlp.datasets.dataset.index.index import FrameObjectData, DatasetIndex
from tdlp.utils import file_system

CATEGORY = 'pedestrian'
SEPARATOR = '+'
N_IMG_DIGITS = 6

class LabelType(enum.Enum):
    DETECTION = 'det'
    GROUND_TRUTH = 'gt'


@dataclass
class SceneInfo(BasicSceneInfo):
    """
    MOT Scene metadata (name, frame shape, ...)
    """
    dirpath: str
    gt_path: str
    framerate: Union[str, int]
    imdir: str
    imext: str

    def __post_init__(self):
        """
        Convert to proper type
        """
        super().__post_init__()
        self.framerate = int(self.framerate)


SceneInfoIndex = Dict[str, SceneInfo]


logger = logging.getLogger('MOTDataset')


class MOTDatasetIndex(DatasetIndex):
    """
    Parses MOT dataset in given format
    """
    def __init__(
        self,
        paths: Union[str, List[str]],
        split: str,
        sequence_list: Optional[List[str]] = None,
        label_type: LabelType = LabelType.GROUND_TRUTH,
        skip_corrupted: bool = False
    ) -> None:
        """
        Args:
            paths: One more dataset paths
            split: Dataset split
            sequence_list: Sequence filter by defined list
            label_type: Label Type
            skip_corrupted: Skip corrupted scenes (otherwise error is raised)
            test: Is this dataset used for test evaluation
        """
        super().__init__(
            split=split,
            sequence_list=sequence_list
        )
        test = split == 'test'

        if isinstance(paths, str):
            paths = [paths]
        paths = [os.path.join(path, self._split) for path in paths]  # Add split suffix

        self._label_type = label_type
        self._scene_info_index, self._n_digits = self._index_dataset(paths, label_type, sequence_list, skip_corrupted, test=test)
        self._data_labels, self._present_object_ids, self._n_labels = \
            self._parse_labels(self._scene_info_index, test=test)

    @property
    def label_type(self) -> LabelType:
        """
        Returns: LabelType
        """
        return self._label_type

    @property
    def scenes(self) -> List[str]:
        return list(self._scene_info_index.keys())

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        assert object_id in self._data_labels, f'Unknown object id "{object_id}".'
        scene_name, scene_object_id = object_id.split(SEPARATOR)
        return scene_name, scene_object_id

    def get_object_category(self, object_id: str) -> str:
        return CATEGORY

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        assert scene_name in self.scenes, f'Unknown scene "{scene_name}". Dataset scenes: {self.scenes}.'
        return [d for d in self._data_labels if d.startswith(scene_name)]

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        return len(self.get_scene_object_ids(scene_name))

    def get_object_data_length(self, object_id: str) -> int:
        return len(self._data_labels[object_id])

    def get_object_data_label_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[dict]:
        data = copy.deepcopy(self._data_labels[object_id][frame_index])
        if data is None:
            return None

        if not relative_bbox_coords:
            scene_name, _ = self.parse_object_id(object_id)
            scene_info = self._scene_info_index[scene_name]
            bbox = data.bbox
            bbox = [
                round(bbox[0] * scene_info.imwidth),
                round(bbox[1] * scene_info.imheight),
                round(bbox[2] * scene_info.imwidth),
                round(bbox[3] * scene_info.imheight)
            ]
            data.bbox = bbox

        return data

    def get_scene_info(self, scene_name: str) -> BasicSceneInfo:
        return self._scene_info_index[scene_name]

    def _get_image_path(self, scene_info: SceneInfo, frame_id: int) -> str:
        """
        Get frame path for given scene and frame id

        Args:
            scene_info: scene metadata
            frame_id: frame number

        Returns:
            Path to image (frame)
        """
        return os.path.join(scene_info.dirpath, scene_info.imdir, f'{frame_id:0{self._n_digits}d}{scene_info.imext}')

    def get_scene_image_path(self, scene_name: str, frame_index: int) -> str:
        scene_info = self._scene_info_index[scene_name]
        return self._get_image_path(scene_info, frame_index + 1)

    @staticmethod
    def _get_data_cache_path(path: str, data_name: str) -> str:
        """
        Get cache path for data object path.

        Args:
            path: Path

        Returns:
            Path where data object is or will be stored.
        """
        filename = Path(path).stem
        cache_filename = f'.{filename}.{data_name}.json'
        dirpath = str(Path(path).parent)
        return os.path.join(dirpath, cache_filename)

    @staticmethod
    def parse_scene_ini_file(scene_directory: str, label_type_name) -> SceneInfo:
        gt_path = os.path.join(scene_directory, label_type_name, f'{label_type_name}.txt')
        scene_info_path = os.path.join(scene_directory, 'seqinfo.ini')
        raw_info = configparser.ConfigParser()
        raw_info.read(scene_info_path)
        raw_info = dict(raw_info['Sequence'])
        raw_info['gt_path'] = gt_path
        raw_info['dirpath'] = scene_directory

        return SceneInfo(**raw_info, category=CATEGORY)

    @staticmethod
    def _index_dataset(
        paths: List[str],
        label_type: LabelType,
        sequence_list: Optional[List[str]],
        skip_corrupted: bool,
        test: bool = False
    ) -> Tuple[SceneInfoIndex, int]:
        """
        Index dataset content. Format: { {scene_name}: {scene_labels_path} }

        Args:
            paths: Dataset paths
            label_type: Use ground truth bboxes or detections
            sequence_list: Filter scenes
            skip_corrupted: Skips incomplete scenes
            test: Is it test split

        Returns:
            Index to scenes, number of digits used in images name convention (may vary between datasets)
        """
        n_digits = N_IMG_DIGITS
        scene_info_index: SceneInfoIndex = {}

        for path in paths:
            scene_names = [file for file in file_system.listdir(path) if not file.startswith('.')]
            logger.debug(f'Found {len(scene_names)} scenes for dataset "{path}". Names: {scene_names}.')

            n_skipped: int = 0

            for scene_name in scene_names:
                if sequence_list is not None and scene_name not in sequence_list:
                    continue

                scene_directory = os.path.join(path, scene_name)
                scene_files = file_system.listdir(scene_directory)

                # Scene content validation
                skip_scene = False
                for filename in [label_type.value, 'seqinfo.ini']:
                    if filename not in scene_files:
                        if filename == label_type.value and test:
                            continue

                        msg = f'Ground truth file "{filename}" not found on path "{scene_directory}". Contents: {scene_files}'
                        if not skip_corrupted:
                            raise FileNotFoundError(msg)

                        logger.warning(f'Skipping scene "{scene_name}" for dataset "{path}". Reason: `{msg}`')
                        skip_scene = True
                        break

                if 'img1' in scene_files:
                    # Check number of digits used in image name (e.g. DanceTrack and MOT20 have different convention)
                    img1_path = os.path.join(scene_directory, 'img1')
                    image_names = file_system.listdir(img1_path)
                    assert len(image_names) > 0, f'Image folder exists but it is empty! Dataset: "{path}"'
                    image_name = Path(image_names[0]).stem
                    n_digits = len(image_name)

                if skip_scene:
                    n_skipped += 1
                    continue

                scene_info = MOTDatasetIndex.parse_scene_ini_file(scene_directory, label_type.value)
                scene_info_index[scene_name] = scene_info
                logger.debug(f'Scene info {scene_info}.')

            if n_digits != N_IMG_DIGITS:
                logger.warning(f'Dataset "{path}" does not have default number of digits in image name. '
                               f'Got {n_digits} where default is {N_IMG_DIGITS}.')

            logger.info(f'Total number of parsed scenes is {len(scene_info_index)}. '
                        f'Number of skipped scenes is {n_skipped} for path "{path}".')

        return scene_info_index, n_digits

    @staticmethod
    def _parse_scene_labels(
        scene_info: SceneInfo,
        data: Dict[str, List[Optional[FrameObjectData]]],
        present_object_ids: Dict[str, Dict[int, List[str]]]
    ) -> Tuple[int, Dict[str, List[Optional[FrameObjectData]]], Dict[str, List[List[str]]]]:
        df = pd.read_csv(scene_info.gt_path, header=None)
        df = df[df[7] == 1]  # Ignoring values that are not evaluated

        df = df.iloc[:, :6]
        df.columns = ['frame_id', 'object_id', 'xmin', 'ymin', 'w', 'h']  # format: xywh
        df['object_global_id'] = \
            scene_info.name + SEPARATOR + df['object_id'].astype(str)  # object id is not unique over all scenes
        df = df.drop(columns='object_id', axis=1)
        df = df.sort_values(by=['object_global_id', 'frame_id'])
        n_labels = df.shape[0]

        object_groups = df.groupby('object_global_id')
        present_object_ids[scene_info.name] = [[] for _ in range(scene_info.seqlength)]
        for object_global_id, df_grp in tqdm(object_groups, desc=f'Parsing {scene_info.name}', unit='pedestrian'):
            df_grp = df_grp.drop(columns='object_global_id', axis=1).set_index('frame_id')

            data[object_global_id] = [None for _ in range(scene_info.seqlength)]
            for frame_id, row in df_grp.iterrows():
                # frame index: [1, N] -> [0, N-1]
                frame_index = int(frame_id) - 1

                # bbox: Absolute -> relative
                bbox = row.values.tolist()
                bbox = [
                    bbox[0] / scene_info.imwidth,
                    bbox[1] / scene_info.imheight,
                    bbox[2] / scene_info.imwidth,
                    bbox[3] / scene_info.imheight
                ]

                data[object_global_id][frame_index] = FrameObjectData(
                    scene=scene_info.name,
                    object_id=object_global_id,
                    frame_index=frame_index,
                    bbox=bbox,
                    score=1.0,
                    category=CATEGORY
                )
                present_object_ids[scene_info.name][frame_index].append(object_global_id)

        return n_labels, data, present_object_ids

    def _parse_labels(self, scene_infos: SceneInfoIndex, test: bool = False) \
            -> Tuple[Dict[str, List[Optional[FrameObjectData]]], Dict[str, List[List[str]]], int]:
        """
        Loads all labels dictionary with format:
        {
            {scene_name}_{object_id}: {
                {frame_id}: [ymin, xmin, w, h]
            }
        }

        Args:
            scene_infos: Scene Metadata
            test: If test then no parsing is performed

        Returns:
            Labels dictionary, labels count
        """
        data: Dict[str, List[Optional[dict]]] = {}
        present_object_ids: Dict[str, List[List[str]]] = {}
        n_labels = 0
        if test:
            # Return empty labels
            return data, present_object_ids, n_labels

        for scene_name, scene_info in tqdm(scene_infos.items(), unit='scene', total=len(scene_infos), desc='Indexing'):
            scene_info = self._scene_info_index[scene_name]
            n_scene_labels, data, present_object_ids = self._parse_scene_labels(
                scene_info=scene_info,
                data=data,
                present_object_ids=present_object_ids
            )
            n_labels += n_scene_labels

        logger.debug(f'Parsed labels. Dataset size is {n_labels}.')
        return data, present_object_ids, n_labels

    def get_objects_present_in_scene_at_frame(self, scene_name: str, frame_index: int) -> List[str]:
        return self._present_object_ids[scene_name][frame_index]
