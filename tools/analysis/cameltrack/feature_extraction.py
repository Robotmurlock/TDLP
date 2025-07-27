import logging
import os.path
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import hydra
import numpy as np
import pandas as pd
import torch
from motrack.library.cv import BBox
from motrack.tracker.matching.utils import hungarian
from tqdm import tqdm

from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.common import BasicSceneInfo
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.datasets.dataset.index.index import FrameObjectData
from mot_jepa.utils import pipeline
from mot_jepa.utils.extra_features import ExtraFeaturesWriter

logger = logging.getLogger('CameltrackFeaturesExtraction')


class CamelTrackParser:
    def __init__(self, states_path: str, temporary_dirpath: str):
        self._states_path = states_path
        self._temporary_path = temporary_dirpath

        # State
        self._scene_mapping: Dict[str, int] = {}
        self._scene_files: Dict[str, Dict[str, str]] = {}
        self._cache: Dict[str, Any] = {}

    @property
    def scene_mapping(self) -> Dict[str, int]:
        return self._scene_mapping

    @property
    def scene_files(self) -> Dict[str, Dict[str, str]]:
        return self._scene_files

    def open(self) -> None:
        with zipfile.ZipFile(self._states_path, 'r') as zip_ref:
            zip_ref.extractall(self._temporary_path)
        pickle_filenames = [filename for filename in os.listdir(self._temporary_path) if filename.endswith('.pkl')]
        pickle_filepaths = [os.path.join(self._temporary_path, filename) for filename in pickle_filenames]

        # Extract scene mapping
        self._scene_mapping.clear()
        for filename, filepath in zip(pickle_filenames, pickle_filepaths):
            if not filepath.endswith('_image.pkl'):
                continue
            video_id = int(filename.replace('_image.pkl', ''))

            with open(filepath, 'rb') as f:
                df_image = pickle.load(f)

            file_path = str(df_image.iloc[0].file_path)
            scene_name = Path(file_path).parent.parent.name
            self._scene_mapping[scene_name] = video_id
        self._scene_mapping = dict(sorted(self._scene_mapping.items()))

        # Extract scene files
        self._scene_files.clear()
        reverse_scene_mapping = {v: k for k, v in self._scene_mapping.items()}
        for filename, filepath in zip(pickle_filenames, pickle_filepaths):
            video_id = int(filename.replace('_image.pkl', '').replace('.pkl', ''))
            scene_name = reverse_scene_mapping[video_id]
            if scene_name not in self._scene_files:
                self._scene_files[scene_name] = {}

            if filename.endswith('_image.pkl'):
                self._scene_files[scene_name]['image'] = filepath
            elif filename.endswith('.pkl'):
                self._scene_files[scene_name]['features'] = filepath
            else:
                raise ValueError(f'Unexpected filename "{filename}"!')

    def close(self):
        if os.path.exists(self._temporary_path):
            shutil.rmtree(self._temporary_path)

    def __enter__(self) -> 'CamelTrackParser':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_scene_dfs(self, scene: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if scene not in self._cache:
            with open(self._scene_files[scene]['features'], 'rb') as f:
                df_features = pickle.load(f)
            with open(self._scene_files[scene]['image'], 'rb') as f:
                df_image = pickle.load(f)
            self._cache[scene] = (df_features, df_image)
        return self._cache[scene]

    def get(self, scene: str, frame_index: int) -> List[dict]:
        df_features, df_image = self.get_scene_dfs(scene)
        image_id = int(df_image[df_image.frame == frame_index].id.iloc[0])
        df_frame = df_features[df_features.image_id == image_id]

        result = []
        for _, row in df_frame.iterrows():
            result.append({
                'bbox_xywh': row.bbox_ltwh.tolist(),
                'bbox_conf': row.bbox_conf,
                'keypoints_xyc': row.keypoints_xyc.tolist(),
                'keypoints_conf': row.keypoints_conf,
                'appearance_embeddings': row.embeddings.tolist()
            })

        return result


def postprocess_data(scene_info: BasicSceneInfo, pred_frame_data: List[dict]) -> List[dict]:
    for object_features in pred_frame_data:
        object_features['bbox_xywh'] = [
            object_features['bbox_xywh'][0] / scene_info.imwidth,
            object_features['bbox_xywh'][1] / scene_info.imheight,
            object_features['bbox_xywh'][2] / scene_info.imwidth,
            object_features['bbox_xywh'][3] / scene_info.imheight
        ]

        object_features['keypoints_xyc'] = [
            [
                xyc[0] / scene_info.imwidth,
                xyc[1] / scene_info.imheight,
                xyc[2]
            ] for xyc in object_features['keypoints_xyc']
        ]

    return pred_frame_data


def add_track_ids(pred_frame_data: List[dict], gt_frame_data: List[FrameObjectData], threshold: float = 0.7) -> Tuple[List[dict], int, int]:
    pred_bboxes: List[BBox] = []
    for pred_object_data in pred_frame_data:
        pred_bboxes.append(BBox.from_xywh(*pred_object_data['bbox_xywh']))
    n_pred_bboxes = len(pred_bboxes)

    gt_bboxes: List[BBox] = [BBox.from_xywh(*data.bbox) for data in gt_frame_data]
    n_gt_bboxes = len(gt_bboxes)

    cost_matrix = np.zeros(shape=(n_pred_bboxes, n_gt_bboxes), dtype=np.float32)
    for pred_i in range(n_pred_bboxes):
        for gt_i in range(n_gt_bboxes):
            iou_score = pred_bboxes[pred_i].iou(gt_bboxes[gt_i])
            cost_matrix[pred_i, gt_i] = -iou_score if iou_score >= threshold else np.inf

    matches, unmatched_preds, _ = hungarian(cost_matrix)
    n_matches = len(matches)
    n_unmatches = len(unmatched_preds)

    for pred_i, gt_i in matches:
        pred_frame_data[pred_i]['object_id'] = gt_frame_data[gt_i].object_id

    for pred_i in unmatched_preds:
        pred_frame_data[pred_i]['object_id'] = None

    return pred_frame_data, n_matches, n_unmatches


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('cameltrack-features-extraction')
def main(cfg: GlobalConfig) -> None:
    # Hardcoded stuff
    SPLIT = 'train'
    CAMELTRACK_STATES_PATH = f'/media/home/cameltrack-states/dancetrack-{SPLIT}.pklz'
    TEMPORARY_DIRPATH = '/media/home/cameltrack-states/tmp'
    EXTRACTED_OUTPUT_PATH = '/media/home/cameltrack-states/extracted-features'

    dataset_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split=SPLIT
    )

    # Code
    n_total_matches, n_total_unmatches = 0, 0
    with CamelTrackParser(
        states_path=CAMELTRACK_STATES_PATH,
        temporary_dirpath=TEMPORARY_DIRPATH
    ) as parser:
        features_writer = ExtraFeaturesWriter(EXTRACTED_OUTPUT_PATH)
        scenes = dataset_index.scenes
        for scene_name in tqdm(scenes, desc='Parsing extra features', unit='scene'):
            scene_info = dataset_index.get_scene_info(scene_name)
            for frame_index in range(scene_info.seqlength):
                object_ids = dataset_index.get_objects_present_in_scene_at_frame(scene_name, frame_index)  # TODO: Check if +1 is needed
                gt_frame_data = [dataset_index.get_object_data_label_by_frame_index(object_id, frame_index) for object_id in object_ids]  # TODO: Check if +1 is needed

                pred_frame_data = parser.get(scene_name, frame_index)
                pred_frame_data = postprocess_data(scene_info, pred_frame_data)
                pred_frame_data, n_matches, n_unmatches = add_track_ids(pred_frame_data, gt_frame_data)
                n_total_matches += n_matches
                n_total_unmatches += n_unmatches

                features_writer.write(scene_name, frame_index, pred_frame_data)

    matches_ratio = n_total_matches / (n_total_matches + n_total_unmatches)
    unmatches_ratio = n_total_unmatches / (n_total_matches + n_total_unmatches)
    logger.info(f'Total matches: {n_total_matches} ({100 * matches_ratio:.1f}%). Total unmatches: {n_total_unmatches} ({100 * unmatches_ratio:.1f}%). ')


if __name__ == '__main__':
    main()
