import logging
import os.path
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import cv2
import hydra
from motrack.library.cv import BBox, PredBBox
from motrack.library.cv import color_palette
from motrack.library.cv.video_writer import MP4Writer
from motrack.tools.visualize import draw_tracklet
from tdlp.common.project import CONFIGS_PATH
from tdlp.config_parser import GlobalConfig
from tdlp.datasets.dataset import dataset_index_factory
from tdlp.datasets.dataset.index.mot import SceneInfo
from tdlp.utils import pipeline
from tdlp.utils.extra_features import ExtraFeaturesReader
import torch

logger = logging.getLogger('CameltrackFeaturesExtraction')




# COCO keypoint connections (pairs of keypoint indices)
COCO_PAIRS = [
    (5, 7), (7, 9),       # Left arm
    (6, 8), (8, 10),      # Right arm
    (11, 13), (13, 15),   # Left leg
    (12, 14), (14, 16),   # Right leg
    (5, 6), (11, 12),     # Shoulders & hips
    (5, 11), (6, 12),     # Torso connections
    (0, 1), (0, 2),       # Nose to eyes
    (1, 3), (2, 4)        # Eyes to ears
]


def draw_keypoints(
    image: np.ndarray,
    keypoints_xyc: np.ndarray,
    color: Tuple[int, int, int],
    imwidth: int,
    imheight: int,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Draw COCO 17 keypoints on the given image using normalized coordinates.

    Args:
        image: Input image as a numpy array (H, W, 3).
        keypoints_xyc: Keypoints array of shape (17, 3), with (x, y, confidence) normalized in [0,1].
        imwidth: Width of the image (for scaling normalized x).
        imheight: Height of the image (for scaling normalized y).
        color: BGR color tuple for drawing points and lines.
        threshold: Minimum confidence score to draw a keypoint.

    Returns:
        Image with keypoints drawn.
    """
    img = image.copy()
    keypoints_px = keypoints_xyc.copy()
    keypoints_px[:, 0] *= imwidth   # Scale x
    keypoints_px[:, 1] *= imheight  # Scale y

    # Draw skeleton lines
    for (i, j) in COCO_PAIRS:
        if keypoints_px[i, 2] > threshold and keypoints_px[j, 2] > threshold:
            pt1 = tuple(keypoints_px[i, :2].astype(int))
            pt2 = tuple(keypoints_px[j, :2].astype(int))
            cv2.line(img, pt1, pt2, color=color, thickness=2)

    # Draw keypoints
    for (x, y, c) in keypoints_px:
        if c > threshold:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)

    return img



def draw_frame_features(frame: np.ndarray, scene_info: SceneInfo, frame_features: List[dict]) -> np.ndarray:
    for features in frame_features:
        bbox = PredBBox.create(
            bbox=BBox.from_xywh(*features['bbox_xywh']),
            conf=features['bbox_conf'],
            label=0
        )
        object_id = int(features['object_id'].split('+')[-1]) if features['object_id'] is not None else -1

        frame = draw_tracklet(
            frame=frame,
            tracklet_id=object_id,
            tracklet_age=0,
            bbox=bbox
        )

        if "keypoints_xyc" in features:
            color = color_palette.ALL_COLORS_EXPECT_BLACK[int(object_id) % len(color_palette.ALL_COLORS_EXPECT_BLACK)]
            frame = draw_keypoints(
                image=frame,
                keypoints_xyc=np.array(features['keypoints_xyc']),
                color=color,
                imwidth=scene_info.imwidth,
                imheight=scene_info.imheight
            )

    return frame


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('cameltrack-features-extraction')
def main(cfg: GlobalConfig) -> None:
    # Hardcoded stuff
    SPLIT = 'train'
    EXTRACTED_OUTPUT_PATH = '/media/home/cameltrack-states/extracted-features-bee24'
    EXTRACTED_VIDEOS_PATH = '/media/home/cameltrack-states/videos-bee24'

    dataset_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split=SPLIT
    )

    scenes = dataset_index.scenes
    Path(EXTRACTED_VIDEOS_PATH).mkdir(parents=True, exist_ok=True)
    for scene_name in tqdm(scenes, desc='Visualizing extra features', unit='scene'):
        features_reader = ExtraFeaturesReader(EXTRACTED_OUTPUT_PATH)
        scene_info = dataset_index.get_scene_info(scene_name)
        with MP4Writer(os.path.join(EXTRACTED_VIDEOS_PATH, f'{scene_name}.mp4'), fps=scene_info.framerate) as mp4_writer:
            for frame_index in range(scene_info.seqlength):
                frame_features = features_reader.read(scene_name, frame_index)
                frame_path = dataset_index.get_scene_image_path(scene_name, frame_index)
                frame = cv2.imread(frame_path)
                assert frame is not None, f'Failed to load image "{frame_path}"!'
                frame = draw_frame_features(frame, scene_info, frame_features)

                mp4_writer.write(frame)


if __name__ == '__main__':
    main()
