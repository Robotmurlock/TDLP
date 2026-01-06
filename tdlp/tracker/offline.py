import logging
from typing import List, Optional

import numpy as np

from motrack.library.cv.bbox import BBox, PredBBox
from motrack.tracker.tracklet import Tracklet
from tdlp.datasets.dataset.transform import Transform
from tdlp.tracker.online import TDLPOnlineTracker
from tdlp.utils.extra_features import ExtraFeaturesReader
from torch import nn

logger = logging.getLogger('TDLPOfflineTracker')


class TDLPOfflineTracker(TDLPOnlineTracker):
    """
    Offline tracker for TDLP.

    Notes:
    - Requires pre-computed features.
    - Supports multi-modal input
    """

    def __init__(
        self,
        transform: Transform,
        model: nn.Module,
        extra_features_reader: ExtraFeaturesReader,
        device: str,
        detection_threshold: float = 0.4,
        sim_threshold: float = 0.5,
        initialization_threshold: int = 1,
        remember_threshold: int = 30,
        clip_length: Optional[int] = None,
        new_tracklet_detection_threshold: float = 0.9,
        use_conf: bool = True
    ):
        """
        Args:
            transform: Transform to apply to the data.
            model: Model to use for tracking.
            extra_features_reader: Extra features reader to use for tracking.
            device: Device to use for tracking.
            detection_threshold: Detection threshold to use for tracking.
            sim_threshold: Similarity threshold to use for tracking.
            initialization_threshold: Initialization threshold to use for tracking.
            remember_threshold: Remember threshold to use for tracking.
            clip_length: Clip length to use for tracking.
            new_tracklet_detection_threshold: New tracklet detection threshold to use for tracking.
            use_conf: Use confidence threshold to filter detections.

        Notes:
        - Requires pre-computed features.
        - Supports multi-modal input
        """
        super().__init__(
            transform=transform,
            model=model,
            device=device,
            detection_threshold=detection_threshold,
            sim_threshold=sim_threshold,
            initialization_threshold=initialization_threshold,
            remember_threshold=remember_threshold,
            clip_length=clip_length,
            new_tracklet_detection_threshold=new_tracklet_detection_threshold,
            use_conf=use_conf
        )

        self._extra_features_reader = extra_features_reader

    def track(self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        scene_name = self.get_scene()
        objects_data = self._extra_features_reader.read(scene_name, frame_index)
        objects_data = [data for data in objects_data if data['bbox_conf'] > self._detection_threshold]
        detections = [PredBBox.create(BBox.from_xywh(*data['bbox_xywh']), label='pedestrian', conf=data['bbox_conf']) for data in objects_data]
        return self._track(tracklets, detections, objects_data, frame_index, frame)
