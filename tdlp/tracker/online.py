import copy
import math
from typing import List, Optional, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker import Tracker
from motrack.tracker.matching.utils import hungarian
from motrack.tracker.tracklet import Tracklet, TrackletState
from tdlp.architectures.tdlp.core import MultiModalTDCP, MultiModalTDSP
from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import (
    PredictionBBoxFeatureExtractor,
    SupportedFeatures,
)
from tdlp.datasets.dataset.transform import Transform
import torch
from torch import nn
from torch.nn import functional as F


class TDLPOnlineTracker(Tracker):
    """
    Online tracker for TDLP.

    Notes:
    - Supports only bbox features.
    """

    def __init__(
        self,
        transform: Transform,
        model: nn.Module,
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
            device: Device to use for tracking.
            detection_threshold: Detection threshold to use for tracking.
            sim_threshold: Similarity threshold to use for tracking.
            initialization_threshold: Initialization threshold to use for tracking.
            remember_threshold: Remember threshold to use for tracking.
            clip_length: Clip length to use for tracking.
            new_tracklet_detection_threshold: New tracklet detection threshold to use for tracking.
            use_conf: Use confidence threshold to filter detections.

        Notes:
        - Supports only bbox features.
        """
        super().__init__()

        self._device = device

        self._transform = transform

        self._model: MultiModalTDCP | MultiModalTDSP = model
        self._model.to(device)
        self._model.eval()

        self._feature_names = set([SupportedFeatures(feature_name) for feature_name in self._model.feature_names])

        self._detection_threshold = detection_threshold
        self._sim_threshold = sim_threshold

        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._clip_length = clip_length if clip_length is not None else self._remember_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold

        self._next_id = 0
        self._use_conf = use_conf

    def _convert_data(
        self,
        tracklets: List[Tracklet],
        objects_data: List[dict],
        frame_index: int,
        frame: Optional[np.ndarray] = None,
    ) -> VideoClipData:
        n_tracks = len(tracklets)
        n_detections = len(objects_data)

        # Determine maximum size
        N = max(n_tracks, n_detections)

        # Observed initialization
        observed_ts = torch.zeros(N, self._clip_length, dtype=torch.long)
        observed_temporal_mask = torch.ones(N, self._clip_length, dtype=torch.bool)
        observed_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=self._feature_names,
            n_tracks=N,
            temporal_length=self._clip_length,
        )

        time_offset = frame_index - self._clip_length
        for t_i, tracklet in enumerate(tracklets):
            for frame_info in tracklet.history:
                hist_frame_index = frame_info.frame_index
                data = frame_info.data
                relative_index = hist_frame_index - time_offset
                if relative_index < 0:
                    continue

                PredictionBBoxFeatureExtractor.set_features(
                    feature_names=self._feature_names,
                    features=observed_features,
                    object_index=t_i,
                    clip_index=relative_index,
                    data=data
                )
                observed_ts[t_i, relative_index] = hist_frame_index
                observed_temporal_mask[t_i, relative_index] = False

        # Unobserved initialization
        unobserved_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=self._feature_names,
            n_tracks=N,
            temporal_length=1,
        )
        unobserved_ts = torch.zeros(N, dtype=torch.long)
        unobserved_temporal_mask = torch.ones(N, dtype=torch.bool)

        unobserved_ts[:n_detections] = frame_index
        unobserved_temporal_mask[:n_detections] = False

        for d_i, data in enumerate(objects_data):
            PredictionBBoxFeatureExtractor.set_features(
                feature_names=self._feature_names,
                features=unobserved_features,
                object_index=d_i,
                clip_index=0,
                data=data
            )

        # Remove temporal dimension
        unobserved_features = {k: v[:, 0] for k, v in unobserved_features.items()}

        return VideoClipData(
            observed=VideoClipPart(
                ids=None,
                ts=observed_ts,
                mask=observed_temporal_mask,
                features=observed_features
            ),
            unobserved=VideoClipPart(
                ids=None,
                ts=unobserved_ts,
                mask=unobserved_temporal_mask,
                features=unobserved_features
            )
        )

    def _association(
        self,
        tracklets: List[Tracklet],
        objects_data: List[dict],
        frame_index: int,
        sim_threshold: float,
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracks = len(tracklets)
        n_detections = len(objects_data)
        if n_tracks == 0:
            return [], [], list(range(n_detections))
        elif n_detections == 0:
            return [], list(range(n_tracks)), []

        data = self._convert_data(tracklets, objects_data, frame_index, frame=frame)
        data = self._transform(data)
        data.apply(lambda x: x.unsqueeze(0).to(self._device))

        if isinstance(self._model, MultiModalTDCP):
            track_mm_features, det_mm_features, _, _ = self._model(
                data.observed.features,
                data.observed.mask,
                data.unobserved.features,
                data.unobserved.mask
            )
            track_mm_features = track_mm_features[0][:n_tracks]
            det_mm_features = det_mm_features[0][:n_detections]
            track_mm_features = F.normalize(track_mm_features, dim=-1).cpu()
            det_mm_features = F.normalize(det_mm_features, dim=-1).cpu()

            cost_matrix = (track_mm_features @ det_mm_features.T).numpy()
            cost_matrix = 1 - (cost_matrix + 1) / 2 # [-1, 1] -> [0, 1]
        elif isinstance(self._model, MultiModalTDSP):
            BATCH_SIZE = 1000
            n_batches = math.ceil(data.unobserved.mask.shape[1] / BATCH_SIZE)
            batch_logit_list = []
            for batch_index in range(n_batches):
                batch_start = batch_index * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, data.unobserved.mask.shape[1])
                batch_features = {k: v[:, batch_start:batch_end] for k, v in data.unobserved.features.items()}
                batch_mask = data.unobserved.mask[:, batch_start:batch_end]
                try:
                    batch_logits, _ = self._model(
                        data.observed.features, 
                        data.observed.mask, 
                        batch_features, 
                        batch_mask
                    )
                except:
                    print(f'{n_tracks=}, {n_detections=}, {batch_start=}, {batch_end=}')
                    observed_feature_shapes = {k: v.shape for k, v in data.observed.features.items()}
                    batch_feature_shapes = {k: v.shape for k, v in batch_features.items()}
                    print(f'{data.observed.mask.shape=}, {batch_mask.shape=}, {observed_feature_shapes=}, {batch_feature_shapes=}')
                    raise
                batch_logit_list.append(batch_logits.cpu())
            logits = torch.cat(batch_logit_list, dim=1)
            probas = torch.sigmoid(logits).numpy()
            cost_matrix = 1 - probas[0, :n_tracks, :n_detections]

        else:
            raise ValueError(f'Unsupported model type: {type(self._model)}')

        cost_matrix[cost_matrix > sim_threshold] = np.inf

        return hungarian(cost_matrix)

    def track(self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        detections = [detection for detection in detections if detection.conf > self._detection_threshold]
        objects_data = [
            {
                'bbox_xywh': detection.as_numpy_xywh().tolist(),
                'bbox_conf': detection.conf,
            } for detection in detections
        ]
        return self._track(tracklets, detections, objects_data, frame_index, frame)

    def _track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        objects_data: List[dict],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        # Remove deleted
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]
        matches, unmatched_tracklets, unmatched_detections = self._association(tracklets, objects_data, frame_index, sim_threshold=self._sim_threshold, frame=frame)

        # Handle matches
        for t_i, d_i in matches:
            tracklet = tracklets[t_i]
            detection = detections[d_i]
            data = objects_data[d_i]

            new_state = TrackletState.ACTIVE
            if tracklet.state == TrackletState.NEW and tracklet.total_matches + 1 < self._initialization_threshold:
                new_state = TrackletState.NEW
            tracklet.update(detection, frame_index, state=new_state, frame_data=data)

        # Handle unmatched tracklets
        for t_i in unmatched_tracklets:
            tracklet = tracklets[t_i]

            lost_time = (frame_index - tracklet.frame_index) if tracklet.frame_index is not None else 0
            if lost_time > self._remember_threshold or tracklet.age < self._initialization_threshold:  # 3 -> 1
                tracklet.state = TrackletState.DELETED
            else:
                tracklet.state = TrackletState.LOST

        # Handle unmatched detections
        new_tracklets: List[Tracklet] = []
        for d_i in unmatched_detections:
            detection = detections[d_i]
            data = objects_data[d_i]

            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=copy.deepcopy(detection),
                frame_index=frame_index,
                _id=self._next_id,
                state=TrackletState.NEW if frame_index > self._initialization_threshold else TrackletState.ACTIVE,
                max_history=self._clip_length - 1,
                frame_data=data
            )
            self._next_id += 1
            new_tracklets.append(new_tracklet)

        tracklets.extend(new_tracklets)

        return tracklets
