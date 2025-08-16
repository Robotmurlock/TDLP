import copy
import json
from typing import Optional, List, Tuple

import hydra
import numpy as np
import torch
from motrack.evaluation.io import TrackerInferenceWriter
from motrack.library.cv.bbox import PredBBox, BBox
from motrack.object_detection import DetectionManager
from motrack.tools.postprocess import run_tracker_postprocess
from motrack.tools.visualize import run_visualize_tracker_inference
from motrack.tracker import Tracker
from motrack.tracker.matching.utils import hungarian
from motrack.tracker.tracklet import Tracklet, TrackletState
from motrack.utils.collections import unpack_n
from motrack.utils.lookup import LookupTable
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from mot_jepa.architectures.tdcp.core import MultiModalTDCP
from mot_jepa.common import conventions
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.datasets.dataset.common.data import VideoClipData, VideoClipPart
from mot_jepa.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import PredictionBBoxFeatureExtractor
from mot_jepa.datasets.dataset.motrack import MotrackDatasetWrapper
from mot_jepa.datasets.dataset.transform import Transform
from mot_jepa.utils import pipeline
from mot_jepa.utils.extra_features import ExtraFeaturesReader




class MyTracker(Tracker):
    def __init__(
        self,
        transform: Transform,
        model: nn.Module,
        extra_features_reader: ExtraFeaturesReader,
        device: str,
        sim_threshold: float = 0.5,
        initialization_threshold: int = 1,
        remember_threshold: int = 30,
        clip_length: Optional[int] = None,
        new_tracklet_detection_threshold: float = 0.7,
        use_conf: bool = True
    ):
        super().__init__()

        self._device = device

        self._transform = transform

        self._model: MultiModalTDCP = model
        self._model.to(device)
        self._model.eval()

        self._feature_names = self._model.feature_names
        self._extra_features_reader = extra_features_reader

        self._sim_threshold = sim_threshold

        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._clip_length = clip_length if clip_length is not None else self._remember_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold

        self._next_id = 0
        self._use_conf = use_conf

    def _association(
        self,
        tracklets: List[Tracklet],
        objects_data: List[dict],
        frame_index: int
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracks = len(tracklets)
        n_detections = len(objects_data)
        if n_tracks == 0:
            return [], [], list(range(n_detections))
        elif n_detections == 0:
            return [], list(range(n_tracks)), []

        data = self._convert_data(tracklets, objects_data, frame_index)
        data = self._transform(data)
        data.apply(lambda x: x.unsqueeze(0).to(self._device))
        track_features, det_features, _, _ = self._model(
            data.observed.features,
            data.observed.mask,
            data.unobserved.features,
            data.unobserved.mask
        )
        track_features = track_features[0].cpu()
        det_features = det_features[0].cpu()
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)

        cost_matrix = (track_features[:n_tracks] @ det_features[:n_detections].T).numpy()
        cost_matrix = 1 - (cost_matrix + 1) / 2 # [-1, 1] -> [0, 1]
        cost_matrix[cost_matrix > self._sim_threshold] = np.inf

        return hungarian(cost_matrix)

    @staticmethod
    def _remove_duplicates(objects_data: List[dict], detections: List[PredBBox]) -> List[dict]:
        override_detections = [PredBBox.create(BBox.from_xywh(*data['bbox_xywh']), label='pedestrian', conf=data['bbox_conf'])
                               for data in objects_data]
        n_override = len(override_detections)
        n_detections = len(detections)
        cost_matrix = np.zeros(shape=(n_override, n_detections), dtype=np.float32)
        for i in range(n_override):
            for j in range(n_detections):
                score = override_detections[i].iou(detections[j])
                cost_matrix[i, j] = -score if score > 0.75 else np.inf

        matches, _, _ = hungarian(cost_matrix)
        return [objects_data[i] for i, _ in matches]


    def track(self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        _, _ = frame, detections  # Ignored (for now)
        scene_name = self.get_scene()
        objects_data = self._extra_features_reader.read(scene_name, frame_index)
        objects_data = [data for data in objects_data if data['bbox_conf'] > 0.6]
        # objects_data = self._remove_duplicates(self._extra_features_reader.read(scene_name, frame_index), detections)
        detections = [PredBBox.create(BBox.from_xywh(*data['bbox_xywh']), label='pedestrian', conf=data['bbox_conf']) for data in objects_data]

        # Remove deleted
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]
        matches, unmatched_tracklets, unmatched_detections = self._association(tracklets, objects_data, frame_index)

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

            if tracklet.lost_time > self._remember_threshold or tracklet.state == TrackletState.NEW:
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

    def _convert_data(
        self,
        tracklets: List[Tracklet],
        objects_data: List[dict],
        frame_index: int
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

                PredictionBBoxFeatureExtractor._set_features(
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
            PredictionBBoxFeatureExtractor._set_features(
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


# class MyByteTracker(MyTracker):
#     def __init__(
#         self,
#         transform: Transform,
#         model: nn.Module,
#         device: str,
#         sim_threshold: float = 0.5,
#         initialization_threshold: int = 1,
#         remember_threshold: int = 30,
#         clip_length: Optional[int] = None,
#         new_tracklet_detection_threshold: float = 0.7,
#         use_conf: bool = True,
#         detection_threshold: float = 0.6
#     ):
#         super().__init__(
#             transform=transform,
#             model=model,
#             device=device,
#             sim_threshold=sim_threshold,
#             initialization_threshold=initialization_threshold,
#             remember_threshold=remember_threshold,
#             clip_length=clip_length,
#             new_tracklet_detection_threshold=new_tracklet_detection_threshold,
#             use_conf=use_conf
#         )
#         self._detection_threshold = detection_threshold
#
#     def track(self,
#         tracklets: List[Tracklet],
#         detections: List[PredBBox],
#         frame_index: int,
#         frame: Optional[np.ndarray] = None
#     ) -> List[Tracklet]:
#         tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]
#
#         # (1) Split detections into low and high
#         high_detections = [d for d in detections if d.conf >= self._detection_threshold]
#         high_det_indices = [i for i, d in enumerate(detections) if d.conf >= self._detection_threshold]
#         low_detections = [d for d in detections if d.conf < self._detection_threshold]
#         low_det_indices = [i for i, d in enumerate(detections) if d.conf < self._detection_threshold]
#
#         # (2) Match high detections with tracklets with states ACTIVE and LOST using HighMatchAlgorithm
#         tracklets_active_and_lost_indices, tracklets_active_and_lost = \
#             unpack_n([(i, t) for i, t in enumerate(tracklets) if t.is_tracked], n=2)
#         high_matches, remaining_tracklet_indices, high_unmatched_detections_indices = \
#             self._association(tracklets_active_and_lost, high_detections, frame_index)
#         high_matches = [(tracklets_active_and_lost_indices[t_i], high_det_indices[d_i]) for t_i, d_i in high_matches]
#         high_unmatched_detections_indices = [high_det_indices[d_i] for d_i in high_unmatched_detections_indices]
#         remaining_tracklets = [tracklets_active_and_lost[t_i] for t_i in remaining_tracklet_indices]
#         remaining_tracklet_indices = [tracklets_active_and_lost_indices[t_i] for t_i in remaining_tracklet_indices]
#
#         # (3) Match remaining ACTIVE tracklets with low detections using LowMatchAlgorithm
#         remaining_active_tracklet_indices, remaining_active_tracklets = \
#             unpack_n([(i, t) for i, t in zip(remaining_tracklet_indices, remaining_tracklets)
#                   if t.state == TrackletState.ACTIVE], n=2)
#         remaining_lost_tracklet_indices = \
#             [i for i, t in enumerate(tracklets) if t.state == TrackletState.LOST and i in remaining_tracklet_indices]
#
#         low_matches, low_unmatched_tracklet_indices, _ = \
#             self._association(remaining_active_tracklets, low_detections, frame_index)
#         low_matches = [(remaining_active_tracklet_indices[t_i], low_det_indices[d_i]) for t_i, d_i in low_matches]
#         unmatched_tracklet_indices = [remaining_active_tracklet_indices[t_i] for t_i in low_unmatched_tracklet_indices] + \
#                                      remaining_lost_tracklet_indices
#
#         # (5) Match NEW tracklets with high detections using NewMatchAlgorithm
#         remaining_high_detections = [detections[d_i] for d_i in high_unmatched_detections_indices]
#         remaining_high_detection_indices = high_unmatched_detections_indices
#         tracklets_new_indices, tracklets_new = \
#             unpack_n([(i, t) for i, t in enumerate(tracklets) if t.state == TrackletState.NEW], n=2)
#         new_matches, new_unmatched_tracklets_indices, new_unmatched_detections_indices = \
#             self._association(tracklets_new, remaining_high_detections, frame_index)
#         new_matches = [(tracklets_new_indices[t_i], high_unmatched_detections_indices[d_i]) for t_i, d_i in new_matches]
#         new_unmatched_tracklets_indices = [tracklets_new_indices[t_i] for t_i in new_unmatched_tracklets_indices]
#         new_unmatched_detections_indices = [remaining_high_detection_indices[d_i] for d_i in new_unmatched_detections_indices]
#
#         # (6) Initialize new tracklets from unmatched high detections
#         new_tracklets: List[Tracklet] = []
#         for d_i in new_unmatched_detections_indices:
#             detection = detections[d_i]
#
#             if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
#                 continue
#
#             new_tracklet = Tracklet(
#                 bbox=copy.deepcopy(detection),
#                 frame_index=frame_index,
#                 _id=self._next_id,
#                 state=TrackletState.NEW if frame_index > self._initialization_threshold else TrackletState.ACTIVE,
#                 max_history=self._clip_length - 1
#             )
#             self._next_id += 1
#             new_tracklets.append(new_tracklet)
#
#         # (7) Update matched tracklets
#         all_matches = high_matches + low_matches + new_matches
#         for t_i, d_i in all_matches:
#             tracklet = tracklets[t_i]
#             detection = detections[d_i]
#
#             new_state = TrackletState.ACTIVE
#             if tracklet.state == TrackletState.NEW and tracklet.total_matches + 1 < self._initialization_threshold:
#                 new_state = TrackletState.NEW
#             tracklet.update(detection, frame_index, state=new_state)
#
#         # (8) Delete new unmatched and long-lost tracklets
#         # Handle unmatched tracklets
#         for t_i in (unmatched_tracklet_indices + new_unmatched_tracklets_indices):
#             tracklet = tracklets[t_i]
#
#             if tracklet.lost_time > self._remember_threshold or tracklet.state == TrackletState.NEW:
#                 tracklet.state = TrackletState.DELETED
#             else:
#                 tracklet.state = TrackletState.LOST
#
#         tracklets.extend(new_tracklets)
#         return tracklets



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    torch.set_printoptions(precision=3, sci_mode=None)

    dataset_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split=cfg.eval.split,
        sequence_list=cfg.dataset.index.sequence_list
    )

    model = cfg.build_model()
    state_dict = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(state_dict['model'])

    if cfg.eval.object_detection.lookup_path is not None:
        with open(cfg.eval.object_detection.lookup_path, 'r', encoding='utf-8') as f:
            lookup = LookupTable.deserialize(json.load(f))
    else:
        lookup = None

    motrack_dataset_wrapper = MotrackDatasetWrapper(dataset_index)
    detection_manager = DetectionManager(
        inference_name=cfg.eval.object_detection.type,
        inference_params=cfg.eval.object_detection.params,
        dataset=motrack_dataset_wrapper,
        cache_path=cfg.eval.object_detection.cache_path,
        lookup=lookup
    )

    extra_features_reader = ExtraFeaturesReader(cfg.dataset.feature_extractor.extractor_params['prediction_path'])
    experiment_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    tracker_active_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        inference_type=conventions.InferenceType.ACTIVE
    )
    tracker_all_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        inference_type=conventions.InferenceType.ALL
    )
    tracker_postprocess_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        inference_type=conventions.InferenceType.POSTPROCESS
    )

    tracker = MyTracker(
        transform=cfg.dataset.build_transform(),
        model=model,
        extra_features_reader=extra_features_reader,
        device=cfg.resources.accelerator,
        remember_threshold=30,
        use_conf=True,
        sim_threshold=0.5
    )

    scene_names = dataset_index.scenes
    for scene_name in tqdm(scene_names):
        scene_info = dataset_index.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracker.reset_state()
        tracker.set_scene(scene_name)
        tracklets = []
        with TrackerInferenceWriter(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth,
                                    clip=True) as tracker_active_inf_writer, \
                TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth,
                                       clip=True) as tracker_all_inf_writer:
            for frame_index in tqdm(range(scene_length), desc=f'Tracking "{scene_name}"', unit='frame'):
                detections = detection_manager.predict(scene_name, frame_index)
                tracklets = tracker.track(tracklets, detections, frame_index)

                active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(frame_index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(frame_index, tracklet)

    if cfg.eval.postprocess:
        run_tracker_postprocess(
            dataset=motrack_dataset_wrapper,
            tracker_active_output=tracker_active_output,
            tracker_all_output=tracker_all_output,
            tracker_postprocess_output=tracker_postprocess_output
        )

    if cfg.eval.visualize:
        run_visualize_tracker_inference(
            dataset=motrack_dataset_wrapper,
            tracker_active_output=tracker_active_output,
            tracker_output_option=tracker_all_output if not cfg.eval.postprocess else tracker_postprocess_output
        )


if __name__ == '__main__':
    main()
