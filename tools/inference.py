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

from mot_jepa.architectures.tdcp.core import MultiModalTDCP, MultiModalTDSP
from mot_jepa.common import conventions
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.datasets.dataset.common.data import VideoClipData, VideoClipPart
from mot_jepa.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import PredictionBBoxFeatureExtractor, SupportedFeatures
from mot_jepa.datasets.dataset.motrack import MotrackDatasetWrapper
from mot_jepa.datasets.dataset.transform import Transform
from mot_jepa.utils import pipeline
from mot_jepa.utils.extra_features import ExtraFeaturesReader
import logging


logger = logging.getLogger('Inference')




class MyTracker(Tracker):
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
        super().__init__()

        self._device = device

        self._transform = transform

        self._model: MultiModalTDCP | MultiModalTDSP = model
        self._model.to(device)
        self._model.eval()

        self._feature_names = set([SupportedFeatures(feature_name) for feature_name in self._model.feature_names])
        self._extra_features_reader = extra_features_reader

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
        sim_threshold: float
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

            ALPHA = 0.0
            if ALPHA > 0.0:
                ema_track_mm_features = torch.zeros_like(track_mm_features)
                for t_i in range(track_mm_features.shape[0]):
                    tracklet = tracklets[t_i]
                    track_ema = tracklet.get('track_ema')
                    if track_ema is None:
                        ema_track_mm_features[t_i] = track_mm_features[t_i]
                    else:
                        ema_track_mm_features[t_i] = ALPHA * track_ema + (1 - ALPHA) * track_mm_features[t_i]
                        ema_track_mm_features[t_i] = F.normalize(ema_track_mm_features[t_i], dim=-1)
                    tracklet.set('track_ema', ema_track_mm_features[t_i])
            else:
                ema_track_mm_features = track_mm_features

            cost_matrix = (ema_track_mm_features @ det_mm_features.T).numpy()
            cost_matrix = 1 - (cost_matrix + 1) / 2 # [-1, 1] -> [0, 1]
        elif isinstance(self._model, MultiModalTDSP):
            logits, _ = self._model(
                data.observed.features,
                data.observed.mask,
                data.unobserved.features,
                data.unobserved.mask
            )
            probas = torch.sigmoid(logits).cpu().numpy()
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
        _, _ = frame, detections  # Ignored (for now)
        scene_name = self.get_scene()
        objects_data = self._extra_features_reader.read(scene_name, frame_index)
        objects_data = [data for data in objects_data if data['bbox_conf'] > self._detection_threshold]
        detections = [PredBBox.create(BBox.from_xywh(*data['bbox_xywh']), label='pedestrian', conf=data['bbox_conf']) for data in objects_data]

        # Remove deleted
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]
        matches, unmatched_tracklets, unmatched_detections = self._association(tracklets, objects_data, frame_index, sim_threshold=self._sim_threshold)

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

            if tracklet.lost_time > self._remember_threshold or tracklet.age < 0:  # 3 -> 1
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
    logger.info(f'Loading model from {cfg.eval.checkpoint}.')
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
        remember_threshold=150,
        use_conf=True,
        detection_threshold=0.1,
        sim_threshold=0.99,
        initialization_threshold=1,
        new_tracklet_detection_threshold=0.4
    )

    # DanceTrack exp111:
    # remember_threshold=50,
    # detection_threshold=0.4,
    # sim_threshold=0.985,
    # initialization_threshold=1,
    # new_tracklet_detection_threshold=0.9

    # SportsMOT exp04:
    # remember_threshold=150,
    # detection_threshold=0.1,
    # sim_threshold=0.99,
    # initialization_threshold=1,
    # new_tracklet_detection_threshold=0.4

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
