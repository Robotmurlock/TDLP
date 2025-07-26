import copy
import json
from typing import List, Optional

import hydra
import numpy as np
import torch
from motrack.evaluation.io import TrackerInferenceWriter
from motrack.library.cv import PredBBox
from motrack.object_detection import DetectionManager
from motrack.tools.visualize import run_visualize_tracker_inference
from motrack.tracker import Tracker
from motrack.tracker.matching.utils import hungarian
from motrack.tracker.tracklet import Tracklet
from motrack.tracker.tracklet import TrackletState
from motrack.utils.lookup import LookupTable
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from mot_jepa.common import conventions
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.datasets.dataset.mot import VideoClipData, MOTClipDataset
from mot_jepa.datasets.dataset.motrack import MotrackDatasetWrapper
from mot_jepa.datasets.dataset.transform import Transform
from mot_jepa.utils import pipeline


class MyTracker(Tracker):
    def __init__(
        self,
        transform: Transform,
        model: nn.Module,
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

        self._model = model
        self._model.to(device)
        self._model.eval()

        self._sim_threshold = sim_threshold

        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._clip_length = clip_length if clip_length is not None else self._remember_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold

        self._next_id = 0
        self._use_conf = use_conf


    def track(self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        # Remove deleted
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]

        _ = frame  # Ignored
        n_tracks = len(tracklets)
        n_detections = len(detections)

        data = self._convert_data(tracklets, detections, frame_index)
        data = self._transform(data).serialize()
        data = {k: v.unsqueeze(0).to(self._device) for k, v in data.items()}
        track_features, det_features = self._model(
            data['observed_bboxes'],
            data['observed_temporal_mask'],
            data['unobserved_bboxes'],
            data['unobserved_temporal_mask'],
        )
        track_features = track_features[0].cpu()
        det_features = det_features[0].cpu()
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)
        cost_matrix = (track_features[:n_tracks] @ det_features[:n_detections].T).numpy()
        cost_matrix = 1 - (cost_matrix + 1) / 2 # [-1, 1] -> [0, 1]
        cost_matrix[cost_matrix > self._sim_threshold] = np.inf

        matches, unmatched_tracklets, unmatched_detections = hungarian(cost_matrix)

        # Handle matches
        for t_i, d_i in matches:
            tracklet = tracklets[t_i]
            detection = detections[d_i]

            new_state = TrackletState.ACTIVE
            if tracklet.state == TrackletState.NEW and tracklet.total_matches + 1 < self._initialization_threshold:
                new_state = TrackletState.NEW
            tracklet.update(detection, frame_index, state=new_state)

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

            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=copy.deepcopy(detection),
                frame_index=frame_index,
                _id=self._next_id,
                state=TrackletState.NEW if frame_index > self._initialization_threshold else TrackletState.ACTIVE,
                max_history=self._clip_length - 1
            )
            self._next_id += 1
            new_tracklets.append(new_tracklet)

        tracklets.extend(new_tracklets)

        return tracklets

    def _convert_data(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int) -> VideoClipData:
        n_tracks = len(tracklets)
        n_detections = len(detections)

        # Determine maximum size
        N = max(n_tracks, n_detections)

        # Observed initialization
        observed_bboxes = torch.zeros(N, self._clip_length, MOTClipDataset.BBOX_DIM, dtype=torch.float32)
        observed_ts = torch.zeros(N, self._clip_length, dtype=torch.long)
        observed_temporal_mask = torch.ones(N, self._clip_length, dtype=torch.bool)

        time_offset = frame_index - self._clip_length
        for t_i, tracklet in enumerate(tracklets):
            for hist_frame_index, bbox in tracklet.history:
                relative_index = hist_frame_index - time_offset
                if relative_index < 0:
                    continue
                if self._use_conf:
                    serialized_bbox = torch.tensor(bbox.as_numpy_xywh().tolist() + [bbox.conf], dtype=torch.float32)
                else:
                    serialized_bbox = torch.tensor(bbox.as_numpy_xywh().tolist() + [1.0], dtype=torch.float32)
                observed_bboxes[t_i, relative_index, :] = serialized_bbox
                observed_ts[t_i, relative_index] = hist_frame_index
                observed_temporal_mask[t_i, relative_index] = False

        # Unobserved initialization
        unobserved_bboxes = torch.zeros(N, MOTClipDataset.BBOX_DIM, dtype=torch.float32)
        unobserved_ts = torch.zeros(N, dtype=torch.long)
        unobserved_temporal_mask = torch.ones(N, dtype=torch.bool)

        unobserved_ts[:n_detections] = frame_index
        unobserved_temporal_mask[:n_detections] = False

        for d_i, detection in enumerate(detections):
            if self._use_conf:
                serialized_bbox = torch.tensor(detection.as_numpy_xywh().tolist() + [detection.conf], dtype=torch.float32)
            else:
                serialized_bbox = torch.tensor(detection.as_numpy_xywh().tolist() + [1.0], dtype=torch.float32)
            unobserved_bboxes[d_i, :] = serialized_bbox

        return VideoClipData(
            observed_bboxes=observed_bboxes,
            observed_ts=observed_ts,
            observed_temporal_mask=observed_temporal_mask,
            unobserved_bboxes=unobserved_bboxes,
            unobserved_ts=unobserved_ts,
            unobserved_temporal_mask=unobserved_temporal_mask
        )


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
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

    experiment_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    tracker_active_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        inference_type=conventions.InferenceType.ACTIVE
    )
    tracker_all_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        inference_type=conventions.InferenceType.ALL
    )

    scene_names = dataset_index.scenes
    for scene_name in tqdm(scene_names):
        scene_info = dataset_index.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracklets = []
        tracker = MyTracker(
            model=model,
            device=cfg.resources.accelerator,
            transform=cfg.dataset.build_transform(),
            remember_threshold=30,
            use_conf=True
        )

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

    if cfg.eval.visualize:
        run_visualize_tracker_inference(
            dataset=motrack_dataset_wrapper,
            tracker_active_output=tracker_active_output,
            tracker_output_option=tracker_all_output
        )


if __name__ == '__main__':
    main()
