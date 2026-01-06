
import json
import logging
import time

import numpy as np
from tqdm import tqdm

import hydra
from motrack.evaluation.io import TrackerInferenceWriter
from motrack.object_detection import DetectionManager
from motrack.tools.postprocess import run_tracker_postprocess
from motrack.tools.visualize import run_visualize_tracker_inference
from motrack.tracker.tracklet import TrackletState
from motrack.utils.lookup import LookupTable
from tdlp.common import conventions
from tdlp.common.project import CONFIGS_PATH
from tdlp.config_parser import GlobalConfig
from tdlp.datasets.dataset import dataset_index_factory
from tdlp.datasets.dataset.motrack import MotrackDatasetWrapper
from tdlp.tracker import TDLPOfflineTracker
from tdlp.utils import pipeline
from tdlp.utils.extra_features import ExtraFeaturesReader
import torch


logger = logging.getLogger('OfflineInference')



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    assert cfg.eval.tracker is not None, 'Tracker config is required for inference.'
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
        split=cfg.eval.split,
        inference_type=conventions.InferenceType.ACTIVE,
    )
    tracker_all_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        split=cfg.eval.split,
        inference_type=conventions.InferenceType.ALL,

    )
    tracker_postprocess_output = conventions.get_inference_path(
        experiment_path=experiment_path,
        split=cfg.eval.split,
        inference_type=conventions.InferenceType.POSTPROCESS
    )

    is_mot17 = cfg.dataset_name.startswith('MOT17')  # MOT17 is differently annotated (bboxes are not clipped to the image)
    tracker = TDLPOfflineTracker(
        transform=cfg.dataset.build_transform(),
        model=model,
        extra_features_reader=extra_features_reader,
        device=cfg.resources.accelerator,
        remember_threshold=cfg.eval.tracker.remember_threshold,
        use_conf=True,
        detection_threshold=cfg.eval.tracker.detection_threshold,
        sim_threshold=cfg.eval.tracker.sim_threshold,
        initialization_threshold=cfg.eval.tracker.initialization_threshold,
        new_tracklet_detection_threshold=cfg.eval.tracker.new_tracklet_detection_threshold
    )

    detector_times = []
    association_times = []

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
                                    clip=(not is_mot17)) as tracker_active_inf_writer, \
                TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth,
                                       clip=(not is_mot17)) as tracker_all_inf_writer:
            for frame_index in tqdm(range(scene_length), desc=f'Tracking "{scene_name}"', unit='frame'):
                # Just for benchmarking (detections are ignored by the tracker during the offline tracking)
                od_start_time = time.time()
                detections = detection_manager.predict(scene_name, frame_index) \
                    if cfg.eval.run_object_detection else []
                od_end_time = time.time()
                detector_times.append(od_end_time - od_start_time)

                # Tracking (association)
                tracking_start_time = time.time()
                tracklets = tracker.track(tracklets, detections, frame_index)
                tracking_end_time = time.time()
                association_times.append(tracking_end_time - tracking_start_time)
                active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(frame_index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(frame_index, tracklet)

    if cfg.eval.postprocess_enable:
        # Note: This is only relevant for debug and analysis (not used for evaluation)
        run_tracker_postprocess(
            dataset=motrack_dataset_wrapper,
            tracker_active_output=tracker_active_output,
            tracker_all_output=tracker_all_output,
            tracker_postprocess_output=tracker_postprocess_output,
            postprocess_cfg=cfg.eval.postprocess,
            clip=(not is_mot17)
        )

    if cfg.eval.visualize:
        run_visualize_tracker_inference(
            dataset=motrack_dataset_wrapper,
            tracker_active_output=tracker_active_output,
            tracker_output_option=tracker_all_output if not cfg.eval.postprocess_enable else tracker_postprocess_output
        )

    print(f'Detector time: {np.mean(detector_times)}')
    print(f'Association time: {np.mean(association_times)}')


if __name__ == '__main__':
    main()
