
import json
import logging

from tqdm import tqdm

import hydra
from motrack.library.cv.video_reader import MP4Reader
from motrack.library.cv.video_writer import MP4Writer
from motrack.object_detection.factory import object_detection_inference_factory
from motrack.tools.visualize import draw_tracklet
from motrack.tracker.tracklet import TrackletState
from motrack.utils.lookup import LookupTable
from tdlp.common.project import CONFIGS_PATH
from tdlp.config_parser import GlobalConfig
from tdlp.tracker import TDLPOnlineTracker
from tdlp.utils import pipeline
import torch


logger = logging.getLogger('OnlineInference')



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    assert cfg.eval.tracker is not None, 'Tracker config is required for inference.'
    assert cfg.eval.demo is not None, 'Demo config is required for demo.'
    torch.set_printoptions(precision=3, sci_mode=None)

    model = cfg.build_model()
    logger.info(f'Loading model from {cfg.eval.checkpoint}.')
    state_dict = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(state_dict['model'])

    if cfg.eval.object_detection.lookup_path is not None:
        with open(cfg.eval.object_detection.lookup_path, 'r', encoding='utf-8') as f:
            lookup = LookupTable.deserialize(json.load(f))
    else:
        lookup = None

    detector = object_detection_inference_factory(
        name=cfg.eval.object_detection.type,
        params=cfg.eval.object_detection.params,
        lookup=lookup
    )

    tracker = TDLPOnlineTracker(
        transform=cfg.dataset.build_transform(),
        model=model,
        device=cfg.resources.accelerator,
        remember_threshold=cfg.eval.tracker.remember_threshold,
        use_conf=True,
        detection_threshold=cfg.eval.tracker.detection_threshold,
        sim_threshold=cfg.eval.tracker.sim_threshold,
        initialization_threshold=cfg.eval.tracker.initialization_threshold,
        new_tracklet_detection_threshold=cfg.eval.tracker.new_tracklet_detection_threshold
    )

    tracklets = []
    input_video_path = cfg.eval.demo.video_path
    output_video_path = cfg.eval.demo.output_path
    fps = cfg.eval.demo.fps
    with MP4Reader(input_video_path) as video_reader, MP4Writer(output_video_path, fps) as video_writer:
        for frame_index, frame in tqdm(video_reader.iter(), desc='Tracking', unit='frame'):
            detections = detector.predict_bboxes(frame)
            tracklets = tracker.track(tracklets, detections, frame_index, frame)

            for tracklet in tracklets:
                if tracklet.state != TrackletState.ACTIVE:
                    continue
                frame = draw_tracklet(
                    frame=frame,
                    tracklet_id=tracklet.id,
                    tracklet_age=tracklet.age,
                    bbox=tracklet.bbox,
                    active=True
                )
            video_writer.write(frame)


if __name__ == '__main__':
    main()
