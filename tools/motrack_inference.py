"""TDLP tracker inference using Motrack's inference pipeline."""
import os

import hydra

# Side-effects: register 'tdlp' in TRACKER_CATALOG and 'tdlp_dancetrack' in DATASET_CATALOG
import tdlp.tracker        # noqa: F401
import tdlp.datasets       # noqa: F401

from motrack.config_parser import GlobalConfig
from motrack.tools import run_inference
from motrack.utils import pipeline
from tdlp.common.project import MOTRACK_CONFIGS_PATH


@hydra.main(config_path=os.path.join(MOTRACK_CONFIGS_PATH, 'dancetrack'), config_name='tdlp_infer', version_base='1.1')
@pipeline.task('motrack-inference')
def main(cfg: GlobalConfig) -> None:
    run_inference(cfg)


if __name__ == '__main__':
    main()
