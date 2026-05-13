"""TDLP tracker hyperparameter optimization using Motrack's optimization core."""
import os

import hydra

# Side-effects: register 'tdlp' in TRACKER_CATALOG and 'tdlp_dancetrack' in DATASET_CATALOG
import tdlp.tracker        # noqa: F401
import tdlp.datasets       # noqa: F401

from motrack.config_parser import GlobalConfig
from motrack.tools import run_optimize
from motrack.utils import pipeline
from tdlp.common.project import MOTRACK_CONFIGS_PATH


@hydra.main(config_path=os.path.join(MOTRACK_CONFIGS_PATH, 'dancetrack'), config_name='tpe_tdlp', version_base='1.1')
@pipeline.task('motrack-optimize')
def main(cfg: GlobalConfig) -> None:
    run_optimize(cfg)


if __name__ == '__main__':
    main()
