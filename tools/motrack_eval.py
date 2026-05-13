"""TDLP tracker evaluation using Motrack's eval pipeline.

Run after ``tools/motrack_inference.py`` to compute HOTA / IDF1 / MOTA on
the inference outputs and log a run to MLflow (when ``cfg.mlflow.enabled``).
"""
import os

import hydra

# Side-effects: register 'tdlp' in TRACKER_CATALOG and 'tdlp_dancetrack' in DATASET_CATALOG
import tdlp.tracker        # noqa: F401
import tdlp.datasets       # noqa: F401

from motrack.config_parser import GlobalConfig
from motrack.tools import run_eval
from motrack.tools.mlflow_logger import load_and_log_run
from motrack.utils import pipeline
from tdlp.common.project import MOTRACK_CONFIGS_PATH


@hydra.main(config_path=os.path.join(MOTRACK_CONFIGS_PATH, 'dancetrack'), config_name='tdlp_infer', version_base='1.1')
@pipeline.task('motrack-eval')
def main(cfg: GlobalConfig) -> None:
    run_eval(cfg)
    load_and_log_run(cfg)


if __name__ == '__main__':
    main()
