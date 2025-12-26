"""
Pipeline (tools scripts) functions.
"""
from dataclasses import fields
from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Callable

from omegaconf import DictConfig, OmegaConf
from tdlp.common import conventions, formats, project
from tdlp.utils import rich_tree

logger = logging.getLogger('PipelineUtils')


def task(task_name: str) -> Callable:
    """
    Optional decorator that wraps the task function in extra utilities.

    Args:
        task_name: Task name

    Returns:
        Task wrapper decorator
    """
    def task_wrapper(task_func: Callable) -> Callable:
        """
        Args:
            task_func: Function to wrap

        Returns:
            Wrapped function
        """

        def wrap(cfg: DictConfig):
            # Extracting `master_path` from optional `cfg.path.master`.
            master_path = None
            paths = cfg.get('path')
            if paths is not None:
                master_path = paths.get('master')
            master_path = project.OUTPUTS_PATH if master_path is None else master_path
            Path(master_path).mkdir(parents=True, exist_ok=True)

            # Store config history
            store_run_history_config(master_path, cfg, task_name=task_name)

            # execute the task
            start_time = time.time()
            # Parse config
            parsed_cfg = OmegaConf.to_object(cfg)

            # Print config
            # noinspection PyDataclass
            rich_tree.print_config_tree(
                cfg=cfg,
                print_order=[f.name for f in fields(parsed_cfg)],
                resolve=True,
                save_to_file=True
            )

            # Run
            task_func(cfg=parsed_cfg)
            logger.info(f"'{task_func.__name__}' execution time: {time.time() - start_time} (s)")

        return wrap

    return task_wrapper


def store_run_history_config(output_dir: str, cfg: DictConfig, task_name: str) -> None:
    """
    Stores run config of the task run.

    Args:
        output_dir: Task output path
        cfg: Task config
        task_name: Task name
    """
    experiment_path = conventions.get_experiment_path(output_dir, cfg.dataset_name, cfg.experiment_name)
    config_dirpath = conventions.get_run_history_path(experiment_path)
    dt = datetime.now().strftime(formats.DATETIME_FORMAT)
    config_path = os.path.join(config_dirpath, f'{dt}_{task_name}.yaml')
    Path(config_dirpath).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(cfg))
