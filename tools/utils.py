import logging
import os
import shutil
import sys
from typing import Optional

from tdlp.common import conventions
from tdlp.trainer import torch_distrib_utils

logger = logging.getLogger('ToolsUtility')


def check_train_experiment_history(
    train_resume: bool,
    train_truncate: bool,
    master_path: str,
    dataset_name: str,
    experiment_name: str,
) -> Optional[str]:
    """
    Checks if trained model already exists. If it already exists then user is asked if he wants to override
    old model (delete old model).

    Args:
        train_resume: Resume training from previous checkpoint.
        train_truncate: Delete all experiment history before starting training from scratch.
        master_path: Output path
        dataset_name: Dataset name
        experiment_name: Experiment name

    Return:
        Checkpoint path if a checkpoint path exists.
    """
    experiment_path = conventions.get_experiment_path(master_path, dataset_name, experiment_name)
    dirpaths = [
        conventions.get_tensorboard_logs_dirpath(master_path, dataset_name, experiment_name),
        conventions.get_checkpoints_dirpath(experiment_path)
    ]

    # Check if there are already some checkpoints or TB logs
    if any(os.path.exists(dirpath) for dirpath in dirpaths):
        logger.warning(f'Experiment "{experiment_name}" already has some history.')
        if train_resume:
            ckpt_path: str = conventions.get_latest_checkpoint_path(experiment_path)
            if not os.path.exists(ckpt_path):
                logger.warning(
                    f'Failed to find latest checkpoint at path "{ckpt_path}". Initializing model from scratch.')
                return None

            return ckpt_path

        if train_truncate:
            if torch_distrib_utils.is_zero_rank():
                logger.warning(f'Deleting all experiment "{experiment_name}" history!')
                for dirpath in dirpaths:
                    if os.path.exists(dirpath):
                        shutil.rmtree(dirpath)

            torch_distrib_utils.dist_barrier()
            return None
        else:
            logger.warning('Please choose train resume or truncate option. Aborting...')
            sys.exit()