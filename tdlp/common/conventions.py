"""
Dataset, model and inference file system structure

Pipeline data structure:
{master_path}/
    {dataset_name}/
        experiments/
            {experiment_name}/
                checkpoints/
                    {checkpoint_name_1}
                    {checkpoint_name_2}
                    ...
                run_history/*
                configs/
                    [train.yaml]
                    [inference.yaml] (last inference config)
                    [visualize.yaml]
                    ...
        tensorboard_logs/
            {experiment_name}/*
    analysis/*
"""
import os
import enum

EXPERIMENTS_DIRNAME = 'experiments'
TENSORBOARD_DIRNAME = 'tensorboard_logs'
CHECKPOINTS_DIRNAME = 'checkpoints'
RUN_HISTORY_DIRNAME = 'run_history'
INFERENCES_DIRNAME = 'inferences'
INFERENCE_VISUALIZATIONS_DIRNAME = 'visualizations'
CONFIGS_DIRNAME = 'configs'
ANALYSIS_DIRNAME = 'analysis'
LAST_CKPT = 'last.pt'


class InferenceType(enum.Enum):
    ACTIVE = 'active'
    ALL = 'all'
    POSTPROCESS = 'postprocess'


def get_experiment_path(master_path: str, dataset_name: str, experiment_name: str) -> str:
    """
    Gets experiment location given 'master_path'

    Args:
        master_path: Global path (master path)
        dataset_name: Dataset name
        experiment_name: Model experiment name

    Returns:
        Experiment absolute path
    """
    return os.path.join(master_path, EXPERIMENTS_DIRNAME, dataset_name, experiment_name)


def get_run_history_path(experiment_path: str) -> str:
    """
    Gets experiment's run history.

    Args:
        experiment_path: Experiment path

    Returns:
        Experiment's run history
    """
    return os.path.join(experiment_path, RUN_HISTORY_DIRNAME)


def get_checkpoints_dirpath(experiment_path: str) -> str:
    """
    Gets path to all checkpoints for given experiment

    Args:
        experiment_path: Path to experiment

    Returns:
        Checkpoints directory
    """
    return os.path.join(experiment_path, CHECKPOINTS_DIRNAME)


def get_checkpoint_path(experiment_path: str, checkpoint_filename: str) -> str:
    """
    Gets checkpoint full path.

    Args:
        experiment_path: Path to experiment
        checkpoint_filename: Checkpoint name

    Returns:
        Checkpoint full path
    """
    return os.path.join(get_checkpoints_dirpath(experiment_path), checkpoint_filename)


def get_latest_checkpoint_path(experiment_path: str) -> str:
    """
    Gets latest checkpoint full path.

    Args:
        experiment_path: Path to experiment

    Returns:
        Latest checkpoint full path
    """
    return get_checkpoint_path(experiment_path, LAST_CKPT)


def get_inferences_path(experiment_path: str) -> str:
    """
    Gets the all experiment inference directory path..

    Args:
        experiment_path: Path to experiment

    Returns:
        Inferences directory path
    """
    return os.path.join(experiment_path, INFERENCES_DIRNAME)


def get_inference_path(experiment_path: str, split: str, inference_type: InferenceType) -> str:
    """
    Gets the inference type path.

    Args:
        experiment_path: Path to experiment
        split: Split
        inference_type: Inference type (all, active, postprocess, etc.)

    Returns:
        Inference path
    """
    return os.path.join(get_inferences_path(experiment_path), split, inference_type.value)


def get_config_dirpath(experiment_path: str) -> str:
    """
    Gets path to experiment configs.

    Args:
        experiment_path: Experiment path

    Returns:
        Path to saved configs (history) for given experiment path
    """
    return os.path.join(experiment_path, CONFIGS_DIRNAME)


def get_config_path(experiment_path: str, config_name: str) -> str:
    """
    Gets configs full path.

    Args:
        experiment_path: Path to experiment
        config_name: Config name

    Returns:
        Config full path
    """
    return os.path.join(get_config_dirpath(experiment_path), config_name)


def get_tensorboard_logs_dirpath(
    master_path: str,
    dataset_name: str,
    experiment_name: str
) -> str:
    """
    Gets location where all tensorboard logs for one dataset are stored.

    Args:
        master_path: Master path
        dataset_name: Dataset name
        experiment_name: Experiment name

    Returns:
        Tensorboard directory path
    """
    return os.path.join(master_path, TENSORBOARD_DIRNAME, dataset_name, experiment_name)
