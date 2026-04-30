"""Trainer factory for building trainers from config."""
from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tdlp.datasets.dataset.transform import Transform
from tdlp.trainer.end_to_end_trainer import EndToEndTrainer
from tdlp.trainer.trainer import ContrastiveTrainer
from tdlp.trainer.types import TrainerType


def build_trainer(
    trainer_type: TrainerType,
    trainer_params: Optional[dict],
    model: nn.Module,
    loss_func: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    n_epochs: int,
    gradient_clip: Optional[float],
    mixed_precision: bool,
    device: Optional[str],
    tensorboard_log_dirpath: str,
    checkpoints_dirpath: str,
    metric_monitor: str,
    transform: Optional[Transform] = None,
) -> ContrastiveTrainer:
    """Build a trainer based on the trainer type.

    Args:
        trainer_type: Which trainer to build.
        trainer_params: Extra parameters forwarded to the trainer constructor
            (e.g. n_gradient_frames, sim_threshold for E2E).
        transform: Data transform, required for E2E trainer
            (dataset must be loaded with disable_transform=True).
    """
    common = dict(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        gradient_clip=gradient_clip,
        mixed_precision=mixed_precision,
        device=device,
        tensorboard_log_dirpath=tensorboard_log_dirpath,
        checkpoints_dirpath=checkpoints_dirpath,
        metric_monitor=metric_monitor,
    )

    if trainer_type == TrainerType.DEFAULT:
        assert trainer_params is None or len(trainer_params) == 0, \
            f'Default trainer does not accept trainer_params, got: {trainer_params}'
        return ContrastiveTrainer(**common)

    if trainer_type == TrainerType.END_TO_END:
        assert transform is not None, \
            'E2E trainer requires a transform (dataset must have disable_transform=True)'
        params = trainer_params or {}
        return EndToEndTrainer(**common, transform=transform, **params)

    raise ValueError(f'Unknown trainer type: {trainer_type}')
