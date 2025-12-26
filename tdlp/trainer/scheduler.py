"""
Implementation of custom schedulers.
"""
from torch import optim


def create_warmup_cosine_annealing_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
    n_warmup_epochs: int,
    epoch_steps: int,
):
    """
    Standard cosine annealing scheduler with warmup.

    Args:
        optimizer: Optimizer
        epochs: Number of epochs
        n_warmup_epochs: Number of warmup epochs
        epoch_steps: Number of steps per epoch

    Returns:
        Optimizer learning rate scheduler.
    """
    def warmup_scheduler_func(current_step: int):
        """
        Warmup scheduler learning rate multiplier for warm-up steps.

        Args:
            current_step: Current step

        Returns:
            Learning rate multiplier.
        """
        if n_warmup_epochs == 0:
            return 1.0
        return current_step / (n_warmup_epochs * epoch_steps)

    return optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=warmup_scheduler_func
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs * epoch_steps
            )
        ],
        milestones=[n_warmup_epochs * epoch_steps]
    )
