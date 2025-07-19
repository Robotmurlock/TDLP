"""
Implementation of Metric meters compatible with distributed computation.
"""
from typing import Union, Dict, Optional
from abc import ABC, abstractmethod

import torch
from torch import distributed as dist


MeterTensor = Union[torch.Tensor, Dict[str, torch.Tensor], float]


class Meter(ABC):
    """
    Trainer metric meter interface.
    """
    @abstractmethod
    def push(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Add the current loss to the total loss and increment the step count.

        Args:
            data: Current data
        """

    @abstractmethod
    def aggregate_and_flush(self) -> float:
        """
        Aggregate the average loss across all processes and reset the meter.

        Returns:
            Aggregated data across all processes.
        """


class LossMeter(Meter):
    """
    Tracks and aggregates the loss over multiple steps.
    """
    def __init__(
        self,
        device: Union[str, int, torch.device]
    ):
        """
        Args:
            device: The device where the loss will be stored.
        """
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sync = self._world_size > 1

        # State
        self._total_loss: torch.Tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._steps: int = 0

    @property
    def step(self) -> int:
        return self._steps

    @torch.no_grad()
    def push(self, data: torch.Tensor) -> torch.Tensor:
        loss = data.detach()

        if self._sync and dist.is_initialized():
            loss_reduced = loss.clone()
            dist.reduce(loss_reduced, dst=0)
            loss = loss_reduced / self._world_size

        self._total_loss += loss
        self._steps += 1

        return loss.item()

    @torch.no_grad()
    def aggregate_and_flush(self) -> float:
        assert self._steps > 0, 'Nothing to aggregate!'

        # Aggregate
        avg_loss = self._total_loss / self._steps
        if not self._sync and dist.is_initialized():
            avg_loss_reduced = avg_loss.clone()
            if dist.is_initialized():
                dist.reduce(avg_loss_reduced, dst=0)
                avg_loss = avg_loss_reduced / self._world_size

        # Flush
        self._total_loss *= 0.0
        self._steps = 0

        return avg_loss.cpu().item()


class LossDictMeter(Meter):
    """
    Tracks and aggregates the dict-like loss over multiple steps.
    """
    def __init__(
        self,
        device: Union[str, int, torch.device]
    ):
        """
        Args:
            device: The device where the loss will be stored.
        """
        self._device = device
        self._meters = {}

    @property
    def step(self) -> int:
        assert len(self._meters) > 0, 'No meters available!'
        for m in self._meters.values():
            return m.step

        return -1

    @torch.no_grad()
    def push(self, data: MeterTensor) -> MeterTensor:
        assert isinstance(data, dict), f'Data should be a dict. Got {type(data)}.'

        output: Dict[str, torch.Tensor] = {}
        for name, value in data.items():
            if name not in self._meters:
                self._meters[name] = LossMeter(self._device)
            output[name] = self._meters[name].push(value)

        return output

    @torch.no_grad()
    def aggregate_and_flush(self) -> MeterTensor:
        output: Dict[str, torch.Tensor] = {}
        for name, meter in self._meters.items():
            output[name] = meter.aggregate_and_flush()

        return output



class AccuracyMeter(Meter):
    """
    Tracks and aggregates the accuracy over multiple steps.
    """
    def __init__(
        self,
        device: Union[str, int, torch.device],
        use_percentages: bool = True
    ):
        """
        Args:
            device: The device where the accuracy will be stored.
            use_percentages: Returns values from interval [0, 100] if True else [0, 1]
        """
        self._use_percentages = use_percentages

        # State
        self._total: torch.Tensor = torch.tensor(0.0, dtype=torch.long, device=device)
        self._correct: torch.Tensor = torch.tensor(0.0, dtype=torch.long, device=device)

    @torch.no_grad()
    def push(self, outputs: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Add the current batch's predictions and targets to the accuracy meter.

        Args:
            outputs: The predicted labels.
            target: The true labels.
            mask: Mask
        """
        if mask is not None:
            outputs = outputs[~mask]
            target = target[~mask]
        outputs, target = outputs.detach().view(-1), target.detach().view(-1)

        self._total += outputs.size(0)
        self._correct += outputs.eq(target).sum()

    @torch.no_grad()
    def aggregate_and_flush(self) -> float:
        """
        Aggregate the accuracy across all processes and reset the meter.

        Returns:
            The accuracy after aggregation across all processes.

        Raises:
            AssertionError: If no samples have been accumulated.
        """
        assert self._total > 0, 'Nothing to aggregate!'

        # Aggregate
        total_reduced = self._total.clone()
        if dist.is_initialized():
            dist.reduce(total_reduced, dst=0)

        correct_reduced = self._correct.clone()
        if dist.is_initialized():
            dist.reduce(correct_reduced, dst=0)

        accuracy = correct_reduced.float() / total_reduced.float() \
            * (100.0 if self._use_percentages else 1.0)

        # Flush
        self._total *= 0
        self._correct *= 0

        return float(accuracy.cpu().item())
