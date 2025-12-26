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
    def __init__(self):
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sync = dist.is_initialized()

        # State
        self._total_loss: float = 0.0
        self._steps: int = 0

    @property
    def step(self) -> int:
        return self._steps

    @torch.no_grad()
    def push(self, data: torch.Tensor) -> float:
        loss = data.detach()

        if self._sync:
            loss_reduced = loss.clone()
            dist.reduce(loss_reduced, dst=0)
            loss_value = float(loss_reduced.cpu() / self._world_size)
        else:
            loss_value = float(loss.cpu())

        self._total_loss += loss_value
        self._steps += 1

        return loss_value

    @torch.no_grad()
    def aggregate_and_flush(self) -> float:
        assert self._steps > 0, 'Nothing to aggregate!'

        # Aggregate
        avg_loss = self._total_loss / self._steps

        # Flush
        self._total_loss = 0.0
        self._steps = 0

        return avg_loss


class LossDictMeter(Meter):
    """
    Tracks and aggregates the dict-like loss over multiple steps.
    """
    def __init__(self):
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
                self._meters[name] = LossMeter()
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
        use_percentages: bool = True
    ):
        """
        Args:
            use_percentages: Returns values from interval [0, 100] if True else [0, 1]
        """
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sync = dist.is_initialized()
        self._use_percentages = use_percentages

        # State
        self._total: float = 0.0
        self._correct: float = 0.0

    @torch.no_grad()
    def push(self, outputs: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Add the current batch's predictions and targets to the accuracy meter.

        Args:
            outputs: The predicted labels.
            target: The true labels.
            mask: Mask
        """
        outputs, target = outputs.detach(), target.detach()
        if mask is not None:
            mask = mask.detach()
            outputs = outputs[~mask]
            target = target[~mask]
        outputs, target = outputs.view(-1), target.view(-1)

        if dist.is_initialized():
            delta_size = outputs.size(0) * self._world_size
            local_delta_correct = outputs.eq(target).sum()
            delta_correct_reduced = local_delta_correct.clone()
            dist.reduce(delta_correct_reduced, dst=0)
            delta_correct = float(delta_correct_reduced.cpu())
        else:
            delta_correct = float(outputs.eq(target).sum().cpu())
            delta_size = outputs.size(0)


        self._total += delta_size
        self._correct += delta_correct

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

        accuracy = self._correct / self._total \
            * (100.0 if self._use_percentages else 1.0)

        # Flush
        self._total *= 0
        self._correct *= 0

        return accuracy
