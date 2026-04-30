"""Abstract base class for all TDLP association models."""
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class TDLPModel(nn.Module, ABC):
    """Abstract interface for all TDLP association models.

    All TDLP models (contrastive and similarity-based) must implement:
    - forward: run the model on track history and detection features.
    - prepare_loss_inputs: bridge model output to loss function signature.
    - compute_cost_matrix: convert model output to an association cost matrix.
    """

    @abstractmethod
    def prepare_loss_inputs(
        self,
        model_output: Any,
        track_mask: torch.Tensor,
        det_mask: torch.Tensor,
        track_ids: torch.Tensor,
        det_ids: torch.Tensor,
    ) -> tuple:
        """Convert model output to loss function arguments.

        Returns a tuple that can be unpacked directly into the loss function:
            loss_func(*prepare_loss_inputs(...))

        Args:
            model_output: Raw output from self.forward().
            track_mask: (B, N, T) boolean mask. True = missing.
            det_mask: (B, N) boolean mask. True = missing.
            track_ids: (B, N, T) track identity labels.
            det_ids: (B, N) detection identity labels.

        Returns:
            Tuple of positional arguments for the loss function.
        """
        ...

    @abstractmethod
    def compute_cost_matrix(
        self,
        model_output: Any,
        n_tracks: int,
        n_dets: int,
    ) -> torch.Tensor:
        """Compute association cost matrix from model output.

        Args:
            model_output: Raw output from self.forward().
            n_tracks: Number of valid tracks (rows in cost matrix).
            n_dets: Number of valid detections (columns in cost matrix).

        Returns:
            (B, n_tracks, n_dets) cost tensor in [0, 1]. Lower = better match.
        """
        ...
