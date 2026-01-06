"""Base loss interfaces for video clip losses."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import nn


class VideoClipLoss(nn.Module, ABC):
    """Abstract base for losses operating on video clips."""
    @abstractmethod
    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        det_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_x: Tensor of shape (B, N, E)
            det_x: Tensor of shape (B, N, E)
            track_mask: Tensor of shape (B, N, T), 1 indicates missing, 0 indicates present
            detection_mask: Tensor of shape (B, N), 1 indicates missing, 0 indicates present
            track_feature_dict: Optional dictionary mapping modality names to track embeddings
            det_feature_dict: Optional dictionary mapping modality names to detection embeddings
            track_ids: Optional tensor of shape (B, N) with track identifiers
            det_ids: Optional tensor of shape (B, N) with detection identifiers
        Returns:
            Dictionary containing loss and additional debug information
        """
