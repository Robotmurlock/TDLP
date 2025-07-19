from torch import nn
import torch
from typing import Dict
from abc import ABC, abstractmethod


class VideoClipLoss(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_x: Tensor of shape (B, N, E)
            det_x: Tensor of shape (B, N, E)
            track_mask: Tensor of shape (B, N, T), 1 indicates missing, 0 indicates present
            detection_mask: Tensor of shape (B, N), 1 indicates missing, 0 indicates present
        Returns:
            Dictionary containing loss and additional debug information
        """