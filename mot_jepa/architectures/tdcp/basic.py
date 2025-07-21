from typing import Tuple

import torch
from torch import nn

from mot_jepa.architectures.tdcp.detection_encoder import DetectionEncoder


class LastBBoxTDCPBasic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self._static_encoder = DetectionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, track_x: torch.Tensor, track_mask: torch.Tensor, det_x: torch.Tensor, det_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        _, _ = track_mask, det_mask
        track_x = track_x[:, :, -1, :]
        track_features = self._static_encoder(track_x)
        det_features = self._static_encoder(det_x)
        return track_features, det_features
