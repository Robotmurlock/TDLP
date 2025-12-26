"""Basic TDCP module using the last bounding box of each track.

The model encodes the final bounding box of each track along with the current
detections using a shared :class:`DetectionEncoder`.
"""

from typing import Tuple

import torch
from torch import nn

from tdlp.architectures.tdlp.feature_encoders import MotionEncoder


class LastBBoxTDCPBasic(nn.Module):
    """TDCP module that compares tracks with detections.

    The implementation encodes the last bounding box from each track and all
    detections with a shared encoder to produce feature embeddings that can be
    compared downstream.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """Args:
            input_dim: Detection feature dimension.
            hidden_dim: Size of the latent representation.
            dropout: Dropout rate applied in the encoder.
        """
        super().__init__()
        self._static_encoder = MotionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        track_x: torch.Tensor,
        track_mask: torch.Tensor,
        det_x: torch.Tensor,
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            track_x: Track tensor of shape ``(B, N, T, D)``.
            track_mask: Mask for ``track_x`` (unused).
            det_x: Detection tensor of shape ``(B, M, D)``.
            det_mask: Mask for ``det_x`` (unused).

        Returns:
            Tuple of ``(track_features, detection_features)`` after encoding.
        """

        _, _ = track_mask, det_mask
        track_x = track_x[:, :, -1, :]
        track_features = self._static_encoder(track_x)
        det_features = self._static_encoder(det_x)
        return track_features, det_features
