"""Core TDCP model combining track and detection encoders."""

import copy
from typing import Tuple, Optional

import torch
from torch import nn

from mot_jepa.architectures.tdcp.detection_encoder import DetectionEncoder
from mot_jepa.architectures.tdcp.projector import TrackToDetectionProjector
from mot_jepa.architectures.tdcp.track_encoder import TrackEncoder
from mot_jepa.architectures.tdcp.object_interaction_encoder import ObjectInteractionEncoder


class TrackDetectionContrastivePrediction(nn.Module):
    """Contrastive model comparing track and detection embeddings."""

    def __init__(
        self,
        detection_encoder: DetectionEncoder,
        track_encoder: TrackEncoder,
        projector: TrackToDetectionProjector,
        object_interaction_encoder: Optional[ObjectInteractionEncoder] = None,
    ) -> None:
        """Args:
            detection_encoder: Encoder applied to raw detections.
            track_encoder: Temporal encoder for track sequences.
            projector: Projects track embeddings to detection space.
            object_interaction_encoder: Optional module modeling interactions
                between tracks and detections.
        """

        super().__init__()
        self._static_encoder = detection_encoder
        self._motion_encoder = copy.deepcopy(detection_encoder)
        self._track_encoder = track_encoder
        self._projector = projector
        self._object_interaction_encoder = object_interaction_encoder

    def forward(
        self,
        track_x: torch.Tensor,
        track_mask: torch.Tensor,
        det_x: torch.Tensor,
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            track_x: Track tensor of shape ``(B, N, T, 2D)`` containing static and
                motion features concatenated along the last dimension.
            track_mask: Boolean mask for ``track_x``.
            det_x: Detection tensor of shape ``(B, M, D)``.
            det_mask: Boolean mask for ``det_x``.

        Returns:
            Tuple ``(projected_track_features, detection_features)``.
        """

        det_features = self._static_encoder(det_x)
        half_dim = track_x.shape[-1] // 2
        track_static_x = self._static_encoder(track_x[..., :half_dim])
        track_motion_x = track_static_x + self._motion_encoder(track_x[..., half_dim:])
        track_features = self._track_encoder(track_motion_x, track_mask)
        projected_features = self._projector(track_features)

        if self._object_interaction_encoder is not None:
            agg_track_mask = track_mask.all(dim=-1)
            projected_features, det_features = self._object_interaction_encoder(
                projected_features, agg_track_mask, det_features, det_mask
            )

        return projected_features, det_features


def build_track_detection_contrastive_prediction_model(
    det_input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    track_encoder_n_heads: int = 8,
    track_encoder_n_layers: int = 6,
    track_encoder_ffn_dim: int = 512,
    projector_intermediate_dim: int = 512,
    interaction_encoder_enable: bool = False,
    interaction_encoder_n_heads: int = 8,
    interaction_encoder_n_layers: int = 6,
    interaction_encoder_ffn_dim: int = 512,
) -> TrackDetectionContrastivePrediction:
    """Build a complete TDCP model with default components.

    Args:
        det_input_dim: Dimensionality of detection inputs.
        hidden_dim: Shared embedding dimension across modules.
        dropout: Dropout rate used in all components.
        track_encoder_n_heads: Number of attention heads in the track encoder.
        track_encoder_n_layers: Number of transformer layers in the track encoder.
        track_encoder_ffn_dim: Feed-forward dimension in the track encoder.
        projector_intermediate_dim: Hidden dimension of the projector MLP.
        interaction_encoder_enable: Whether to include the interaction encoder.
        interaction_encoder_n_heads: Attention heads for the interaction encoder.
        interaction_encoder_n_layers: Layers in the interaction encoder.
        interaction_encoder_ffn_dim: Feed-forward dimension for the interaction encoder.

    Returns:
        Instantiated :class:`TrackDetectionContrastivePrediction` model.
    """

    detection_encoder = DetectionEncoder(
        input_dim=det_input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    track_encoder = TrackEncoder(
        hidden_dim=hidden_dim,
        n_heads=track_encoder_n_heads,
        n_layers=track_encoder_n_layers,
        ffn_dim=track_encoder_ffn_dim,
        dropout=dropout,
    )

    projector = TrackToDetectionProjector(
        hidden_dim=hidden_dim,
        intermediate_hidden_dim=projector_intermediate_dim,
    )

    if interaction_encoder_enable:
        object_interaction_encoder = ObjectInteractionEncoder(
            hidden_dim=hidden_dim,
            n_heads=interaction_encoder_n_heads,
            n_layers=interaction_encoder_n_layers,
            ffn_dim=interaction_encoder_ffn_dim,
            dropout=dropout,
        )
    else:
        object_interaction_encoder = None

    return TrackDetectionContrastivePrediction(
        detection_encoder=detection_encoder,
        track_encoder=track_encoder,
        projector=projector,
        object_interaction_encoder=object_interaction_encoder,
    )


def run_test() -> None:
    tdcp = build_track_detection_contrastive_prediction_model(
        det_input_dim=4,
        hidden_dim=4,
        track_encoder_n_heads=2,
        track_encoder_n_layers=1,
        track_encoder_ffn_dim=8,
        projector_intermediate_dim=8
    )

    track_features = torch.randn(3, 4, 5, 8)
    track_mask = torch.zeros(3, 4, 5, dtype=torch.bool)
    det_features = torch.randn(3, 4, 4)
    det_mask = torch.zeros(3, 4, dtype=torch.bool)
    x_output = tdcp(track_features, track_mask, det_features, det_mask)
    expected_shape = (3, 4, 4)
    assert x_output.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {x_output.shape}.'


if __name__ == '__main__':
    run_test()
