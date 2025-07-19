import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mot_jepa.architectures.tdcp.detection_encoder import DetectionEncoder
from mot_jepa.architectures.tdcp.projector import TrackToDetectionProjector
from mot_jepa.architectures.tdcp.track_encoder import TrackEncoder


class TrackDetectionContrastivePrediction(nn.Module):
    def __init__(
        self,
        detection_encoder: DetectionEncoder,
        track_encoder: TrackEncoder,
        projector: TrackToDetectionProjector
    ):
        super().__init__()
        self._static_encoder = detection_encoder
        self._motion_encoder = copy.deepcopy(detection_encoder)
        self._track_encoder = track_encoder
        self._projector = projector

    def forward(self, track_x: torch.Tensor, track_mask: torch.Tensor, det_x: torch.Tensor, det_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        det_features = self._static_encoder(det_x)
        half_dim = track_x.shape[-1] // 2
        track_static_x = self._static_encoder(track_x[..., :half_dim])
        track_motion_x = track_static_x + self._motion_encoder(track_x[..., half_dim:])
        track_features = self._track_encoder(track_motion_x, track_mask)
        projected_features = self._projector(track_features)

        return projected_features, det_features


def build_track_detection_contrastive_prediction_model(
    det_input_dim: int,
    track_input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    track_encoder_n_heads: int = 8,
    track_encoder_n_layers: int = 6,
    track_encoder_ffn_dim: int = 512,
    projector_intermediate_dim: int = 512
) -> TrackDetectionContrastivePrediction:
    detection_encoder = DetectionEncoder(
        input_dim=det_input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    track_encoder = TrackEncoder(
        input_dim=track_input_dim,
        hidden_dim=hidden_dim,
        n_heads=track_encoder_n_heads,
        n_layers=track_encoder_n_layers,
        ffn_dim=track_encoder_ffn_dim,
        dropout=dropout
    )

    projector = TrackToDetectionProjector(
        hidden_dim=hidden_dim,
        intermediate_hidden_dim=projector_intermediate_dim
    )

    return TrackDetectionContrastivePrediction(
        detection_encoder=detection_encoder,
        track_encoder=track_encoder,
        projector=projector
    )


def run_test() -> None:
    tdcp = build_track_detection_contrastive_prediction_model(
        det_input_dim=4,
        track_input_dim=8,
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
