"""Feed-forward encoder for detection features."""

from typing import Dict, Any

import torch
from torch import nn


class MotionEncoder(nn.Module):
    """Two-layer feed-forward network with normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Args:
            input_dim: Dimensionality of the input detections.
            hidden_dim: Size of the latent representation.
            dropout: Dropout rate applied between layers.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of detection features.

        Returns:
            Encoded detection features with shape ``(..., hidden_dim)``.
        """

        return self.encoder(x)


class PartsAppearanceEncoder(nn.Module):
    """Encodes part-based appearance embeddings into a shared hidden space."""
    NUM_PARTS = 6

    def __init__(self, emb_size: int, hidden_dim: int, dropout: float = 0.1):
        """
        Linear projection of part-based features to a single hidden dimension.
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._emb_size = emb_size
        self._dropout = dropout

        self._linear_layers = nn.ModuleList([nn.Linear(emb_size, hidden_dim, bias=True)] * self.NUM_PARTS)
        self._drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P, E = x.shape[-2:]
        assert P == self.NUM_PARTS
        assert E == (self._emb_size + 1)
        embeddings = x[..., :self._emb_size]
        visibilities = x[..., self._emb_size]

        embeddings = embeddings * visibilities.unsqueeze(-1)
        projected = self._linear_layers[0](embeddings[..., 0, :])
        for i, layer in enumerate(self._linear_layers[1:]):
            projected += layer(embeddings[..., i + 1, :]) * visibilities[..., i + 1].unsqueeze(-1)
        projected = self._drop(projected)
        return projected


FEATURE_ENCODER_CATALOG = {
    'motion': MotionEncoder,
    'parts_appearance': PartsAppearanceEncoder
}


def feature_encoder_factory(feature_encoder_type: str, feature_encoder_params: Dict[str, Any]) -> nn.Module:
    agg_cls = FEATURE_ENCODER_CATALOG[feature_encoder_type]
    return agg_cls(**feature_encoder_params)


def run_test() -> None:
    de = MotionEncoder(
        input_dim=4,
        hidden_dim=3
    )

    x_input = torch.randn(3, 4, 4)
    x_output = de(x_input)
    expected_shape = (3, 4, 3)
    assert x_output.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {x_output.shape}.'


if __name__ == '__main__':
    run_test()
