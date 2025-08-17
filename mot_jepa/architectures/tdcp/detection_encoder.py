"""Feed-forward encoder for detection features."""

import torch
from torch import nn


class DetectionEncoder(nn.Module):
    """Two-layer feed-forward network with normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
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


def run_test() -> None:
    de = DetectionEncoder(
        input_dim=4,
        hidden_dim=3
    )

    x_input = torch.randn(3, 4, 4)
    x_output = de(x_input)
    expected_shape = (3, 4, 3)
    assert x_output.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {x_output.shape}.'


if __name__ == '__main__':
    run_test()
