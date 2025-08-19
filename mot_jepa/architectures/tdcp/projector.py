"""Projection network transforming track embeddings to detection space."""

import torch
from torch import nn
from typing import Optional


class TrackToDetectionProjector(nn.Module):
    """Map track features into the same space as detection embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_hidden_dim: int,
        input_dim: Optional[int] = None,
    ) -> None:
        """Args:
            hidden_dim: Target embedding dimension.
            intermediate_hidden_dim: Size of the hidden layer.
            input_dim: Optional input dimension; defaults to ``hidden_dim``.
        """

        super().__init__()
        input_dim = input_dim if input_dim is not None else hidden_dim
        self._hidden_dim = hidden_dim
        self._projector = nn.Sequential(
            nn.Linear(input_dim, intermediate_hidden_dim),
            nn.LayerNorm(intermediate_hidden_dim),
            nn.SiLU(),
            nn.Linear(intermediate_hidden_dim, hidden_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Track feature tensor.

        Returns:
            Projected features with shape ``(..., hidden_dim)``.
        """

        return self._projector(x)


def run_test() -> None:
    ttdp = TrackToDetectionProjector(
        hidden_dim=4,
        intermediate_hidden_dim=8
    )

    x_input = torch.randn(3, 4, 4)
    x_output = ttdp(x_input)
    expected_shape = (3, 4, 4)
    assert x_output.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {x_output.shape}.'


if __name__ == '__main__':
    run_test()
