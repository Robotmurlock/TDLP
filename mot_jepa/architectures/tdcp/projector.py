import torch
from torch import nn


class TrackToDetectionProjector(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_hidden_dim: int):
        super().__init__()
        self._projector = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_hidden_dim),
            nn.LayerNorm(intermediate_hidden_dim),
            nn.SiLU(),
            nn.Linear(intermediate_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
