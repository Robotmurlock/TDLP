"""Transformer-based encoder modeling interactions between tracks and detections."""

from typing import Tuple

import einops
import torch
from torch import nn


class ObjectInteractionEncoder(nn.Module):
    """Encode joint track and detection features with self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Args:
            hidden_dim: Size of the feature embeddings.
            n_heads: Number of attention heads.
            n_layers: Number of transformer encoder layers.
            ffn_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout rate applied inside transformer layers.
        """

        super().__init__()
        self._hidden_dim = hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=nn.SiLU(),
            batch_first=False,
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        track_x: torch.Tensor,
        track_mask: torch.Tensor,
        det_x: torch.Tensor,
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            track_x: Track feature tensor of shape ``(B, N, E)``.
            track_mask: Boolean mask for ``track_x``.
            det_x: Detection feature tensor of shape ``(B, M, E)``.
            det_mask: Boolean mask for ``det_x``.

        Returns:
            Tuple ``(track_features, detection_features)`` with updated
            representations after self-attention.
        """

        N = track_x.shape[1]

        x = torch.cat([track_x, det_x], dim=1)
        masks = torch.cat([track_mask, det_mask], dim=1)

        x = einops.rearrange(x, 'b n e -> n b e')
        x = self._encoder(x, src_key_padding_mask=masks)
        x = einops.rearrange(x, 'n b e -> b n e')

        track_x, det_x = x[:, :N, :].contiguous(), x[:, N:, :].contiguous()

        return track_x, det_x


def run_test() -> None:
    oin = ObjectInteractionEncoder(
        hidden_dim=4,
        n_heads=2,
        n_layers=1,
        ffn_dim=6
    )

    x_input = torch.randn(3, 5, 4)
    x_mask = torch.zeros(3, 5, dtype=torch.bool)
    x_output = oin(x_input, x_mask, x_input, x_mask)
    output_shapes = [xo.shape for xo in x_output]
    expected_shape = (3, 5, 4)
    assert all((shape == expected_shape) for shape in output_shapes), f'Test failed! Expected shape {expected_shape} but got {output_shapes}.'


if __name__ == '__main__':
    run_test()
