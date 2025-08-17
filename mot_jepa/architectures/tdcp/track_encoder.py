"""Track encoder components used in TDCP architectures."""

import math
import einops
import torch
from torch import nn


class ReversedPositionalEncoding(nn.Module):
    """
    Reversed positional encoding. Same as the standard positional encoding
    make sure the last element always gets the same encoding.

    Note: Missing values should be padded before usage.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Model size
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Generate positions in reverse order
        position = torch.arange(max_len - 1, -1, -1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    @property
    def encoding(self) -> torch.Tensor:
        """
        Get the positional encoding vector.

        Returns:
            Positional encoding vector
        """
        return self.pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[-x.size(0):]
        return self.dropout(x)


class TrackEncoder(nn.Module):
    """Temporal encoder that processes per-track sequences."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Args:
            hidden_dim: Feature dimension for each object.
            n_heads: Number of attention heads.
            n_layers: Number of transformer encoder layers.
            ffn_dim: Hidden size of the feed-forward networks.
            dropout: Dropout rate used throughout the encoder.
        """

        super().__init__()
        self._hidden_dim = hidden_dim
        self._pos_encoder = ReversedPositionalEncoding(hidden_dim)
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

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape ``(B, N, T, D)`` where ``B`` is batch size,
                ``N`` number of tracks, ``T`` temporal length and ``D`` feature
                dimension.
            masks: Boolean mask tensor of shape ``(B, N, T)`` marking padded
                timesteps.

        Returns:
            Tensor of shape ``(B, N, D)`` representing encoded tracks.
        """

        B, N, T, D = x.shape  # batch, objects, temporal, dim

        # Tokenize
        x = einops.rearrange(x, 'b n t e -> t (b n) e')
        masks = einops.rearrange(masks, 'b n t -> (b n) t')

        # Transformer preprocess (position encoding and normalization)
        x = self._pos_encoder(x)

        # Expand and prepend CLS token
        cls_token = self.cls_token.expand(1, B * N, -1)  # (1, N * B, E)
        x = torch.cat([x, cls_token], dim=0)  # (T + 1, N * B, E)

        # Adjust masks to account for added CLS token (set to False -> not masked)
        cls_mask = torch.zeros(B * N, 1, dtype=torch.bool, device=masks.device)
        masks = torch.cat([masks, cls_mask], dim=1)  # (T + 1, B * N)

        # Transformer
        x = self._encoder(x, src_key_padding_mask=masks)

        x = einops.rearrange(x, 't (b n) e -> b n t e', b=B, n=N)
        return x[:, :, -1, :]  # Return CLS token output


def run_test() -> None:
    te = TrackEncoder(
        hidden_dim=4,
        n_heads=2,
        n_layers=1,
        ffn_dim=6
    )

    x_input = torch.randn(3, 4, 5, 5)
    x_mask = torch.zeros(3, 4, 5, dtype=torch.bool)
    x_output = te(x_input, x_mask)
    expected_shape = (3, 4, 4)
    assert x_output.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {x_output.shape}.'


if __name__ == '__main__':
    run_test()
