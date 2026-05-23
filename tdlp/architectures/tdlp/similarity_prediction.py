"""Similarity prediction heads for TDLP."""
import math
from typing import Optional

from torch import nn
import torch
from torch.nn import functional as F


def create_pair_embedding(
    track_features: torch.Tensor,
    det_features: torch.Tensor,
) -> torch.Tensor:
    """
    Create pairwise embeddings between all tracks and all detections.

    Args:
        track_features: Track embeddings of shape (B, N, E)
        det_features: Detection embeddings of shape (B, M, E)
    
    Returns:
        Pair embeddings of shape (B, N, M, 3E) containing [z1, z2, |z1-z2|]
        for each track-detection pair
    """
    B, N, E = track_features.shape
    B, M, E = det_features.shape

    # Expand dimensions for pairwise comparison
    # track_features: (B, N, E) -> (B, N, 1, E)
    track_expanded = track_features.unsqueeze(2)
    # det_features: (B, M, E) -> (B, 1, M, E)
    det_expanded = det_features.unsqueeze(1)

    # Broadcast to create all pairwise combinations
    # track_broadcasted: (B, N, M, E)
    # det_broadcasted: (B, N, M, E)
    track_broadcasted = track_expanded.expand(B, N, M, E)
    det_broadcasted = det_expanded.expand(B, N, M, E)

    # Calculate absolute difference for each pair
    diff_features = torch.abs(track_broadcasted - det_broadcasted)

    # Concatenate along the last dimension: [z1, z2, |z1-z2|]
    pair_embeddings = torch.cat([track_broadcasted, det_broadcasted, diff_features], dim=-1)

    return pair_embeddings


class TDSPMLPHead(nn.Module):
    """MLP head for similarity prediction using full pair embeddings [z1, z2, |z1-z2|]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = self._projector = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        track_features: torch.Tensor,
        det_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for similarity prediction.
        
        Args:
            track_features: Track embeddings of shape (B, N, E)
            det_features: Detection embeddings of shape (B, M, E)
            
        Returns:
            Similarity scores of shape (B, N, M, 1)
        """
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)
        pair_embeddings = create_pair_embedding(track_features, det_features)
        B, N, M, E3 = pair_embeddings.shape
        pair_embeddings_flat = pair_embeddings.view(B * N * M, E3)

        similarity_scores_flat = self._mlp(pair_embeddings_flat)
        similarity_scores = similarity_scores_flat.view(B, N, M)
        return similarity_scores


class TDSPCompactMLPHead(nn.Module):
    """Compact MLP head for similarity prediction.

    Uses only |z1 - z2| as pair embedding (instead of [z1, z2, |z1-z2|])
    with an optional low-dimensional projection before pair creation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        proj_dim: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Dimension of input track/detection embeddings.
            hidden_dim: Hidden dimension of the MLP.
            proj_dim: If set, project inputs to this dimension before pair creation.
                      Reduces pair tensor from (B, N, M, input_dim) to (B, N, M, proj_dim).
        """
        super().__init__()
        self._proj = nn.Linear(input_dim, proj_dim) if proj_dim is not None else None
        pair_input_dim = proj_dim if proj_dim is not None else input_dim
        self._mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        track_features: torch.Tensor,
        det_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for compact similarity prediction.

        Args:
            track_features: Track embeddings of shape (B, N, E)
            det_features: Detection embeddings of shape (B, M, E)

        Returns:
            Similarity scores of shape (B, N, M)
        """
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)

        if self._proj is not None:
            track_features = self._proj(track_features)
            det_features = self._proj(det_features)

        diff = torch.abs(track_features.unsqueeze(2) - det_features.unsqueeze(1))
        B, N, M, D = diff.shape
        scores = self._mlp(diff.view(B * N * M, D)).view(B, N, M)
        return scores


class TDSPConcatMLPHead(nn.Module):
    """MLP head over plain concatenation [z1, z2] — no absolute-difference term."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        track_features: torch.Tensor,
        det_features: torch.Tensor,
    ) -> torch.Tensor:
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)
        B, N, E = track_features.shape
        _, M, _ = det_features.shape
        t = track_features.unsqueeze(2).expand(B, N, M, E)
        d = det_features.unsqueeze(1).expand(B, N, M, E)
        pair = torch.cat([t, d], dim=-1)
        scores = self._mlp(pair.reshape(B * N * M, 2 * E)).view(B, N, M)
        return scores


class TDSPDotProductHead(nn.Module):
    """Dot-product head: logit = tau * (z_t . z_d) + bias.

    L2-normalized inputs give cosine similarity in [-1, 1]; a learnable temperature
    and bias calibrate it for BCE-with-logits. Without calibration the sigmoid
    would saturate at ~0.73 for perfect matches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,  # noqa: ARG002 — kept for factory-signature compatibility
        init_temperature: float = 4.0,
    ):
        super().__init__()
        self._log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self._bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        track_features: torch.Tensor,
        det_features: torch.Tensor,
    ) -> torch.Tensor:
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)
        cos_sim = torch.bmm(track_features, det_features.transpose(1, 2))
        return cos_sim * torch.exp(self._log_temperature) + self._bias


SIMILARITY_HEAD_CATALOG = {
    'mlp': TDSPMLPHead,
    'compact_mlp': TDSPCompactMLPHead,
    'concat_mlp': TDSPConcatMLPHead,
    'dot_product': TDSPDotProductHead,
}


def similarity_head_factory(head_type: str, **kwargs) -> nn.Module:
    """Create a similarity prediction head by type name."""
    cls = SIMILARITY_HEAD_CATALOG.get(head_type)
    if cls is None:
        raise ValueError(f'Unknown similarity head type: {head_type}. Available: {list(SIMILARITY_HEAD_CATALOG.keys())}')
    return cls(**kwargs)


def test_pair_embedding():
    """Test function for pair embedding creation."""
    B, N, M, E = 2, 4, 6, 8

    # Create dummy track and detection features
    track_features = torch.randn(B, N, E)
    det_features = torch.randn(B, M, E)

    # Create pair embeddings
    pair_embeddings = create_pair_embedding(track_features, det_features)

    print(f'Track features shape: {track_features.shape}')
    print(f'Detection features shape: {det_features.shape}')
    print(f'Pair embeddings shape: {pair_embeddings.shape}')
    print(f'Expected shape: ({B}, {N}, {M}, {3*E})')

    # Test MLP head
    mlp_head = TDSPMLPHead(input_dim=E, hidden_dim=64)
    similarity_scores = mlp_head(track_features, det_features)
    print(f'Similarity scores shape: {similarity_scores.shape}')
    print(f'Expected shape: ({B}, {N}, {M}, 1)')

    # Verify pairwise structure
    print('\nPairwise verification:')
    print(f'Number of track-detection pairs: {N} × {M} = {N*M}')
    print(f'Each pair embedding dimension: {3*E}')
    print(f'Total pair embeddings: {B} × {N} × {M} × {3*E} = {B*N*M*3*E}')


def test_compact_mlp_head():
    """Test compact MLP head with and without projection."""
    B, N, M, E = 2, 4, 6, 8

    track_features = torch.randn(B, N, E)
    det_features = torch.randn(B, M, E)

    # Without projection
    head = TDSPCompactMLPHead(input_dim=E, hidden_dim=64)
    scores = head(track_features, det_features)
    assert scores.shape == (B, N, M), f'Expected ({B}, {N}, {M}) but got {scores.shape}'
    print(f'Compact MLP (no proj): {scores.shape}')

    # With projection
    head_proj = TDSPCompactMLPHead(input_dim=E, hidden_dim=64, proj_dim=4)
    scores_proj = head_proj(track_features, det_features)
    assert scores_proj.shape == (B, N, M), f'Expected ({B}, {N}, {M}) but got {scores_proj.shape}'
    print(f'Compact MLP (proj_dim=4): {scores_proj.shape}')

    # Factory test
    head_factory = similarity_head_factory('compact_mlp', input_dim=E, hidden_dim=64, proj_dim=4)
    scores_factory = head_factory(track_features, det_features)
    assert scores_factory.shape == (B, N, M)
    print('Factory test passed')


if __name__ == '__main__':
    test_pair_embedding()
    test_compact_mlp_head()

