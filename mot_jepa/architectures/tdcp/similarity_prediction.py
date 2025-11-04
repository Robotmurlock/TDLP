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
    """MLP head for similarity prediction."""

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


def test_pair_embedding():
    """Test function for pair embedding creation."""
    B, N, M, E = 2, 4, 6, 8
    
    # Create dummy track and detection features
    track_features = torch.randn(B, N, E)
    det_features = torch.randn(B, M, E)
    
    # Create pair embeddings
    pair_embeddings = create_pair_embedding(track_features, det_features)
    
    print(f"Track features shape: {track_features.shape}")
    print(f"Detection features shape: {det_features.shape}")
    print(f"Pair embeddings shape: {pair_embeddings.shape}")
    print(f"Expected shape: ({B}, {N}, {M}, {3*E})")
    
    # Test MLP head
    mlp_head = TDSPMLPHead(input_dim=E, hidden_dim=64)
    similarity_scores = mlp_head(track_features, det_features)
    
    print(f"Similarity scores shape: {similarity_scores.shape}")
    print(f"Expected shape: ({B}, {N}, {M}, 1)")
    
    # Verify pairwise structure
    print(f"\nPairwise verification:")
    print(f"Number of track-detection pairs: {N} × {M} = {N*M}")
    print(f"Each pair embedding dimension: {3*E}")
    print(f"Total pair embeddings: {B} × {N} × {M} × {3*E} = {B*N*M*3*E}")


if __name__ == '__main__':
    test_pair_embedding()
        