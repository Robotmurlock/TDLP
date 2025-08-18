from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any

import einops
import torch
from torch import Tensor, nn


class TDCPAggregator(nn.Module, ABC):
    """Base class for modality aggregators.

    Args:
        n_features: Number of expected modalities.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    @abstractmethod
    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Aggregate a list of features.

        Args:
            features: Sequence of tensors each shaped [B, D].

        Returns:
            Tensor of shape [B, D].
        """
        raise NotImplementedError


class TDCPSumAggregator(TDCPAggregator):
    """Simple sum across modalities."""
    def forward(self, features: Sequence[Tensor]) -> Tensor:
        x = torch.stack(features, dim=1)
        return x.sum(dim=1)


class TDCPLinearSumAggregator(TDCPAggregator):
    """Per-modality linear projections + sum."""
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__(n_features)
        self._hidden_dim = hidden_dim
        self.proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_features)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        assert len(features) == self.n_features
        proj = [p(f) for p, f in zip(self.proj, features)]  # each [B, D]
        x = torch.stack(proj, dim=1)                         # [B, M, D]
        return self.norm(x.sum(dim=1))                       # [B, D]


class TDCPStaticSoftmaxSum(TDCPAggregator):
    """Learn global scalar weights per modality (input-independent)."""
    def __init__(self, n_features: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__(n_features)
        self._hidden_dim = hidden_dim
        self.logits = nn.Parameter(torch.zeros(n_features))
        self.temperature = temperature

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        x = torch.stack(features, dim=1)                     # [B, M, D]
        w = torch.softmax(self.logits / self.temperature, dim=0)  # [M]
        return (w.view(1, -1, 1) * x).sum(dim=1)             # [B, D]


class TDCPAttnWeightedSum(TDCPAggregator):
    """Input-dependent attention pooling across modalities via MLP scoring."""
    def __init__(self, n_features: int, hidden_dim: int, hidden: int = 256):
        super().__init__(n_features)
        self._hidden_dim = hidden_dim
        self.score_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        x = torch.stack(features, dim=1)     # [B, M, D]
        scores = self.score_net(x).squeeze(-1)        # [B, M]
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, M, 1]
        return (w * x).sum(dim=1)             # [B, D]


class TDCPQueryAttentionPool(TDCPAggregator):
    """Learned-query multihead attention pooling (Transformer-style)."""
    def __init__(self, n_features: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__(n_features)
        self._hidden_dim = hidden_dim
        self.q = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, D]
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        x = torch.stack(features, dim=2)  # [B, N, M, E]
        B, N, M, _ = x.shape
        x = einops.rearrange(x, 'b n m e -> (b n) m e')
        q = self.q.expand(B * N, -1, -1)     # [B, 1, E]
        pooled, _ = self.attn(q, x, x)   # [B, 1, E]
        pooled = self.out_norm(pooled[:, 0, :])  # [B, E]
        return einops.rearrange(pooled, '(b n) e -> b n e', b=B, n=N)


TDCP_AGGREGATOR_CATALOG = {
    'sum': TDCPSumAggregator,
    'linear_sum': TDCPLinearSumAggregator,
    'static_softmax': TDCPStaticSoftmaxSum,
    'attn': TDCPAttnWeightedSum,
    'query': TDCPQueryAttentionPool
}


def tdcp_aggregator_factory(aggregator_type: str, aggregator_params: Dict[str, Any], n_features: int) \
        -> TDCPAggregator:
    agg_cls = TDCP_AGGREGATOR_CATALOG[aggregator_type]
    return agg_cls(**aggregator_params, n_features=n_features)
