"""Core TDCP model combining track and detection encoders."""

import copy
import random
from typing import List, Tuple, Optional, Dict, Any, Set

import torch
from torch import nn

from mot_jepa.architectures.tdcp import utils as tdcp_utils
from mot_jepa.architectures.tdcp.aggregators import tdcp_aggregator_factory, TDCPAggregator
from mot_jepa.architectures.tdcp.feature_encoders import feature_encoder_factory
from mot_jepa.architectures.tdcp.object_interaction_encoder import ObjectInteractionEncoder
from mot_jepa.architectures.tdcp.projector import TrackToDetectionProjector
from mot_jepa.architectures.tdcp.track_encoder import TrackEncoder


class TrackDetectionContrastivePrediction(nn.Module):
    """Contrastive model comparing track and detection embeddings."""

    def __init__(
        self,
        feature_encoder: nn.Module,
        track_encoder: TrackEncoder,
        projector: TrackToDetectionProjector,
        object_interaction_encoder: Optional[ObjectInteractionEncoder] = None,
        enable_motion_encoder: bool = True
    ) -> None:
        """Args:
            feature_encoder: Encoder applied to raw detections.
            track_encoder: Temporal encoder for track sequences.
            projector: Projects track embeddings to detection space.
            object_interaction_encoder: Optional module modeling interactions
                between tracks and detections.
            enable_motion_encoder: Enable motion encoder (use if FoD is applied in the transform function)
        """
        super().__init__()
        self._static_encoder = feature_encoder
        self._motion_encoder = copy.deepcopy(feature_encoder) if enable_motion_encoder else None
        self._track_encoder = track_encoder
        self._projector = projector
        self._object_interaction_encoder = object_interaction_encoder

    @property
    def output_dim(self) -> int:
        return self._projector.output_dim

    def forward(
        self,
        track_x: torch.Tensor,
        track_mask: torch.Tensor,
        det_x: torch.Tensor,
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            track_x: Track tensor of shape ``(B, N, T, 2D)`` containing static and
                motion features concatenated along the last dimension.
            track_mask: Boolean mask for ``track_x``.
            det_x: Detection tensor of shape ``(B, M, D)``.
            det_mask: Boolean mask for ``det_x``.

        Returns:
            Tuple ``(projected_track_features, detection_features)``.
        """

        det_features = self._static_encoder(det_x)

        if self._motion_encoder is not None:
            half_dim = track_x.shape[-1] // 2
            track_static_x = self._static_encoder(track_x[..., :half_dim])
            track_x = track_static_x + self._motion_encoder(track_x[..., half_dim:])
        else:
            track_x = self._static_encoder(track_x)

        track_features = self._track_encoder(track_x, track_mask)
        projected_features = self._projector(track_features)

        if self._object_interaction_encoder is not None:
            agg_track_mask = track_mask.all(dim=-1)
            projected_features, det_features = self._object_interaction_encoder(
                projected_features, agg_track_mask, det_features, det_mask
            )

        return projected_features, det_features


class MultiModalTDCP(nn.Module):
    def __init__(
        self,
        tdcps: Dict[str, TrackDetectionContrastivePrediction],
        mm_dim: int,
        aggregator: TDCPAggregator,
        drop_mm_probas: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self._tdcps = nn.ModuleDict(tdcps)
        self._mm_linear_layers = nn.ModuleList([nn.Linear(tdcp.output_dim, mm_dim) for tdcp in tdcps.values()])
        self._aggregator = aggregator
        self._drop_mm_probas = drop_mm_probas

        if self._drop_mm_probas is not None:
            assert set(self._drop_mm_probas.keys()) == set(self._tdcps.keys()), \
                f'drop_mm_probas keys must match tdcps keys. Got {set(self._drop_mm_probas.keys())} and {set(self._tdcps.keys())}'

    @property
    def feature_names(self) -> Set[str]:
        return set(self._tdcps.keys())

    def _drop_mm_tokens(self, mm_track_features: List[torch.Tensor], mm_det_features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply token dropping to multimodal features during training.
        
        Args:
            mm_track_features: List of track feature tensors
            mm_det_features: List of detection feature tensors
            
        Returns:
            Tuple of (modified_track_features, modified_det_features)
        """
        # Skip dropping during evaluation
        if not self.training or self._drop_mm_probas is None:
            return mm_track_features, mm_det_features

        # Generate drop flags for each modality
        feature_names = list(self._tdcps.keys())
        drop_flags: List[bool] = []
        for feature_name in feature_names:
            flag = random.random() < self._drop_mm_probas[feature_name]
            drop_flags.append(flag)

        # edge-case: If all drop flags are True, return original features
        if all(drop_flags):
            return mm_track_features, mm_det_features

        # Apply dropping to each modality
        filtered_track_features = []
        filtered_det_features = []
        for i, (feature_name, track_feat, det_feat) in enumerate(zip(feature_names, mm_track_features, mm_det_features)):
            if not drop_flags[i]:
                filtered_track_features.append(track_feat)
                filtered_det_features.append(det_feat)
        
        return filtered_track_features, filtered_det_features 
        


    def forward(
        self,
        track_features: Dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert set(track_features.keys()) == set(det_features.keys()) == set(self._tdcps.keys())

        for key in self._tdcps:
            track_features[key], det_features[key] = self._tdcps[key](
                track_x=track_features[key],
                track_mask=track_mask,
                det_x=det_features[key],
                det_mask=det_mask
            )

        mm_track_features = [lin_layer(mm_feat) for lin_layer, mm_feat in zip(self._mm_linear_layers, list(track_features.values()))]
        mm_det_features = [lin_layer(mm_feat) for lin_layer, mm_feat in zip(self._mm_linear_layers, list(det_features.values()))]
        mm_track_features, mm_det_features = self._drop_mm_tokens(mm_track_features, mm_det_features)
        agg_track_features = self._aggregator(mm_track_features)
        agg_det_features = self._aggregator(mm_det_features)

        return agg_track_features, agg_det_features, track_features, det_features


def build_tdcp_model(
    feature_encoder_type: str = 'motion',
    feature_encoder_params: Dict[str, Any] = None,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    track_encoder_n_heads: int = 8,
    track_encoder_n_layers: int = 6,
    track_encoder_ffn_dim: int = 512,
    track_encoder_enable_motion_encoder: bool = True,
    projector_intermediate_dim: int = 512,
    interaction_encoder_enable: bool = False,
    interaction_encoder_n_heads: int = 8,
    interaction_encoder_n_layers: int = 6,
    interaction_encoder_ffn_dim: int = 512,
) -> TrackDetectionContrastivePrediction:
    """Build a complete TDCP model with default components.

    Args:
        feature_encoder_type: Feature encoder type
        feature_encoder_params: Feature encoder parameters
        hidden_dim: Shared embedding dimension across modules.
        dropout: Dropout rate used in all components.
        track_encoder_n_heads: Number of attention heads in the track encoder.
        track_encoder_n_layers: Number of transformer layers in the track encoder.
        track_encoder_ffn_dim: Feed-forward dimension in the track encoder.
        track_encoder_enable_motion_encoder: Enable motion encoder (use if FoD is applied in the transform function)
        projector_intermediate_dim: Hidden dimension of the projector MLP.
        interaction_encoder_enable: Whether to include the interaction encoder.
        interaction_encoder_n_heads: Attention heads for the interaction encoder.
        interaction_encoder_n_layers: Layers in the interaction encoder.
        interaction_encoder_ffn_dim: Feed-forward dimension for the interaction encoder.

    Returns:
        Instantiated :class:`TrackDetectionContrastivePrediction` model.
    """
    feature_encoder_params['hidden_dim'] = feature_encoder_params.get('hidden_dim', hidden_dim)
    feature_encoder = feature_encoder_factory(
        feature_encoder_type=feature_encoder_type,
        feature_encoder_params=feature_encoder_params
    )

    track_encoder = TrackEncoder(
        hidden_dim=hidden_dim,
        n_heads=track_encoder_n_heads,
        n_layers=track_encoder_n_layers,
        ffn_dim=track_encoder_ffn_dim,
        dropout=dropout,
    )

    projector = TrackToDetectionProjector(
        hidden_dim=hidden_dim,
        intermediate_hidden_dim=projector_intermediate_dim,
    )

    if interaction_encoder_enable:
        object_interaction_encoder = ObjectInteractionEncoder(
            hidden_dim=hidden_dim,
            n_heads=interaction_encoder_n_heads,
            n_layers=interaction_encoder_n_layers,
            ffn_dim=interaction_encoder_ffn_dim,
            dropout=dropout,
        )
    else:
        object_interaction_encoder = None

    return TrackDetectionContrastivePrediction(
        feature_encoder=feature_encoder,
        track_encoder=track_encoder,
        projector=projector,
        object_interaction_encoder=object_interaction_encoder,
        enable_motion_encoder=track_encoder_enable_motion_encoder
    )


def build_mm_tdcp_model(
    per_feature_params: Dict[str, Any],
    common_params: Dict[str, Any],
    mm_dim: int,
    aggregator_type: str,
    aggregator_params: Dict[str, Any],
    drop_mm_probas: Optional[Dict[str, float]] = None
) -> MultiModalTDCP:
    tdcps: Dict[str, TrackDetectionContrastivePrediction] = {}
    for feature_name in per_feature_params:
        params = tdcp_utils.merge_configs(common_params, per_feature_params[feature_name])
        tdcps[feature_name] = build_tdcp_model(**params)

    aggregator = tdcp_aggregator_factory(
        aggregator_type=aggregator_type,
        aggregator_params=aggregator_params,
        n_features=len(per_feature_params)
    )
    return MultiModalTDCP(
        tdcps=tdcps,
        aggregator=aggregator,
        mm_dim=mm_dim,
        drop_mm_probas=drop_mm_probas
    )


def run_test() -> None:
    tdcp = build_tdcp_model(
        input_dim=4,
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
