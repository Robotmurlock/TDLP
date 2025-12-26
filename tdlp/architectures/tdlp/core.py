"""Core TDCP model combining track and detection encoders."""

import copy
import logging
from typing import Any, Dict, Optional, Set, Tuple

from tdlp.architectures.tdlp import utils as tdcp_utils
from tdlp.architectures.tdlp.aggregators import TDCPAggregator, tdcp_aggregator_factory
from tdlp.architectures.tdlp.feature_encoders import feature_encoder_factory
from tdlp.architectures.tdlp.object_interaction_encoder import ObjectInteractionEncoder
from tdlp.architectures.tdlp.projector import TrackToDetectionProjector
from tdlp.architectures.tdlp.similarity_prediction import TDSPMLPHead
from tdlp.architectures.tdlp.track_encoder import TrackEncoder
import torch
from torch import nn

logger = logging.getLogger('Architecture')


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
                projected_features, 
                agg_track_mask, 
                det_features, 
                det_mask
            )

        return projected_features, det_features


class MultiModalTDCP(nn.Module):
    def __init__(
        self,
        tdcps: Dict[str, TrackDetectionContrastivePrediction],
        mm_dim: int,
        aggregator: TDCPAggregator,
        object_interaction_encoder: Optional[ObjectInteractionEncoder] = None
    ):
        super().__init__()
        self._mm_dim = mm_dim

        self._tdcps = nn.ModuleDict(tdcps)
        self._mm_linear_layers = nn.ModuleDict({
            feature_name: nn.Linear(tdcp.output_dim, mm_dim)
            for feature_name, tdcp in tdcps.items()
        })
        self._aggregator = aggregator
        self._object_interaction_encoder = object_interaction_encoder

    @property
    def output_dim(self) -> int:
        return self._mm_dim

    @property
    def feature_names(self) -> Set[str]:
        return set(self._tdcps.keys())

    def get_tdcp(self, feature_name: str) -> TrackDetectionContrastivePrediction:
        return self._tdcps[feature_name]

    def forward(
        self,
        track_features: Dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert set(track_features.keys()) == set(det_features.keys()) == set(self._tdcps.keys())

        for key in self._tdcps:
            track_features[key], det_features[key] = self._tdcps[key](
                track_x=track_features[key],
                track_mask=track_mask,
                det_x=det_features[key],
                det_mask=det_mask,
            )

        mm_track_features = {feature_name: self._mm_linear_layers[feature_name](track_features[feature_name]) for feature_name in track_features}
        mm_det_features = {feature_name: self._mm_linear_layers[feature_name](det_features[feature_name]) for feature_name in det_features}
        agg_track_features = self._aggregator(list(mm_track_features.values()))
        agg_det_features = self._aggregator(list(mm_det_features.values()))

        if self._object_interaction_encoder is not None:
            agg_track_mask = track_mask.all(dim=-1)
            agg_track_features, agg_det_features = self._object_interaction_encoder(
                agg_track_features, 
                agg_track_mask, 
                agg_det_features, 
                det_mask
            )

        return agg_track_features, agg_det_features, track_features, det_features


class TrackDetectionSimilarityPrediction(nn.Module):
    """Similarity model comparing track and detection embeddings."""

    def __init__(
        self,
        tdcp: TrackDetectionContrastivePrediction,
        similarity_prediction_head: TDSPMLPHead
    ) -> None:
        """Args:
            tdcp: Temporal encoder for track sequences.
            similarity_prediction_head: MLP head for similarity prediction.
        """
        super().__init__()
        self._tdcp = tdcp
        self._similarity_prediction_head = similarity_prediction_head

    def forward(
        self,
        track_x: torch.Tensor,
        track_mask: torch.Tensor,
        det_x: torch.Tensor,
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        track_features, det_features = self._tdcp(track_x, track_mask, det_x, det_mask)
        logits = self._similarity_prediction_head(track_features, det_features)
        return logits


class MultiModalTDSP(nn.Module):
    def __init__(
        self,
        mm_tdcp: MultiModalTDCP,
        sphs: Dict[str, TDSPMLPHead],
        mm_sph: TDSPMLPHead
    ) -> None:
        super().__init__()
        self._mm_tdcp = mm_tdcp
        self._sphs = nn.ModuleDict(sphs)
        self._mm_sph = mm_sph

    @property
    def feature_names(self) -> Set[str]:
        return set(self._sphs.keys())

    def forward(
        self,
        track_features: Dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert set(track_features.keys()) == set(det_features.keys()) == set(self._sphs.keys())

        agg_track_features, agg_det_features, track_features, det_features = self._mm_tdcp(
            track_features=track_features,
            track_mask=track_mask,
            det_features=det_features,
            det_mask=det_mask,
        )
        
        agg_logits = self._mm_sph(agg_track_features, agg_det_features)
        sphs_logits = {
            key: sph(track_features[key], det_features[key])
            for key, sph in self._sphs.items()
        }

        return agg_logits, sphs_logits

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
    per_feature_checkpoint: Optional[Dict[str, str]] = None,
    object_interaction_encoder_enable: bool = False,
    object_interaction_encoder_params: Optional[Dict[str, Any]] = None,
    tdcps_prefix: str = '_tdcps',
    mm_linear_layers_prefix: str = '_mm_linear_layers'
) -> MultiModalTDCP:
    per_feature_checkpoint = per_feature_checkpoint or {}
    state_dicts: Dict[str, Dict[str, Any]] = {
        feature_name: torch.load(per_feature_checkpoint[feature_name])['model']
        for feature_name in per_feature_params
        if feature_name in per_feature_checkpoint
    }

    tdcps: Dict[str, TrackDetectionContrastivePrediction] = {}
    for feature_name in per_feature_params:
        params = tdcp_utils.merge_configs(common_params, per_feature_params[feature_name])
        tdcps[feature_name] = build_tdcp_model(**params)
        if feature_name in per_feature_checkpoint:
            logger.info(f'Loading checkpoint for {feature_name} from {per_feature_checkpoint[feature_name]}')
            state_dict = state_dicts[feature_name]
            state_dict = {
                k.replace(f'{tdcps_prefix}.{feature_name}.', ''): v 
                for k, v in state_dict.items()
                if k.startswith(f'{tdcps_prefix}.{feature_name}.')
            }
            tdcps[feature_name].load_state_dict(state_dict)

    aggregator = tdcp_aggregator_factory(
        aggregator_type=aggregator_type,
        aggregator_params=aggregator_params,
        n_features=len(per_feature_params)
    )

    if object_interaction_encoder_enable:
        assert object_interaction_encoder_params is not None
        object_interaction_encoder = ObjectInteractionEncoder(**object_interaction_encoder_params)
    else:
        object_interaction_encoder = None

    mm_tdcp = MultiModalTDCP(
        tdcps=tdcps,
        aggregator=aggregator,
        mm_dim=mm_dim,
        object_interaction_encoder=object_interaction_encoder
    )

    for feature_name in per_feature_params:
        if feature_name in per_feature_checkpoint:
            state_dict = state_dicts[feature_name]
            state_dict = {
                k.replace(f'{mm_linear_layers_prefix}.{feature_name}.', ''): v 
                for k, v in state_dict.items()
                if k.startswith(f'{mm_linear_layers_prefix}.{feature_name}.')
            }
            mm_tdcp._mm_linear_layers[feature_name].load_state_dict(state_dict)

    return mm_tdcp




def build_tdsp_model(
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
    similarity_prediction_head_hidden_dim: int = 256,
    tdcps_prefix: str = '_tdcp._tdcps'
) -> TrackDetectionSimilarityPrediction:
    tdcp = build_tdcp_model(
        feature_encoder_type=feature_encoder_type,
        feature_encoder_params=feature_encoder_params,
        hidden_dim=hidden_dim,
        dropout=dropout,
        track_encoder_n_heads=track_encoder_n_heads,
        track_encoder_n_layers=track_encoder_n_layers,
        track_encoder_ffn_dim=track_encoder_ffn_dim,
        track_encoder_enable_motion_encoder=track_encoder_enable_motion_encoder,
        projector_intermediate_dim=projector_intermediate_dim,
        interaction_encoder_enable=interaction_encoder_enable,
        interaction_encoder_n_heads=interaction_encoder_n_heads,
        interaction_encoder_n_layers=interaction_encoder_n_layers,
        interaction_encoder_ffn_dim=interaction_encoder_ffn_dim,
        tdcps_prefix=tdcps_prefix
    )
    similarity_prediction_head = TDSPMLPHead(
        input_dim=hidden_dim,
        hidden_dim=similarity_prediction_head_hidden_dim,
    )
    return TrackDetectionSimilarityPrediction(
        tdcp=tdcp,
        similarity_prediction_head=similarity_prediction_head,
    )


def build_mm_tdsp_model(
    per_feature_params: Dict[str, Any],
    common_params: Dict[str, Any],
    sph_per_feature_params: Dict[str, Any],
    sph_common_params: Dict[str, Any],
    mm_dim: int,
    aggregator_type: str,
    aggregator_params: Dict[str, Any],
    similarity_prediction_head_hidden_dim: int = 256,
    object_interaction_encoder_enable: bool = False,
    object_interaction_encoder_params: Optional[Dict[str, Any]] = None,
    per_feature_checkpoint: Optional[Dict[str, str]] = None,
    tdcps_prefix: str = '_mm_tdcp._tdcps',
    tdcp_mm_linear_layers_prefix: str = '_mm_tdcp._mm_linear_layers'
) -> MultiModalTDSP:
    mm_tdcp = build_mm_tdcp_model(
        per_feature_params=per_feature_params,
        common_params=common_params,
        mm_dim=mm_dim,
        aggregator_type=aggregator_type,
        aggregator_params=aggregator_params,
        object_interaction_encoder_enable=object_interaction_encoder_enable,
        object_interaction_encoder_params=object_interaction_encoder_params,
        per_feature_checkpoint=per_feature_checkpoint,
        tdcps_prefix=tdcps_prefix,
        mm_linear_layers_prefix=tdcp_mm_linear_layers_prefix
    )
    sphs: Dict[str, TDSPMLPHead] = {}
    for feature_name in sph_per_feature_params:
        params = tdcp_utils.merge_configs(sph_common_params, sph_per_feature_params[feature_name])
        sphs[feature_name] = TDSPMLPHead(
            input_dim=mm_tdcp.get_tdcp(feature_name).output_dim,
            **params
        )
    if per_feature_checkpoint is not None:
        for key in per_feature_checkpoint:
            logger.info(f'Loading SPH checkpoint for {key} from {per_feature_checkpoint[key]}')
            state_dict = torch.load(per_feature_checkpoint[key])['model']
            state_dict = {
                k.replace(f'_sphs.{key}.', ''): v 
                for k, v in state_dict.items()
                if k.startswith(f'_sphs.{key}')
            }
            sphs[key].load_state_dict(state_dict)

    mm_sph = TDSPMLPHead(
        input_dim=mm_tdcp.output_dim,
        hidden_dim=similarity_prediction_head_hidden_dim,
    )
    return MultiModalTDSP(
        mm_tdcp=mm_tdcp,
        sphs=sphs,
        mm_sph=mm_sph,
    )


def run_test_tdcp() -> None:
    tdcp = build_tdcp_model(
        feature_encoder_type='motion',
        feature_encoder_params={
            'input_dim': 4
        },
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
    track_x, det_x = tdcp(track_features, track_mask, det_features, det_mask)
    expected_shape = (3, 4, 4)
    assert track_x.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {track_x.shape}.'
    assert det_x.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {det_x.shape}.'


def run_test_tdsp() -> None:
    tdsp = build_tdsp_model(
        feature_encoder_type='motion',
        feature_encoder_params={
            'input_dim': 4
        },
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
    logits = tdsp(track_features, track_mask, det_features, det_mask)
    expected_shape = (3, 4, 4, 1)
    assert logits.shape == expected_shape, f'Test failed! Expected shape {expected_shape} but got {logits.shape}.'


if __name__ == '__main__':
    run_test_tdcp()
    run_test_tdsp()
