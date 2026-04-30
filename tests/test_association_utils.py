"""Tests for model.compute_cost_matrix (TDLPModel interface)."""
import torch
from torch.nn import functional as F

from tdlp.architectures.tdlp.core import build_mm_tdsp_model, build_mm_tdcp_model


def _build_small_tdsp():
    return build_mm_tdsp_model(
        per_feature_params={'bbox': {
            'feature_encoder_type': 'motion',
            'feature_encoder_params': {'input_dim': 5},
        }},
        common_params={
            'hidden_dim': 16, 'dropout': 0.0,
            'track_encoder_n_heads': 2, 'track_encoder_n_layers': 1,
            'track_encoder_ffn_dim': 32, 'projector_intermediate_dim': 16,
        },
        sph_per_feature_params={'bbox': {'hidden_dim': 16}},
        sph_common_params={'hidden_dim': 16},
        mm_dim=16,
        similarity_prediction_head_hidden_dim=16,
        similarity_head_type='compact_mlp',
        aggregator_type='sum',
        aggregator_params={},
    )


def _build_small_tdcp():
    return build_mm_tdcp_model(
        per_feature_params={'bbox': {
            'feature_encoder_type': 'motion',
            'feature_encoder_params': {'input_dim': 5},
        }},
        common_params={
            'hidden_dim': 16, 'dropout': 0.0,
            'track_encoder_n_heads': 2, 'track_encoder_n_layers': 1,
            'track_encoder_ffn_dim': 32, 'projector_intermediate_dim': 16,
        },
        mm_dim=16,
        aggregator_type='sum',
        aggregator_params={},
    )


class TestComputeCostMatrixTDSP:
    def test_returns_correct_shape(self):
        model = _build_small_tdsp()
        logits = torch.randn(2, 5, 5)
        cost = model.compute_cost_matrix((logits, None), n_tracks=3, n_dets=4)
        assert cost.shape == (2, 3, 4)

    def test_values_in_zero_one(self):
        model = _build_small_tdsp()
        logits = torch.randn(2, 5, 5)
        cost = model.compute_cost_matrix((logits, None), n_tracks=5, n_dets=5)
        assert (cost >= 0).all() and (cost <= 1).all()

    def test_high_logits_give_low_cost(self):
        model = _build_small_tdsp()
        logits = torch.full((1, 3, 3), 10.0)
        cost = model.compute_cost_matrix((logits, None), n_tracks=3, n_dets=3)
        assert (cost < 0.01).all()


class TestComputeCostMatrixTDCP:
    def test_returns_correct_shape(self):
        model = _build_small_tdcp()
        track_feat = torch.randn(2, 5, 16)
        det_feat = torch.randn(2, 5, 16)
        cost = model.compute_cost_matrix((track_feat, det_feat, None, None), n_tracks=3, n_dets=4)
        assert cost.shape == (2, 3, 4)

    def test_values_in_zero_one(self):
        model = _build_small_tdcp()
        track_feat = torch.randn(2, 5, 16)
        det_feat = torch.randn(2, 5, 16)
        cost = model.compute_cost_matrix((track_feat, det_feat, None, None), n_tracks=5, n_dets=5)
        assert (cost >= 0).all() and (cost <= 1).all()

    def test_identical_embeddings_zero_cost(self):
        model = _build_small_tdcp()
        feat = F.normalize(torch.randn(1, 3, 16), dim=-1)
        cost = model.compute_cost_matrix((feat, feat, None, None), n_tracks=3, n_dets=3)
        assert torch.allclose(cost.diagonal(dim1=1, dim2=2), torch.zeros(1, 3), atol=1e-5)
