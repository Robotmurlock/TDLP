"""Tests for the EndToEndTrainer."""
from unittest.mock import patch
import pytest
import torch

from tdlp.architectures.tdlp.core import build_mm_tdsp_model
from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.transform import ComposeTransform, IdentityTransform
from tdlp.datasets.dataset.transform.bbox import BBoxXYWHtoXYXY, FeatureFODStandardization
from tdlp.trainer.end_to_end_trainer import EndToEndTrainer, TrackState
from tdlp.trainer.losses.bce import ClipLevelBCE
from tdlp.trainer import torch_helper


# --- Helpers ---

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


def _build_fod_transform():
    """Build an FOD transform that converts 5D raw features to 10D (5 coords + 5 FoD)."""
    return ComposeTransform([
        FeatureFODStandardization(
            coord_mean={'bbox': [0.5, 0.5, 0.5, 0.5, 0.5]},
            coord_std={'bbox': [0.1, 0.1, 0.1, 0.1, 1.0]},
            fod_mean={'bbox': [0.0, 0.0, 0.0, 0.0, 0.0]},
            fod_std={'bbox': [0.05, 0.05, 0.05, 0.05, 1.0]},
            fod_time_scaled=False,
        )
    ])


def _build_e2e_trainer(model=None, loss_func=None, transform=None, n_gradient_frames=1):
    if model is None:
        model = _build_small_tdsp()
    if loss_func is None:
        loss_func = ClipLevelBCE()
    if transform is None:
        transform = _build_fod_transform()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    return EndToEndTrainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=1,
        tensorboard_log_dirpath='/tmp/test_e2e_tb',
        checkpoints_dirpath='/tmp/test_e2e_ckpt',
        device='cpu',
        transform=transform,
        n_gradient_frames=n_gradient_frames,
        sim_threshold=0.9,
    )


def _build_raw_batch(B=2, N=4, T=5, F=5):
    """Build a synthetic raw (untransformed) batch.

    All N objects present at all T+1 frames with unique IDs 0..N-1.
    """
    return {
        'observed': {
            'features': {'bbox': torch.rand(B, N, T, F)},
            'mask': torch.zeros(B, N, T, dtype=torch.bool),
            'ids': torch.arange(N).unsqueeze(0).unsqueeze(-1).expand(B, N, T).clone(),
            'ts': torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, N, T).clone(),
        },
        'unobserved': {
            'features': {'bbox': torch.rand(B, N, F)},
            'mask': torch.zeros(B, N, dtype=torch.bool),
            'ids': torch.arange(N).unsqueeze(0).expand(B, N).clone(),
            'ts': torch.full((B, N), T, dtype=torch.long),
        }
    }


def _build_raw_batch_with_masking(B=2, N=4, T=5, F=5):
    """Build batch where some objects appear/disappear mid-clip.

    Object 0: present at all frames
    Object 1: appears at frame 2
    Object 2: present at all frames
    Object 3: masked at all frames (padding)
    """
    batch = _build_raw_batch(B, N, T, F)
    # Object 1 absent before frame 2
    batch['observed']['mask'][:, 1, :2] = True
    batch['observed']['features']['bbox'][:, 1, :2, :] = 0
    # Object 3 fully masked (padding)
    batch['observed']['mask'][:, 3, :] = True
    batch['observed']['features']['bbox'][:, 3, :, :] = 0
    batch['unobserved']['mask'][:, 3] = True
    batch['unobserved']['features']['bbox'][:, 3, :] = 0
    batch['unobserved']['ids'][:, 3] = -1
    batch['observed']['ids'][:, 3, :] = -1
    return batch


# --- Gradient Step Index Tests ---

class TestGradientStepIndices:
    def test_single(self):
        indices = EndToEndTrainer._get_gradient_step_indices(total_steps=10, n_gradient_frames=1)
        assert indices == {10}

    def test_two(self):
        indices = EndToEndTrainer._get_gradient_step_indices(total_steps=10, n_gradient_frames=2)
        assert 10 in indices
        assert len(indices) == 2

    def test_three(self):
        indices = EndToEndTrainer._get_gradient_step_indices(total_steps=9, n_gradient_frames=3)
        assert indices == {3, 6, 9}

    def test_all_when_n_exceeds_total(self):
        indices = EndToEndTrainer._get_gradient_step_indices(total_steps=5, n_gradient_frames=10)
        assert indices == {1, 2, 3, 4, 5}

    def test_always_includes_last(self):
        for total in [3, 5, 10, 50]:
            for n in [1, 2, 3, 5]:
                indices = EndToEndTrainer._get_gradient_step_indices(total, n)
                assert total in indices


# --- Transform Application Tests ---

class TestApplyTransformBatched:
    def test_preserves_raw_features(self):
        """Transform should not modify the raw feature buffers (clone boundary)."""
        transform = ComposeTransform([BBoxXYWHtoXYXY()])
        trainer = _build_e2e_trainer(transform=transform)

        state = trainer._init_track_state(B=2, N=3, history_len=4, feature_shapes={'bbox': 5}, device='cpu')
        state.raw_features['bbox'][:] = torch.rand_like(state.raw_features['bbox'])
        state.mask[:, :, -1] = False  # last slot has data
        raw_copy = state.raw_features['bbox'].clone()

        det_features = {'bbox': torch.rand(2, 3, 5)}
        det_mask = torch.zeros(2, 3, dtype=torch.bool)
        det_ts = torch.ones(2, 3, dtype=torch.long)

        trainer._apply_transform_batched(state, det_features, det_mask, det_ts)

        assert torch.equal(state.raw_features['bbox'], raw_copy), 'Raw features mutated by transform'


# --- Forward and Loss Integration Tests ---

class TestForwardAndLoss:
    def test_returns_correct_keys(self):
        trainer = _build_e2e_trainer(n_gradient_frames=1)
        trainer._on_start()

        data = _build_raw_batch(B=2, N=4, T=5, F=5)
        data = torch_helper.to_device(data, device=trainer._device)
        loss_dict = trainer._forward_and_loss(data)

        required_keys = {'loss', 'track_loss', 'det_loss', 'track_labels', 'det_labels',
                         'track_predictions', 'det_predictions', 'track_mask', 'det_mask'}
        assert required_keys.issubset(set(loss_dict.keys()))

    def test_loss_is_scalar(self):
        trainer = _build_e2e_trainer(n_gradient_frames=1)
        trainer._on_start()

        data = _build_raw_batch(B=2, N=4, T=5, F=5)
        data = torch_helper.to_device(data, device=trainer._device)
        loss_dict = trainer._forward_and_loss(data)

        assert loss_dict['loss'].dim() == 0

    def test_loss_has_grad(self):
        trainer = _build_e2e_trainer(n_gradient_frames=1)
        trainer._on_start()

        data = _build_raw_batch(B=1, N=3, T=3, F=5)
        data = torch_helper.to_device(data, device=trainer._device)
        loss_dict = trainer._forward_and_loss(data)

        assert loss_dict['loss'].requires_grad

    def test_with_masking_no_nan(self):
        trainer = _build_e2e_trainer(n_gradient_frames=1)
        trainer._on_start()

        data = _build_raw_batch_with_masking(B=2, N=4, T=5, F=5)
        data = torch_helper.to_device(data, device=trainer._device)
        loss_dict = trainer._forward_and_loss(data)

        assert not torch.isnan(loss_dict['loss'])
        assert not torch.isinf(loss_dict['loss'])

    def test_backward_succeeds_all_gradient_frames(self):
        for n_grad in [1, 2, 3]:
            model = _build_small_tdsp()
            trainer = _build_e2e_trainer(model=model, n_gradient_frames=n_grad)
            trainer._on_start()

            data = _build_raw_batch(B=1, N=3, T=5, F=5)
            data = torch_helper.to_device(data, device=trainer._device)
            loss_dict = trainer._forward_and_loss(data)
            loss_dict['loss'].backward()

            has_grad = any(p.grad is not None for p in model.parameters())
            assert has_grad, f'No gradients with n_gradient_frames={n_grad}'
            trainer._optimizer.zero_grad()

    def test_loss_is_finite(self):
        trainer = _build_e2e_trainer(n_gradient_frames=2)
        trainer._on_start()

        data = _build_raw_batch(B=2, N=4, T=5, F=5)
        data = torch_helper.to_device(data, device=trainer._device)
        loss_dict = trainer._forward_and_loss(data)

        assert torch.isfinite(loss_dict['loss'])


# --- Track State Tests ---

class TestTrackInitialization:
    def test_frame0_init(self):
        trainer = _build_e2e_trainer()
        state = trainer._init_track_state(B=1, N=3, history_len=5, feature_shapes={'bbox': 5}, device='cpu')

        det_features = {'bbox': torch.rand(1, 3, 5)}
        det_mask = torch.tensor([[False, False, True]])  # obj 0,1 present; obj 2 absent
        det_ids = torch.tensor([[10, 20, -1]])
        det_ts = torch.tensor([[0, 0, 0]])

        trainer._initialize_tracks_from_detections(state, det_features, det_mask, det_ids, det_ts)

        assert state.active[0, 0].item() is True
        assert state.active[0, 1].item() is True
        assert state.active[0, 2].item() is False
        assert state.assigned_ids[0, 0].item() == 10
        assert state.assigned_ids[0, 1].item() == 20
        # Data placed at last history position
        assert state.mask[0, 0, -1].item() is False
        assert state.mask[0, 0, 0].item() is True  # earlier positions still masked


class TestTrackStateUpdate:
    def test_matched_appends_features(self):
        trainer = _build_e2e_trainer()
        state = trainer._init_track_state(B=1, N=3, history_len=5, feature_shapes={'bbox': 5}, device='cpu')

        # Setup: track 0 active with 1 frame of history
        state.active[0, 0] = True
        state.raw_features['bbox'][0, 0, -1] = torch.ones(5)
        state.mask[0, 0, -1] = False
        state.assigned_ids[0, 0] = 10

        det_features = {'bbox': torch.tensor([[[2.0, 2.0, 2.0, 2.0, 2.0],
                                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]]])}
        det_mask = torch.tensor([[False, False, True]])
        det_ids = torch.tensor([[10, 20, -1]])
        det_ts = torch.tensor([[1, 1, 1]])

        associations = [
            ([(0, 0)], [], [1]),  # track 0 matched to det 0; det 1 unmatched
        ]

        trainer._update_track_state(state, associations, det_features, det_mask, det_ids, det_ts)

        # Track 0 should now have det 0's features at last position
        assert torch.allclose(state.raw_features['bbox'][0, 0, -1], torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]))
        assert state.assigned_ids[0, 0].item() == 10
        # Original data should have shifted left
        assert state.mask[0, 0, -2].item() is False

    def test_unmatched_det_creates_new_track(self):
        trainer = _build_e2e_trainer()
        state = trainer._init_track_state(B=1, N=4, history_len=5, feature_shapes={'bbox': 5}, device='cpu')

        state.active[0, 0] = True  # only track 0 active
        state.assigned_ids[0, 0] = 10

        det_features = {'bbox': torch.tensor([[[0.0] * 5, [7.0, 7.0, 7.0, 7.0, 7.0], [0.0] * 5, [0.0] * 5]])}
        det_mask = torch.tensor([[True, False, True, True]])  # only det 1 present
        det_ids = torch.tensor([[-1, 20, -1, -1]])
        det_ts = torch.tensor([[1, 1, 1, 1]])

        associations = [
            ([], [0], [1]),  # track 0 unmatched; det 1 unmatched
        ]

        trainer._update_track_state(state, associations, det_features, det_mask, det_ids, det_ts)

        # Det 1 should be assigned to first free slot (slot 1)
        assert state.active[0, 1].item() is True
        assert state.assigned_ids[0, 1].item() == 20
        assert torch.allclose(state.raw_features['bbox'][0, 1, -1], torch.tensor([7.0, 7.0, 7.0, 7.0, 7.0]))
        assert state.mask[0, 1, -1].item() is False
        assert state.mask[0, 1, 0].item() is True  # rest of history masked


# --- Aggregate Loss Dict Tests ---

class TestAggregateLossDicts:
    def test_averages_losses(self):
        d1 = {
            'loss': torch.tensor(2.0),
            'track_loss': torch.tensor(1.0),
            'det_loss': torch.tensor(1.0),
            'track_labels': torch.tensor([0, 1]),
            'det_labels': torch.tensor([0, 1]),
            'track_predictions': torch.tensor([0, 0]),
            'det_predictions': torch.tensor([0, 0]),
            'track_mask': None,
            'det_mask': None,
        }
        d2 = {
            'loss': torch.tensor(4.0),
            'track_loss': torch.tensor(3.0),
            'det_loss': torch.tensor(1.0),
            'track_labels': torch.tensor([2, 3]),
            'det_labels': torch.tensor([2, 3]),
            'track_predictions': torch.tensor([2, 2]),
            'det_predictions': torch.tensor([2, 2]),
            'track_mask': None,
            'det_mask': None,
        }

        agg = EndToEndTrainer._aggregate_loss_dicts([d1, d2])

        assert torch.isclose(agg['loss'], torch.tensor(3.0))
        assert torch.isclose(agg['track_loss'], torch.tensor(2.0))
        assert len(agg['track_labels']) == 4
        assert len(agg['track_predictions']) == 4

    def test_empty_raises(self):
        with pytest.raises(RuntimeError):
            EndToEndTrainer._aggregate_loss_dicts([])


# --- GT-association correctness test ---

class TestGTAssociationCorrectness:
    """When every Hungarian match is correct (GT), the E2E track history at the
    last frame must equal the GT per-frame features in order.

    Setup: B=1, N=3 objects, T=4 observed frames + 1 unobserved (5 total).
    All objects present at all frames with unique IDs.
    sim_threshold=1.0 so no matches are rejected.

    After autoregressive processing (5 steps: init + 4 association steps):
    - Frame 0 initialises tracks (placed at history[-1]).
    - Frames 1..4 shift history left and append at history[-1].
    - With T=4 history slots, after 4 association steps the buffer contains
      exactly frames [1, 2, 3, 4] (frame 0 was shifted out).
    """

    def _gt_associations(self, model_output, state, det_mask):
        """Return identity matches: track i <-> detection i for all present pairs."""
        B = det_mask.shape[0]
        results = []
        for b in range(B):
            active = state.active[b].nonzero(as_tuple=True)[0].tolist()
            present = (~det_mask[b]).nonzero(as_tuple=True)[0].tolist()
            matched_indices = set(active) & set(present)
            matches = [(i, i) for i in sorted(matched_indices)]
            unmatched_tracks = [i for i in active if i not in matched_indices]
            unmatched_dets = [i for i in present if i not in matched_indices]
            results.append((matches, unmatched_tracks, unmatched_dets))
        return results

    def test_history_matches_gt_when_associations_are_perfect(self):
        B, N, T, F = 1, 3, 4, 5
        torch.manual_seed(42)

        # Build raw batch: all objects present everywhere, unique IDs
        raw_features = torch.rand(B, N, T + 1, F)
        data = {
            'observed': {
                'features': {'bbox': raw_features[:, :, :T, :]},
                'mask': torch.zeros(B, N, T, dtype=torch.bool),
                'ids': torch.arange(N).unsqueeze(0).unsqueeze(-1).expand(B, N, T).clone(),
                'ts': torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, N, T).clone(),
            },
            'unobserved': {
                'features': {'bbox': raw_features[:, :, T, :]},
                'mask': torch.zeros(B, N, dtype=torch.bool),
                'ids': torch.arange(N).unsqueeze(0).expand(B, N).clone(),
                'ts': torch.full((B, N), T, dtype=torch.long),
            }
        }

        trainer = _build_e2e_trainer(n_gradient_frames=1)
        trainer._on_start()

        data = torch_helper.to_device(data, device=trainer._device)

        # Patch _compute_associations to return GT (identity) matches
        with patch.object(trainer, '_compute_associations', side_effect=self._gt_associations):
            _, state = trainer._forward_and_loss(data, return_state=True)

        # After T association steps with T history slots, history contains
        # frames [1, 2, 3, 4] (frame 0 shifted out).
        expected = raw_features[:, :, 1:, :]  # (B, N, T, F) = frames 1..T
        actual = state.raw_features['bbox']

        for i in range(N):
            assert state.active[0, i].item() is True, f'Track {i} should be active'
            assert state.assigned_ids[0, i].item() == i, f'Track {i} ID should be {i}'
            assert torch.allclose(actual[0, i], expected[0, i], atol=1e-6), \
                f'Track {i} history does not match GT frames [1..{T}]'
            assert not state.mask[0, i].any(), f'Track {i} should have no masked slots'
