"""
End-to-end autoregressive trainer for TDLP.

Problem (exposure bias):
    Standard training feeds ground-truth track histories to the model, but during
    inference the tracker builds histories from its own (possibly wrong) association
    predictions.  When histories are short or error-prone (video start, objects that
    frequently appear/disappear), this distribution mismatch degrades performance.

Solution:
    This trainer simulates the inference loop during training.  Given a clip of T+1
    frames it processes them autoregressively:

    1. Frame 0 -- initialise one track per present detection.
    2. For each subsequent frame t = 1 .. T:
       a. Build a ``VideoClipData`` from the *current* track history and the
          frame-t detections, then apply the data transform (including FoD,
          which depends on the actual -- possibly wrong -- history).
       b. Run the model forward pass.
       c. Compute the association cost matrix via ``model.compute_cost_matrix``
          and solve with Hungarian matching.
       d. Update track histories with the matched detections.  Unmatched
          detections become new tracks; unmatched tracks keep their history.
    3. Return the averaged loss from the selected gradient frames.

    Because the transform has temporal dependencies (finite-order differences)
    and the association errors compound over time, the model learns to operate
    on the kind of noisy histories it actually encounters at inference.

Gradient frame selection:
    Not every autoregressive step needs to contribute gradients.  The
    ``n_gradient_frames`` parameter selects an evenly-spaced subset of steps
    (always including the last).  Steps outside this set run under
    ``torch.no_grad()``, which keeps VRAM usage at roughly
    ``n_gradient_frames x single_forward_pass``.

    Formula:  ``{total_steps * (i + 1) // n  for i in range(n)}``
    Examples (50 steps): n=1 -> {50}, n=2 -> {25, 50}, n=3 -> {17, 34, 50}

Identity supervision contract:
    After a wrong association a track's ``assigned_id`` may diverge from its
    original object.  The BCE loss at subsequent steps uses this *noisy* ID for
    supervision.  This is a deliberate approximation -- it mirrors the same
    noise the model sees during inference and teaches it to be robust to
    accumulated association errors.

Workflow:
    Pretrain with ``ContrastiveTrainer``, then fine-tune with
    ``EndToEndTrainer``.  The pretrained model's similarity threshold is reused
    as the ``sim_threshold`` for Hungarian matching during E2E training.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from motrack.tracker.matching.utils import hungarian

from tdlp.architectures.tdlp.base import TDLPModel
from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.transform import Transform
from tdlp.trainer.trainer import ContrastiveTrainer
from tdlp.trainer import torch_distrib_utils, torch_helper

logger = logging.getLogger('EndToEndTrainer')


@dataclass
class TrackState:
    """Mutable track-history state carried across autoregressive steps.

    All tensors share the same batch (B) and track-slot (N) dimensions.
    The history buffer has a fixed temporal length equal to the clip length T.
    New observations are appended at the *last* position; when the buffer is
    full the oldest entry is shifted out.

    Attributes:
        raw_features: ``{key: (B, N, T, F_key)}`` -- raw (pre-transform) features.
        mask: ``(B, N, T)`` -- ``True`` = missing / padded entry.
        ts: ``(B, N, T)`` -- frame timestamps for each history slot.
        assigned_ids: ``(B, N)`` -- the identity label currently assigned to each
            track (updated on every match; may diverge from ground truth after
            a wrong association -- see module docstring on identity supervision).
        active: ``(B, N)`` -- whether each track slot is in use.
    """
    raw_features: Dict[str, torch.Tensor]
    mask: torch.Tensor
    ts: torch.Tensor
    assigned_ids: torch.Tensor
    active: torch.Tensor


class EndToEndTrainer(ContrastiveTrainer):
    """Autoregressive end-to-end trainer for TDLP.

    Overrides :meth:`_forward_and_loss` to process frames sequentially,
    building track histories from the model's own association predictions
    (via Hungarian matching) rather than from ground-truth data.

    See the module docstring for a full description of the algorithm,
    gradient frame selection, and the identity supervision contract.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        n_epochs: int,
        tensorboard_log_dirpath: str,
        checkpoints_dirpath: str,
        transform: Transform,
        n_gradient_frames: int = 1,
        sim_threshold: float = 0.5,
        metric_monitor: str = 'val-epoch/loss',
        metric_monitor_minimize: bool = True,
        gradient_clip: Optional[float] = None,
        mixed_precision: bool = False,
        device: Optional[str] = None
    ):
        """
        Args:
            transform: Data transform to apply per autoregressive step.
                The dataset must be loaded with disable_transform=True.
            n_gradient_frames: Number of evenly-spaced frames that compute gradients.
                1 = only last frame, 2 = middle + last, etc.
            sim_threshold: Association threshold for Hungarian matching (from pretrained model).
        """
        super().__init__(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            tensorboard_log_dirpath=tensorboard_log_dirpath,
            checkpoints_dirpath=checkpoints_dirpath,
            metric_monitor=metric_monitor,
            metric_monitor_minimize=metric_monitor_minimize,
            gradient_clip=gradient_clip,
            mixed_precision=mixed_precision,
            device=device
        )
        self._e2e_transform = transform
        self._n_gradient_frames = n_gradient_frames
        self._sim_threshold = sim_threshold

    @staticmethod
    def _get_gradient_step_indices(total_steps: int, n_gradient_frames: int) -> Set[int]:
        """
        Compute evenly-spaced gradient step indices, always including the last step.

        Args:
            total_steps: Total number of autoregressive steps (1-indexed).
            n_gradient_frames: Desired number of gradient frames.

        Returns:
            Set of 1-indexed step indices where gradients should be computed.
        """
        n = min(n_gradient_frames, total_steps)
        return {total_steps * (i + 1) // n for i in range(n)}

    def _init_track_state(
        self,
        B: int,
        N: int,
        history_len: int,
        feature_shapes: Dict[str, int],
        device: torch.device,
    ) -> TrackState:
        """Initialize empty track state buffers."""
        raw_features = {
            key: torch.zeros(B, N, history_len, F, device=device)
            for key, F in feature_shapes.items()
        }
        return TrackState(
            raw_features=raw_features,
            mask=torch.ones(B, N, history_len, dtype=torch.bool, device=device),
            ts=torch.zeros(B, N, history_len, dtype=torch.long, device=device),
            assigned_ids=torch.full((B, N), -1, dtype=torch.long, device=device),
            active=torch.zeros(B, N, dtype=torch.bool, device=device),
        )

    def _initialize_tracks_from_detections(
        self,
        state: TrackState,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
        det_ids: torch.Tensor,
        det_ts: torch.Tensor,
    ) -> None:
        """
        Initialize tracks from frame-0 detections (vectorized).

        All present detections at frame 0 become tracks with 1-frame history
        placed at the last position of the history buffer.
        """
        present = ~det_mask  # (B, N) True where detection is present
        for key in state.raw_features:
            state.raw_features[key][:, :, -1, :][present] = det_features[key][present]
        state.mask[:, :, -1][present] = False
        state.ts[:, :, -1][present] = det_ts[present]
        state.assigned_ids[present] = det_ids[present]
        state.active[present] = True

    def _apply_transform_batched(
        self,
        state: TrackState,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
        det_ts: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply transform per batch sample, then re-stack.

        Clones raw features before transform to prevent in-place mutation of state buffers.

        Returns:
            (transformed_track_features, transformed_track_mask,
             transformed_det_features, transformed_det_mask)
        """
        B = state.mask.shape[0]
        track_features_list = []
        track_mask_list = []
        det_features_list = []
        det_mask_list = []

        for b in range(B):
            obs_features = {k: v[b].clone() for k, v in state.raw_features.items()}
            obs_mask = state.mask[b].clone()
            obs_ts = state.ts[b].clone()

            unobs_features = {k: v[b].clone() for k, v in det_features.items()}
            unobs_mask = det_mask[b].clone()
            unobs_ts = det_ts[b].clone()

            clip_data = VideoClipData(
                observed=VideoClipPart(
                    ids=None,
                    ts=obs_ts,
                    mask=obs_mask,
                    features=obs_features,
                ),
                unobserved=VideoClipPart(
                    ids=None,
                    ts=unobs_ts,
                    mask=unobs_mask,
                    features=unobs_features,
                ),
            )
            transformed = self._e2e_transform(clip_data)

            track_features_list.append(transformed.observed.features)
            track_mask_list.append(transformed.observed.mask)
            det_features_list.append(transformed.unobserved.features)
            det_mask_list.append(transformed.unobserved.mask)

        feature_keys = list(track_features_list[0].keys())
        batched_track_features = {
            k: torch.stack([t[k] for t in track_features_list])
            for k in feature_keys
        }
        batched_track_mask = torch.stack(track_mask_list)

        det_keys = list(det_features_list[0].keys())
        batched_det_features = {
            k: torch.stack([d[k] for d in det_features_list])
            for k in det_keys
        }
        batched_det_mask = torch.stack(det_mask_list)

        return batched_track_features, batched_track_mask, batched_det_features, batched_det_mask

    def _compute_associations(
        self,
        model_output: Any,
        state: TrackState,
        det_mask: torch.Tensor,
    ) -> List[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
        """
        Run Hungarian matching per batch element on model output.

        Returns:
            List (one per batch element) of (matches, unmatched_tracks, unmatched_dets).
            Indices are in padded (N-dim) space.
        """
        B = det_mask.shape[0]
        unwrapped: TDLPModel = torch_distrib_utils.get_model(self._model)
        cost_matrix_batch = unwrapped.compute_cost_matrix(
            model_output,
            n_tracks=det_mask.shape[1],
            n_dets=det_mask.shape[1],
        )

        results: List[Tuple[List[Tuple[int, int]], List[int], List[int]]] = []
        for b in range(B):
            active_indices = state.active[b].nonzero(as_tuple=True)[0].tolist()
            det_indices = (~det_mask[b]).nonzero(as_tuple=True)[0].tolist()

            n_active = len(active_indices)
            n_dets = len(det_indices)

            if n_active == 0:
                results.append(([], [], det_indices))
                continue
            if n_dets == 0:
                results.append(([], active_indices, []))
                continue

            sub_cost = cost_matrix_batch[b, active_indices][:, det_indices].cpu().numpy()
            sub_cost[sub_cost > self._sim_threshold] = np.inf
            sub_matches, sub_unmatched_tracks, sub_unmatched_dets = hungarian(sub_cost)

            matches = [(active_indices[t], det_indices[d]) for t, d in sub_matches]
            unmatched_tracks = [active_indices[t] for t in sub_unmatched_tracks]
            unmatched_dets = [det_indices[d] for d in sub_unmatched_dets]
            results.append((matches, unmatched_tracks, unmatched_dets))

        return results

    def _update_track_state(
        self,
        state: TrackState,
        associations: List[Tuple[List[Tuple[int, int]], List[int], List[int]]],
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
        det_ids: torch.Tensor,
        det_ts: torch.Tensor,
    ) -> None:
        """
        Update track state based on Hungarian matching results.

        - Matched tracks: shift history left, append matched detection at last position.
        - Unmatched detections: assign to first free track slot with 1-frame history.
        - Unmatched tracks: no change (persist with existing history).
        """
        B = state.mask.shape[0]

        for b in range(B):
            matches, _unmatched_tracks, unmatched_dets = associations[b]

            # Matched tracks: shift history and append new detection
            for t_idx, d_idx in matches:
                # Shift history left by 1
                for key in state.raw_features:
                    state.raw_features[key][b, t_idx, :-1] = state.raw_features[key][b, t_idx, 1:].clone()
                    state.raw_features[key][b, t_idx, -1] = det_features[key][b, d_idx].detach()
                state.mask[b, t_idx, :-1] = state.mask[b, t_idx, 1:].clone()
                state.mask[b, t_idx, -1] = False
                state.ts[b, t_idx, :-1] = state.ts[b, t_idx, 1:].clone()
                state.ts[b, t_idx, -1] = det_ts[b, d_idx]
                state.assigned_ids[b, t_idx] = det_ids[b, d_idx]

            # Unmatched detections: create new tracks in free slots
            free_slots = (~state.active[b]).nonzero(as_tuple=True)[0]
            slot_idx = 0
            for d_idx in unmatched_dets:
                if det_mask[b, d_idx]:
                    continue
                if slot_idx >= len(free_slots):
                    break
                slot = free_slots[slot_idx].item()
                slot_idx += 1

                for key in state.raw_features:
                    state.raw_features[key][b, slot] = 0
                    state.raw_features[key][b, slot, -1] = det_features[key][b, d_idx].detach()
                state.mask[b, slot] = True
                state.mask[b, slot, -1] = False
                state.ts[b, slot] = 0
                state.ts[b, slot, -1] = det_ts[b, d_idx]
                state.assigned_ids[b, slot] = det_ids[b, d_idx]
                state.active[b, slot] = True

    @staticmethod
    def _aggregate_loss_dicts(
        loss_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate loss dicts from multiple gradient steps."""
        if not loss_dicts:
            raise RuntimeError('No gradient frames produced a loss.')

        agg: Dict[str, Any] = {}
        n = len(loss_dicts)

        # Average scalar losses
        for key in loss_dicts[0]:
            if 'loss' in key:
                agg[key] = sum(d[key] for d in loss_dicts) / n
            elif key in ('track_labels', 'det_labels', 'track_predictions', 'det_predictions'):
                tensors = [d[key] for d in loss_dicts if d[key] is not None and d[key].numel() > 0]
                agg[key] = torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.long)
            elif key in ('track_mask', 'det_mask'):
                agg[key] = None
            else:
                agg[key] = loss_dicts[-1][key]

        return agg

    def _forward_and_loss(
        self,
        data: Dict[str, torch.Tensor],
        return_state: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive forward pass.

        Receives **raw** (untransformed) batched clip data and processes frames
        one-by-one, building track histories from predicted associations.

        Input shapes (batch dimension omitted for clarity)::

            observed.features  : {key: (N, T, F)}   -- raw features for T frames
            observed.mask      : (N, T)
            observed.ids       : (N, T)
            unobserved.features: {key: (N, F)}       -- detections at frame T
            unobserved.mask    : (N,)
            unobserved.ids     : (N,)

        The method merges these into T+1 frames, then:
        - Frame 0 initialises tracks.
        - Frames 1..T are processed autoregressively (see module docstring).
        - Loss is accumulated only at gradient frames and averaged.

        Args:
            data: Raw (untransformed) batched clip data from the DataLoader.
            return_state: If ``True``, return ``(loss_dict, final_state)``
                instead of just ``loss_dict``.  Useful for testing that the
                autoregressive history matches ground truth when associations
                are perfect.

        Returns:
            Loss dict compatible with the training loop (same keys as
            ``ContrastiveTrainer._forward_and_loss``).  If *return_state* is
            ``True``, returns a ``(loss_dict, TrackState)`` tuple.
        """
        B, N, T = data['observed']['mask'].shape
        device = data['observed']['mask'].device

        # 1. Merge observed + unobserved into all T+1 frames
        all_features: Dict[str, torch.Tensor] = {}
        for key in data['observed']['features']:
            obs = data['observed']['features'][key]       # (B, N, T, F)
            unobs = data['unobserved']['features'][key]   # (B, N, F)
            all_features[key] = torch.cat([obs, unobs.unsqueeze(2)], dim=2)

        all_mask = torch.cat([
            data['observed']['mask'],
            data['unobserved']['mask'].unsqueeze(2)
        ], dim=2)  # (B, N, T+1)

        all_ids = torch.cat([
            data['observed']['ids'],
            data['unobserved']['ids'].unsqueeze(2)
        ], dim=2)  # (B, N, T+1)

        all_ts = torch.cat([
            data['observed']['ts'],
            data['unobserved']['ts'].unsqueeze(2)
        ], dim=2)  # (B, N, T+1)

        total_steps = T  # steps 1..T (step 0 is initialization)

        # 2. Gradient step indices
        gradient_indices = self._get_gradient_step_indices(total_steps, self._n_gradient_frames)

        # 3. Initialize track state
        feature_shapes = {key: v.shape[-1] for key, v in all_features.items()}
        state = self._init_track_state(B, N, T, feature_shapes, device)

        # 4. Step 0: initialize tracks from frame-0 detections
        frame0_features = {key: all_features[key][:, :, 0, :] for key in all_features}
        frame0_mask = all_mask[:, :, 0]
        frame0_ids = all_ids[:, :, 0]
        frame0_ts = all_ts[:, :, 0]
        self._initialize_tracks_from_detections(
            state, frame0_features, frame0_mask, frame0_ids, frame0_ts
        )

        # 5. Autoregressive loop: steps 1..T
        gradient_loss_dicts: List[Dict[str, torch.Tensor]] = []

        for step_t in range(1, T + 1):
            # a. Extract frame-t detections
            det_features_t = {key: all_features[key][:, :, step_t, :] for key in all_features}
            det_mask_t = all_mask[:, :, step_t]
            det_ids_t = all_ids[:, :, step_t]
            det_ts_t = all_ts[:, :, step_t]

            # Skip if no active tracks (shouldn't happen after step 0)
            if not state.active.any():
                continue

            # b. Apply transform per batch sample
            trans_track_feat, trans_track_mask, trans_det_feat, trans_det_mask = \
                self._apply_transform_batched(state, det_features_t, det_mask_t, det_ts_t)

            # c. Determine grad context
            use_grad = (step_t in gradient_indices) and torch.is_grad_enabled()
            context = torch.enable_grad() if use_grad else torch.no_grad()

            # d. Model forward + loss
            with context:
                track_ids_for_loss = state.assigned_ids.unsqueeze(-1).expand_as(state.mask)

                with autocast(enabled=self._mixed_precision):
                    loss_dict, model_output = self._compute_model_output_and_loss(
                        trans_track_feat, trans_track_mask, track_ids_for_loss,
                        trans_det_feat, trans_det_mask, det_ids_t
                    )

                if use_grad:
                    gradient_loss_dicts.append(loss_dict)

            # e. Hungarian matching (always no grad)
            with torch.no_grad():
                associations = self._compute_associations(
                    model_output, state, det_mask_t
                )

            # f. Update track state
            self._update_track_state(
                state, associations,
                det_features_t, det_mask_t, det_ids_t, det_ts_t
            )

        # 6. Aggregate losses from gradient frames
        loss_dict = self._aggregate_loss_dicts(gradient_loss_dicts)
        if return_state:
            return loss_dict, state
        return loss_dict
