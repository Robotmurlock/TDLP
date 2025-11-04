from typing import Dict, Optional, List

import torch
from pytorch_metric_learning import losses, distances, reducers
from torch.nn import functional as F

from mot_jepa.trainer.losses.base import VideoClipLoss


def torch_combine(xs: List[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """
    Concatenate tensors or return empty tensor if list is empty.

    Args:
        xs: List of tensors to concatenate
        dtype: Data type for empty tensor when xs is empty

    Returns:
        Concatenated tensor or empty tensor with specified dtype
    """
    if len(xs) > 0:
        return torch.cat(xs)
    return torch.empty(0, dtype=dtype)


class ClipLevelInfoNCE(VideoClipLoss):
    """
    InfoNCE loss computed separately for each clip in the batch.

    Applies InfoNCE to tracks and detections within each clip as positive pairs.
    Uses cosine similarity and averages over non-zero valid pairs per clip.
    """
    def __init__(self):
        super().__init__()
        self._loss_func = losses.NTXentLoss(distance=distances.CosineSimilarity(), reducer=reducers.AvgNonZeroReducer())

    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        det_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_x: Tensor of shape (B, N, E)
            det_x: Tensor of shape (B, N, E)
            track_mask: Tensor of shape (B, N, T), 1 indicates missing, 0 indicates present
            detection_mask: Tensor of shape (B, N), 1 indicates missing, 0 indicates present
        Returns:
            Dictionary containing loss and additional debug information
        """
        B, N, E = track_x.shape
        agg_track_mask = track_mask.all(dim=-1)  # shape: (B, N), True indicates missing
        track_labels = torch.arange(N).to(track_x).unsqueeze(0).repeat(B, 1).long()
        det_labels = torch.arange(N).to(det_x).unsqueeze(0).repeat(B, 1).long()
        losses = []
        for b_i in range(B):
            # extract non-masked labels
            sub_track_labels = track_labels[b_i][~agg_track_mask[b_i]]
            sub_det_labels = det_labels[b_i][~detection_mask[b_i]]
            sub_labels = torch.cat([sub_track_labels, sub_det_labels], dim=0)

            # extract non-masked embeddings
            sub_track_x = track_x[b_i][~agg_track_mask[b_i]]
            sub_det_x = det_x[b_i][~detection_mask[b_i]]
            sub_embeddings = torch.cat([sub_track_x, sub_det_x], dim=0)

            # Calculate losses
            sub_loss = self._loss_func(sub_embeddings, sub_labels)
            losses.append(sub_loss)

        # Labels and predictions are still calculated at clip level
        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []
        with torch.no_grad():
            for b_i in range(B):
                combined_mask = ~agg_track_mask[b_i] & ~detection_mask[b_i]
                if not bool(combined_mask.any().item()):
                    continue

                sub_track_labels = track_labels[b_i][combined_mask]
                sub_det_labels = det_labels[b_i][combined_mask]
                sub_track_x = track_x[b_i][combined_mask]
                sub_det_x = det_x[b_i][combined_mask]
                sub_track_x = F.normalize(sub_track_x, dim=-1)
                sub_det_x = F.normalize(sub_det_x, dim=-1)

                n_sub_tracks = sub_track_x.shape[0]
                n_sub_det = sub_det_x.shape[0]
                if n_sub_tracks > 0 and n_sub_det > 0:
                    distances = sub_track_x @ sub_det_x.T
                    sub_track_predictions = sub_track_labels[torch.argmax(distances, dim=1)]
                    sub_det_predictions = sub_det_labels[torch.argmax(distances, dim=0)]

                    filtered_track_labels_list.append(sub_track_labels)
                    filtered_det_labels_list.append(sub_det_labels)
                    track_predictions_list.append(sub_track_predictions)
                    det_predictions_list.append(sub_det_predictions)

        filtered_track_labels = torch_combine(filtered_track_labels_list, dtype=torch.long)
        filtered_det_labels_list = torch_combine(filtered_det_labels_list, dtype=torch.long)
        track_predictions = torch_combine(track_predictions_list, dtype=torch.long)
        det_predictions = torch_combine(det_predictions_list, dtype=torch.long)

        loss = torch.stack(losses).mean()
        return {
            'loss': loss,
            'track_loss': loss,
            'det_loss': loss,
            'track_labels': filtered_track_labels,
            'det_labels': filtered_det_labels_list,
            'track_predictions': track_predictions,
            'det_predictions': det_predictions,
            'track_mask': None,
            'det_mask': None
        }


class BatchLevelInfoNCE(VideoClipLoss):
    """
    InfoNCE loss computed across the entire batch.

    Allows cross-clip matching by treating all tracks and detections as potential
    positive pairs. Uses cosine similarity and averages over non-zero valid pairs.
    """
    def __init__(self):
        super().__init__()
        self._loss_func = losses.NTXentLoss(distance=distances.CosineSimilarity(), reducer=reducers.AvgNonZeroReducer())

    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        det_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_x: Tensor of shape (B, N, E)
            det_x: Tensor of shape (B, N, E)
            track_mask: Tensor of shape (B, N, T), 1 indicates missing, 0 indicates present
            detection_mask: Tensor of shape (B, N), 1 indicates missing, 0 indicates present
        Returns:
            Dictionary containing loss and additional debug information
        """
        B, N, E = track_x.shape
        agg_track_mask = track_mask.all(dim=-1)  # shape: (B, N), True indicates missing
        global_track_labels = torch.arange(B * N).to(track_x).long()
        global_det_labels = torch.arange(B * N).to(det_x).long()

        flatten_agg_track_mask = agg_track_mask.view(-1)
        flatten_track_x = track_x.view(B * N, -1)
        flatten_det_x = det_x.view(B * N, -1)
        flatten_detection_mask = detection_mask.view(-1)

        global_track_labels = global_track_labels[~flatten_agg_track_mask]
        global_det_labels = global_det_labels[~flatten_detection_mask]
        labels = torch.cat([global_track_labels, global_det_labels], dim=0)
        filtered_track_x = flatten_track_x[~flatten_agg_track_mask]
        filtered_det_x = flatten_det_x[~flatten_detection_mask]
        embeddings = torch.cat([filtered_track_x, filtered_det_x], dim=0)
        loss = self._loss_func(embeddings, labels)

        # Labels and predictions are still calculated at clip level
        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []
        with torch.no_grad():
            track_labels = torch.arange(N).to(track_x).unsqueeze(0).repeat(B, 1).long()
            det_labels = torch.arange(N).to(det_x).unsqueeze(0).repeat(B, 1).long()
            for b_i in range(B):
                combined_mask = ~agg_track_mask[b_i] & ~detection_mask[b_i]
                if not bool(combined_mask.any().item()):
                    continue

                sub_track_labels = track_labels[b_i][combined_mask]
                sub_det_labels = det_labels[b_i][combined_mask]
                sub_track_x = track_x[b_i][combined_mask]
                sub_det_x = det_x[b_i][combined_mask]
                sub_track_x = F.normalize(sub_track_x, dim=-1)
                sub_det_x = F.normalize(sub_det_x, dim=-1)

                n_sub_tracks = sub_track_x.shape[0]
                n_sub_det = sub_det_x.shape[0]
                if n_sub_tracks > 0 and n_sub_det > 0:
                    distances = sub_track_x @ sub_det_x.T
                    sub_track_predictions = sub_track_labels[torch.argmax(distances, dim=1)]
                    sub_det_predictions = sub_det_labels[torch.argmax(distances, dim=0)]

                    filtered_track_labels_list.append(sub_track_labels)
                    filtered_det_labels_list.append(sub_det_labels)
                    track_predictions_list.append(sub_track_predictions)
                    det_predictions_list.append(sub_det_predictions)

        filtered_track_labels = torch_combine(filtered_track_labels_list, dtype=torch.long)
        filtered_det_labels_list = torch_combine(filtered_det_labels_list, dtype=torch.long)
        track_predictions = torch_combine(track_predictions_list, dtype=torch.long)
        det_predictions = torch_combine(det_predictions_list, dtype=torch.long)

        return {
            'loss': loss,
            'track_loss': loss,
            'det_loss': loss,
            'track_labels': filtered_track_labels,
            'det_labels': filtered_det_labels_list,
            'track_predictions': track_predictions,
            'det_predictions': det_predictions,
            'track_mask': None,
            'det_mask': None,
        }


class IDLevelInfoNCE(VideoClipLoss):
    """
    InfoNCE loss using object IDs to determine positive pairs.

    Uses explicit identity information rather than spatial/temporal correspondence.
    Requires track_ids and det_ids parameters. Uses cosine similarity.
    """
    def __init__(self):
        super().__init__()
        self._loss_func = losses.NTXentLoss(distance=distances.CosineSimilarity(), reducer=reducers.AvgNonZeroReducer())

    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        det_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss using object IDs for positive pairs.

        Args:
            track_x: Track embeddings (B, N, E)
            det_x: Detection embeddings (B, N, E)
            track_mask: Track mask (B, N, T), True=missing
            detection_mask: Detection mask (B, N), True=missing
            track_feature_dict: Optional modality-specific track features
            det_feature_dict: Optional modality-specific detection features
            track_ids: Track identifiers (B, N) - required
            det_ids: Detection identifiers (B, N) - required

        Returns:
            Dictionary with loss, predictions, and evaluation metrics

        Raises:
            ValueError: If track_ids or det_ids not provided
        """
        if track_ids is None or det_ids is None:
            raise ValueError('IDLevelInfoNCE requires track_ids and det_ids.')

        B, N, E = track_x.shape
        agg_track_mask = track_mask.all(dim=-1)
        agg_track_ids = track_ids.max(dim=-1).values

        flatten_track_mask = agg_track_mask.view(-1)
        flatten_det_mask = detection_mask.view(-1)
        flatten_track_x = track_x.view(B * N, -1)
        flatten_det_x = det_x.view(B * N, -1)

        track_id_flat = agg_track_ids.view(-1)[~flatten_track_mask]
        det_id_flat = det_ids.view(-1)[~flatten_det_mask]
        labels = torch.cat([track_id_flat, det_id_flat], dim=0)
        embeddings = torch.cat([
            flatten_track_x[~flatten_track_mask],
            flatten_det_x[~flatten_det_mask]
        ], dim=0)
        loss = self._loss_func(embeddings, labels)

        # Accuracy metrics (clip level like BatchLevelInfoNCE)
        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []
        with torch.no_grad():
            track_labels = torch.arange(N).to(track_x).unsqueeze(0).repeat(B, 1).long()
            det_labels = torch.arange(N).to(det_x).unsqueeze(0).repeat(B, 1).long()
            for b_i in range(B):
                combined_mask = ~agg_track_mask[b_i] & ~detection_mask[b_i]
                if not bool(combined_mask.any().item()):
                    continue

                sub_track_labels = track_labels[b_i][combined_mask]
                sub_det_labels = det_labels[b_i][combined_mask]
                sub_track_x = track_x[b_i][combined_mask]
                sub_det_x = det_x[b_i][combined_mask]
                sub_track_x = F.normalize(sub_track_x, dim=-1)
                sub_det_x = F.normalize(sub_det_x, dim=-1)

                n_sub_tracks = sub_track_x.shape[0]
                n_sub_det = sub_det_x.shape[0]
                if n_sub_tracks > 0 and n_sub_det > 0:
                    distances = sub_track_x @ sub_det_x.T
                    sub_track_predictions = sub_track_labels[torch.argmax(distances, dim=1)]
                    sub_det_predictions = sub_det_labels[torch.argmax(distances, dim=0)]

                    filtered_track_labels_list.append(sub_track_labels)
                    filtered_det_labels_list.append(sub_det_labels)
                    track_predictions_list.append(sub_track_predictions)
                    det_predictions_list.append(sub_det_predictions)

        filtered_track_labels = torch_combine(filtered_track_labels_list, dtype=torch.long)
        filtered_det_labels_list = torch_combine(filtered_det_labels_list, dtype=torch.long)
        track_predictions = torch_combine(track_predictions_list, dtype=torch.long)
        det_predictions = torch_combine(det_predictions_list, dtype=torch.long)

        return {
            'loss': loss,
            'track_loss': loss,
            'det_loss': loss,
            'track_labels': filtered_track_labels,
            'det_labels': filtered_det_labels_list,
            'track_predictions': track_predictions,
            'det_predictions': det_predictions,
            'track_mask': None,
            'det_mask': None,
        }


class MultiFeatureLoss(VideoClipLoss):
    """Compose losses over multimodal and modality-specific embeddings."""

    def __init__(
        self,
        mm_loss: VideoClipLoss,
        per_feature_losses: Optional[Dict[str, VideoClipLoss]] = None,
        per_feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the composite loss.

        Args:
            mm_loss: Loss applied to the fused multimodal features.
            per_feature_losses: Optional mapping from modality name to the
                loss used for that modality.
            per_feature_weights: Optional mapping providing weights for each
                modality-specific loss. Defaults to 1.0 when not provided.
        """
        super().__init__()
        self._mm_loss = mm_loss
        self._per_feature_losses = per_feature_losses or {}
        self._per_feature_weights = per_feature_weights or {}

    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        det_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Compute loss on fused multimodal embeddings
        result = self._mm_loss(
            track_x,
            det_x,
            track_mask,
            detection_mask,
            track_feature_dict=None,
            det_feature_dict=None,
            track_ids=track_ids,
            det_ids=det_ids,
        )

        total_loss = result['loss']

        if self._per_feature_losses:
            if track_feature_dict is None or det_feature_dict is None:
                raise ValueError('MultiFeatureLoss requires feature dictionaries for tracks and detections.')
            for key, loss_fn in self._per_feature_losses.items():
                if key not in track_feature_dict or key not in det_feature_dict:
                    raise KeyError(f'Missing feature "{key}" in feature dicts.')
                sub_result = loss_fn(
                    track_feature_dict[key],
                    det_feature_dict[key],
                    track_mask,
                    detection_mask,
                    track_feature_dict=None,
                    det_feature_dict=None,
                    track_ids=track_ids,
                    det_ids=det_ids,
                )
                weight = self._per_feature_weights.get(key, 1.0)
                result[f'{key}_loss'] = sub_result['loss']
                total_loss = total_loss + weight * sub_result['loss']

        result['loss'] = total_loss
        return result


def run_test() -> None:
    track_x = torch.tensor([
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]
        ],
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]
        ]
    ], dtype=torch.float32)
    track_mask = torch.tensor([[[0], [0], [0], [1]], [[0], [0], [0], [1]]], dtype=torch.bool)
    det_x = torch.tensor([
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]
        ],
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]
        ]
    ], dtype=torch.float32)
    det_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.bool)

    for loss_fn in [ClipLevelInfoNCE(), BatchLevelInfoNCE()]:
        outputs = loss_fn(track_x, det_x, track_mask, det_mask)
        print(outputs)


if __name__ == '__main__':
    run_test()
