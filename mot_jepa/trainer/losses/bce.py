from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ClipLevelBCE(nn.Module):
    """
    Binary Cross-Entropy loss computed separately for each clip in the batch.

    Performs binary classification between track and detection embeddings using
    their differences. Uses BCE loss to classify whether embeddings belong to
    the same identity (positive) or different identities (negative).
    """
    
    def __init__(
        self, 
        pos_weight: Optional[float] = 10.0,
        assoc_threshold: float = 0.5, 
        fp_label_threshold: int = 1_000_000
    ) -> None:
        super().__init__()
        pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self._loss_func = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        self._assoc_threshold = assoc_threshold
        self._fp_label_threshold = fp_label_threshold

        # State
        self._k = 0

    def forward(
        self,
        logits: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
        logits_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCE loss using embedding differences for binary classification.

        Args:
            logits: Logits (B, N, M, 1)
            track_mask: Track mask (B, N, T), True=missing
            detection_mask: Detection mask (B, N), True=missing
            track_ids: Track identifiers (B, N) - required
            det_ids: Detection identifiers (B, N) - required
            logits_dict: Dictionary containing modality-specific logits (B, N, M, 1)

        Returns:
            Dictionary containing loss and additional debug information
        """
        B = logits.shape[0]
        _ = logits_dict  # unused
        
        agg_track_mask = track_mask.all(dim=-1)
        agg_track_ids = track_ids.max(dim=-1).values
        mask = ~agg_track_mask.unsqueeze(-1) | ~detection_mask.unsqueeze(-2)
        id_match_mask = (agg_track_ids.unsqueeze(-1) == det_ids.unsqueeze(-2)).float()
        loss = self._loss_func(logits[mask], id_match_mask[mask])

        # Accuracy metrics (clip level like BatchLevelInfoNCE)
        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []

        probas = torch.sigmoid(logits)
        with torch.no_grad():
            for b_i in range(B):
                sub_mask = mask[b_i]
                if not bool(sub_mask.any().item()):
                    continue

                sub_track_labels = agg_track_ids[b_i]
                sub_det_labels = det_ids[b_i]
                sub_probas = probas[b_i]
                sub_probas[~sub_mask] = -1

                if torch.numel(sub_probas) > 0:
                    track_max_probas, track_max_indices = torch.max(sub_probas, dim=1)
                    track_max_indices[track_max_probas <= self._assoc_threshold] = -1
                    track_max_indices = track_max_indices[~agg_track_mask[b_i]]
                    track_predictions = sub_track_labels[track_max_indices]
                    sub_track_labels = sub_track_labels[~agg_track_mask[b_i]]
                    sub_track_labels[sub_track_labels >= self._fp_label_threshold] = -1

                    det_max_probas, det_max_indices = torch.max(sub_probas, dim=0)
                    det_max_indices[det_max_probas <= self._assoc_threshold] = -1
                    det_max_indices = det_max_indices[~detection_mask[b_i]]
                    det_predictions = sub_det_labels[det_max_indices]
                    sub_det_labels = sub_det_labels[~detection_mask[b_i]]
                    sub_det_labels[sub_det_labels >= self._fp_label_threshold] = -1

                    filtered_track_labels_list.append(sub_track_labels)
                    filtered_det_labels_list.append(sub_det_labels)
                    track_predictions_list.append(track_predictions)
                    det_predictions_list.append(det_predictions)

        self._k += 1
        if self._k % 100 == 0:
            print(
                f'{probas[mask].mean()=}', 
                f'{probas[mask].max()=}', 
                f'{id_match_mask[mask].sum() / id_match_mask[mask].numel()=}', 
                f'{id_match_mask[mask].numel()=}'
            )
            print(
                f'{filtered_track_labels_list[0]=}', 
                f'{filtered_det_labels_list[0]=}', 
                f'{track_predictions_list[0]=}', 
                f'{det_predictions_list[0]=}',
                end='\n\n'
            )

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
            'det_mask': None
        }


class MultiFeatureBCELoss(nn.Module):
    """Compose losses over multimodal and modality-specific embeddings."""

    def __init__(
        self,
        mm_loss: nn.Module,
        per_feature_losses: Optional[Dict[str, nn.Module]] = None,
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
        logits: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
        det_ids: Optional[torch.Tensor] = None,
        logits_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        result = self._mm_loss(
            logits,
            track_mask,
            detection_mask,
            track_ids=track_ids,
            det_ids=det_ids,
            logits_dict=logits_dict,
        )

        total_loss = result['loss']

        if self._per_feature_losses:
            if logits_dict is None:
                raise ValueError('MultiFeatureBCELoss requires logits dictionary.')
            for key, loss_fn in self._per_feature_losses.items():
                if key not in logits_dict:
                    raise KeyError(f'Missing logits for modality "{key}".')
                sub_result = loss_fn(
                    logits_dict[key],
                    track_mask,
                    detection_mask,
                    track_ids=track_ids,
                    det_ids=det_ids,
                    logits_dict={key: logits_dict[key]},
                )
                weight = self._per_feature_weights.get(key, 1.0)
                result[f'{key}_loss'] = sub_result['loss']
                total_loss = total_loss + weight * sub_result['loss']

        result['loss'] = total_loss
        return result


def run_test() -> None:
    """Test function for ClipLevelBCE."""
    B, N, E = 2, 4, 8
    
    # Create dummy difference embeddings
    diff_x = torch.randn(B, N, E)
    
    # Create masks: some valid differences, some identity matches
    diff_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool)
    id_match_mask = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.bool)
    
    # Create optional feature dictionary
    diff_feature_dict = {'additional_feature': torch.randn(B, N, E)}
    
    # Test the loss
    loss_fn = ClipLevelBCE()
    outputs = loss_fn(
        diff_x=diff_x,
        diff_mask=diff_mask,
        diff_feature_dict=diff_feature_dict,
        id_match_mask=id_match_mask
    )
    
    print("BCE Loss outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} - {value.item() if value.numel() == 1 else 'tensor'}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    run_test()
