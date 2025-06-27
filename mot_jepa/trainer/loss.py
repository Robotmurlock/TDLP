import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Dict


class SymmetricContrastiveLossFunction(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, logits: torch.Tensor, track_mask: torch.Tensor, det_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, N, M) similarity matrix
            track_mask: (T, B, N), 1 = missing
            det_mask: (B, M), 1 = missing

        Returns:
            Scalar contrastive loss
        """
        B, N, M = logits.shape
        assert N == M, f'Track and detection masks should match! Got {N=}, {M=}'

        agg_track_mask = track_mask.all(dim=0)

        # Deduce labels
        labels = torch.arange(N).to(logits).unsqueeze(0).repeat(B, 1)
        track_labels = labels.clone()
        det_labels = labels.clone()
        track_labels[agg_track_mask] = -100
        det_labels[det_mask] = -100

        # Deduce predictions
        track_predictions = torch.argmax(logits, dim=1)
        det_predictions = torch.argmax(logits, dim=2)

        # Compute loss
        track_loss = F.cross_entropy(logits, track_labels)
        det_loss = F.cross_entropy(einops.rearrange(logits, 'b n m -> b m n'), det_labels)
        loss = (track_loss + det_loss) / 2

        return {
            'loss': loss,
            'track_loss': track_loss,
            'det_loss': det_loss,
            'track_labels': track_labels,
            'det_labels': det_labels,
            'track_predictions': track_predictions,
            'det_predictions': det_predictions
        }
