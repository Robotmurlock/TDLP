import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Dict


class SymmetricContrastiveLossFunction(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, track_features: torch.Tensor, det_features: torch.Tensor, track_mask: torch.Tensor, det_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_features: (B, N, E)
            det_features: (B, M, E)
            track_mask: (T, B, N), 1 = missing
            det_mask: (B, M), 1 = missing

        Returns:
            Scalar contrastive loss
        """
        # Compute logits
        # projected_features = F.normalize(projected_features, dim=-1)  # (B, N1, E)
        # det_features = F.normalize(det_features, dim=-1)  # (B, N2, E)
        logits = torch.bmm(track_features, det_features.transpose(1, 2))  # (B, N1, N2)
        # if self.training:
        #     logits = logits / self._temp

        # Adjust masked logits
        track_mask_agg = track_mask.all(dim=-1)  # (B, N1)
        combined_mask = track_mask_agg.unsqueeze(2) | det_mask.unsqueeze(1)
        logits = logits.masked_fill(combined_mask, -1e9)

        B, N, M = logits.shape
        assert N == M, f'Track and detection masks should match! Got {N=}, {M=}'

        agg_track_mask = track_mask.all(dim=-1)
        mask = agg_track_mask | det_mask

        # Deduce labels
        labels = torch.arange(N).to(logits).unsqueeze(0).repeat(B, 1).long()
        track_labels = labels.clone()
        det_labels = labels.clone()
        track_labels[mask] = -100
        det_labels[mask] = -100

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
            'det_predictions': det_predictions,
            'track_mask': agg_track_mask,
            'det_mask': det_mask
        }


def run_test() -> None:
    loss_fn = SymmetricContrastiveLossFunction()

    logits = torch.tensor([
        [
            [1e9, 0, -1e9],
            [0, 1e9, -1e9],
            [0, 0, -1e9]
        ],
        [
            [1e9, 0, 0],
            [0, 1e9, 0],
            [0, 0, 1e9]
        ]
    ], dtype=torch.float32)
    # logits = torch.tensor([
    #     [
    #         [1e9, 0, -1e9],
    #         [0, 1e9, -1e9],
    #         [0, 0, -1e9]
    #     ]
    # ], dtype=torch.float32)
    # logits = torch.tensor([
    #     [
    #         [1e9, 0, 0],
    #         [0, 1e9, 0],
    #         [1e-9, 1e-9, -1e9]
    #     ]
    # ], dtype=torch.float32)
    # logits = torch.tensor([
    #     [
    #         [1e9, 0, 0],
    #         [0, 1e9, 0],
    #         [0, 0, 1e9]
    #     ]
    # ], dtype=torch.float32)

    track_mask = torch.tensor([[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]], dtype=torch.bool)
    det_mask = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.bool)
    # track_mask = torch.tensor([[[0, 0], [0, 0], [1, 1]]], dtype=torch.bool)
    # det_mask = torch.tensor([[0, 0, 0]], dtype=torch.bool)
    # track_mask = torch.tensor([[[0, 0], [0, 0], [0, 0]]], dtype=torch.bool)
    # det_mask = torch.tensor([[0, 0, 1]], dtype=torch.bool)
    # track_mask = torch.tensor([[[0, 0], [0, 0], [1, 0]]], dtype=torch.bool)
    # det_mask = torch.tensor([[0, 0, 0]], dtype=torch.bool)
    loss = loss_fn(logits, track_mask, det_mask)

    print(loss)


if __name__ == '__main__':
    run_test()
