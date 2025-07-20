import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from torch.nn import functional as F

from mot_jepa.trainer.losses.base import VideoClipLoss


class ClipLevelTripletLoss(VideoClipLoss):
    def __init__(self, margin: float = 0.5, type_of_triplets: str = 'all'):
        super().__init__()
        self._miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=type_of_triplets)
        self._loss_func = losses.TripletMarginLoss(margin=margin)

    def forward(self, track_x, det_x, track_mask, detection_mask):
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

        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []

        losses = []
        for b_i in range(B):
            sub_track_labels = track_labels[b_i][~agg_track_mask[b_i]]
            sub_det_labels = det_labels[b_i][~detection_mask[b_i]]
            sub_labels = torch.cat([sub_track_labels, sub_det_labels], dim=0)
            filtered_track_labels_list.append(sub_track_labels)
            filtered_det_labels_list.append(sub_det_labels)

            sub_track_x = track_x[b_i][~agg_track_mask[b_i]]
            sub_det_x = det_x[b_i][~detection_mask[b_i]]
            sub_embeddings = torch.cat([sub_track_x.detach(), sub_det_x], dim=0)

            hard_pairs = self._miner(sub_embeddings, sub_labels)
            sub_loss = self._loss_func(sub_embeddings, sub_labels, hard_pairs)
            losses.append(sub_loss)

            distances = torch.cdist(sub_embeddings[:sub_track_x.shape[0]], sub_embeddings[-sub_det_x.shape[0]:], p=2)

            sub_track_predictions = torch.argmin(distances, dim=1)
            sub_det_predictions = torch.argmin(distances, dim=0)
            track_predictions_list.append(sub_track_predictions)
            det_predictions_list.append(sub_det_predictions)

        # Postprocess
        filtered_track_labels = torch.cat(filtered_track_labels_list)
        filtered_det_labels_list = torch.cat(filtered_det_labels_list)
        track_predictions = torch.cat(track_predictions_list)
        det_predictions = torch.cat(det_predictions_list)

        loss = sum(losses) / len(losses)
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


class BatchLevelTripletLoss(VideoClipLoss):
    def __init__(self, margin: float = 0.5, type_of_triplets: str = 'all'):
        super().__init__()
        self._miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=type_of_triplets)
        self._loss_func = losses.TripletMarginLoss(margin=margin)

    def forward(self, track_x, det_x, track_mask, detection_mask):
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
        agg_track_mask = track_mask.all(dim=-1) # shape: (B, N), True indicates missing
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
        embeddings = torch.cat([filtered_track_x.detach(), filtered_det_x], dim=0)
        hard_pairs = self._miner(embeddings, labels)
        loss = self._loss_func(embeddings, labels, hard_pairs)

        # Labels and predictions are still calculated at clip level
        filtered_track_labels_list = []
        filtered_det_labels_list = []
        track_predictions_list = []
        det_predictions_list = []
        with torch.no_grad():
            track_labels = torch.arange(N).to(track_x).unsqueeze(0).repeat(B, 1).long()
            det_labels = torch.arange(N).to(det_x).unsqueeze(0).repeat(B, 1).long()
            for b_i in range(B):
                sub_track_labels = track_labels[b_i][~agg_track_mask[b_i]]
                sub_det_labels = det_labels[b_i][~detection_mask[b_i]]

                sub_track_x = track_x[b_i][~agg_track_mask[b_i]]
                sub_det_x = det_x[b_i][~detection_mask[b_i]]
                sub_track_x = F.normalize(sub_track_x, dim=-1)
                sub_det_x = F.normalize(sub_det_x, dim=-1)

                distances = torch.cdist(sub_track_x, sub_det_x, p=2)
                sub_track_predictions = torch.argmin(distances, dim=1)
                sub_det_predictions = torch.argmin(distances, dim=0)

                filtered_track_labels_list.append(sub_track_labels)
                filtered_det_labels_list.append(sub_det_labels)
                track_predictions_list.append(sub_track_predictions)
                det_predictions_list.append(sub_det_predictions)

        filtered_track_labels = torch.cat(filtered_track_labels_list)
        filtered_det_labels_list = torch.cat(filtered_det_labels_list)
        track_predictions = torch.cat(track_predictions_list)
        det_predictions = torch.cat(det_predictions_list)

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


def run_test() -> None:
    track_x = torch.tensor([
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0]
        ],
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0]
        ]
    ], dtype=torch.float32)
    track_mask = torch.tensor([[[0], [0], [0], [1]], [[0], [0], [0], [1]]], dtype=torch.bool)
    det_x = torch.tensor([
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0]
        ],
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0]
        ]
    ], dtype=torch.float32)
    det_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.bool)

    for loss_fn in [ClipLevelTripletLoss(), BatchLevelTripletLoss()]:
        outputs = loss_fn(track_x, det_x, track_mask, det_mask)
        print(outputs)


if __name__ == '__main__':
    run_test()
