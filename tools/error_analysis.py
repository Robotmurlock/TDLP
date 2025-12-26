import cv2
import hydra
from tdlp.common.project import CONFIGS_PATH
from tdlp.config_parser import GlobalConfig
from tdlp.datasets.dataset import dataset_index_factory
from tdlp.utils import pipeline
import torch
import torch.nn.functional as F


def bbox_iou_matrix_xywh(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the NxN IoU matrix for bounding boxes in (x, y, w, h) format.

    Args:
        bboxes (torch.Tensor): Tensor of shape (N, 4) in (x, y, w, h) format.

    Returns:
        torch.Tensor: NxN IoU matrix.
    """
    # Convert to (x1, y1, x2, y2)
    x1y1 = bboxes[:, :2]
    x2y2 = bboxes[:, :2] + bboxes[:, 2:]
    bboxes_xyxy = torch.cat([x1y1, x2y2], dim=1)  # (N, 4)

    # Compute IoU matrix as before
    boxes1 = bboxes_xyxy[:, None, :]  # (N, 1, 4)
    boxes2 = bboxes_xyxy[None, :, :]  # (1, N, 4)

    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])
    area2 = area1
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou




@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('error_analysis')
def main(cfg: GlobalConfig) -> None:
    torch.set_printoptions(precision=2, sci_mode=None)

    index = 1200

    val_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='val',
        sequence_list=cfg.dataset.index.sequence_list
    )

    val_dataset = cfg.dataset.build_dataset(val_index)
    model = cfg.build_model()
    state_dict = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(state_dict['model'])
    model.to(cfg.resources.accelerator)
    model.eval()

    scene_image = val_dataset.visualize_scene(index)
    data = val_dataset[index]
    data = {k: v.unsqueeze(0).to(cfg.resources.accelerator) for k, v in data.items()}
    K = 8
    track_features, det_features = model(
        data['observed_bboxes'],
        data['observed_temporal_mask'],
        data['unobserved_bboxes'],
        data['unobserved_temporal_mask'],
    )
    track_features = track_features[0].cpu()
    det_features = det_features[0].cpu()
    track_features = F.normalize(track_features, dim=-1)
    det_features = F.normalize(det_features, dim=-1)
    logits = track_features[:K] @ det_features[:K].T

    indices = logits.argmax(dim=1)

    print(logits[:K, :K])
    print(indices[:K])

    cv2.imwrite('/work/test.png', scene_image)

    raw = val_dataset.get_raw(index)
    obs_bboxes = raw.observed_bboxes[:K, -1, :-1]
    print(obs_bboxes)
    print(bbox_iou_matrix_xywh(obs_bboxes))


if __name__ == '__main__':
    main()
