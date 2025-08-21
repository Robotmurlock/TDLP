import torch

from mot_jepa.trainer.losses.infonce import ClipLevelInfoNCE, BatchLevelInfoNCE


def sample_inputs():
    track_x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    det_x = track_x.clone()
    track_mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    det_mask = torch.zeros(1, 2, dtype=torch.bool)
    return track_x, det_x, track_mask, det_mask


def test_clip_level_infonce_runs():
    track_x, det_x, track_mask, det_mask = sample_inputs()
    loss_dict = ClipLevelInfoNCE()(track_x, det_x, track_mask, det_mask)
    assert loss_dict['loss'].shape == ()


def test_batch_level_infonce_runs():
    track_x, det_x, track_mask, det_mask = sample_inputs()
    loss_dict = BatchLevelInfoNCE()(track_x, det_x, track_mask, det_mask)
    assert loss_dict['loss'].shape == ()
