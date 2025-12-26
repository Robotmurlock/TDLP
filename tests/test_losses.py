import pytest
from tdlp.trainer.losses.base import VideoClipLoss
from tdlp.trainer.losses.infonce import (
    BatchLevelInfoNCE,
    IDLevelInfoNCE,
    MultiFeatureLoss,
)
import torch


class ConstantLoss(VideoClipLoss):
    """Simple loss returning a constant scalar."""

    def __init__(self, value: float):
        super().__init__()
        self.value = torch.tensor(float(value))

    def forward(
        self,
        track_x: torch.Tensor,
        det_x: torch.Tensor,
        track_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        track_feature_dict=None,
        det_feature_dict=None,
        track_ids=None,
        det_ids=None,
    ):
        return {"loss": self.value}


def _dummy_features():
    track_x = torch.ones(1, 1, 1)
    det_x = torch.ones(1, 1, 1)
    track_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
    det_mask = torch.zeros(1, 1, dtype=torch.bool)
    feature_dict = {"a": torch.ones(1, 1, 1), "b": torch.ones(1, 1, 1)}
    return track_x, det_x, track_mask, det_mask, feature_dict


def test_multi_feature_loss_combines_losses_with_weights():
    track_x, det_x, track_mask, det_mask, feature_dict = _dummy_features()
    mm_loss = ConstantLoss(1.0)
    per_losses = {"a": ConstantLoss(2.0), "b": ConstantLoss(3.0)}
    weights = {"a": 0.5, "b": 2.0}
    loss_fn = MultiFeatureLoss(mm_loss, per_losses, weights)

    result = loss_fn(
        track_x,
        det_x,
        track_mask,
        det_mask,
        track_feature_dict=feature_dict,
        det_feature_dict=feature_dict,
    )

    expected_total = 1.0 + 0.5 * 2.0 + 2.0 * 3.0
    assert torch.isclose(result["loss"], torch.tensor(expected_total))
    assert torch.isclose(result["a_loss"], torch.tensor(2.0))
    assert torch.isclose(result["b_loss"], torch.tensor(3.0))


def test_multi_feature_loss_requires_feature_dicts():
    track_x, det_x, track_mask, det_mask, feature_dict = _dummy_features()
    mm_loss = ConstantLoss(0.0)
    per_losses = {"a": ConstantLoss(1.0)}
    loss_fn = MultiFeatureLoss(mm_loss, per_losses)

    with pytest.raises(ValueError):
        loss_fn(track_x, det_x, track_mask, det_mask)

    with pytest.raises(KeyError):
        loss_fn(
            track_x,
            det_x,
            track_mask,
            det_mask,
            track_feature_dict={"a": torch.ones(1, 1, 1)},
            det_feature_dict={},
        )


def test_id_level_infonce_matches_batch_level_when_ids_equal_indices():
    track_x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    det_x = track_x.clone()
    track_mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    det_mask = torch.zeros(1, 2, dtype=torch.bool)
    track_ids = torch.tensor([[0, 1]])
    det_ids = torch.tensor([[0, 1]])

    # Both should produce same result when IDs equal sequential indices
    batch_loss = BatchLevelInfoNCE()(track_x, det_x, track_mask, det_mask, track_ids=track_ids, det_ids=det_ids)
    id_loss = IDLevelInfoNCE()(track_x, det_x, track_mask, det_mask, track_ids=track_ids, det_ids=det_ids)

    assert torch.isclose(batch_loss["loss"], id_loss["loss"], atol=1e-6)


def test_id_level_infonce_requires_ids():
    track_x = torch.ones(1, 1, 1)
    det_x = torch.ones(1, 1, 1)
    track_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
    det_mask = torch.zeros(1, 1, dtype=torch.bool)
    loss_fn = IDLevelInfoNCE()

    with pytest.raises(ValueError):
        loss_fn(track_x, det_x, track_mask, det_mask)
