"""Tests that the _compute_model_output_and_loss refactor doesn't break ContrastiveTrainer."""
import torch

from tdlp.architectures.tdlp.core import build_mm_tdsp_model
from tdlp.trainer.losses.bce import ClipLevelBCE
from tdlp.trainer.trainer import ContrastiveTrainer
from tdlp.trainer import torch_helper


def _build_small_tdsp():
    # input_dim=5: model splits 10D input into 5D static + 5D motion
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


def _build_transformed_batch(B=2, N=4, T=5):
    """Build a batch that looks like post-transform data.

    Observed features are 10D (5 coords + 5 FoD).
    Unobserved features are 5D (standardized only, no FoD for detections).
    """
    return {
        'observed': {
            'features': {'bbox': torch.randn(B, N, T, 10)},
            'mask': torch.zeros(B, N, T, dtype=torch.bool),
            'ids': torch.arange(N).unsqueeze(0).unsqueeze(-1).expand(B, N, T).clone(),
            'ts': torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, N, T).clone(),
        },
        'unobserved': {
            'features': {'bbox': torch.randn(B, N, 5)},
            'mask': torch.zeros(B, N, dtype=torch.bool),
            'ids': torch.arange(N).unsqueeze(0).expand(B, N).clone(),
            'ts': torch.full((B, N), T, dtype=torch.long),
        }
    }


def _build_trainer(model, loss_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    return ContrastiveTrainer(
        model=model,
        loss_func=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=1,
        tensorboard_log_dirpath='/tmp/test_tb',
        checkpoints_dirpath='/tmp/test_ckpt',
        device='cpu',
    )


def test_compute_model_output_and_loss_returns_model_output():
    model = _build_small_tdsp()
    loss_fn = ClipLevelBCE()
    trainer = _build_trainer(model, loss_fn)
    trainer._on_start()

    data = _build_transformed_batch()
    data = torch_helper.to_device(data, device=trainer._device)

    loss_dict, model_output = trainer._compute_model_output_and_loss(
        data['observed']['features'], data['observed']['mask'], data['observed']['ids'],
        data['unobserved']['features'], data['unobserved']['mask'], data['unobserved']['ids']
    )

    assert 'loss' in loss_dict
    assert model_output is not None


def test_forward_and_loss_returns_loss_with_grad():
    model = _build_small_tdsp()
    loss_fn = ClipLevelBCE()
    trainer = _build_trainer(model, loss_fn)
    trainer._on_start()

    torch.manual_seed(42)
    data = _build_transformed_batch()
    data = torch_helper.to_device(data, device=trainer._device)

    loss_dict = trainer._forward_and_loss(data)
    assert 'loss' in loss_dict
    assert loss_dict['loss'].requires_grad


def test_forward_and_loss_backward_succeeds():
    model = _build_small_tdsp()
    loss_fn = ClipLevelBCE()
    trainer = _build_trainer(model, loss_fn)
    trainer._on_start()

    data = _build_transformed_batch()
    data = torch_helper.to_device(data, device=trainer._device)

    loss_dict = trainer._forward_and_loss(data)
    loss_dict['loss'].backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
