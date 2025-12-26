from pathlib import Path
from typing import Dict

import hydra
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from mot_jepa.architectures.tdcp.core import MultiModalTDCP, MultiModalTDSP
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset.common.data import VideoClipData, VideoClipPart
from mot_jepa.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import (
    PredictionBBoxFeatureExtractor,
    SupportedFeatures,
)
from mot_jepa.datasets.dataset.transform.base import IdentityTransform, Transform
from mot_jepa.utils import pipeline
import logging


logger = logging.getLogger('Inference')


def _get_default_configs(base_bbox_full: list[float]) -> list[dict]:
    """Build the default list of motion/config options."""
    return [
        {'name': 'static', 'desc': 'Static box center-ish, high conf.', 'base_bbox': base_bbox_full},
        {'name': 'static_conf_decay', 'desc': 'Static, slow confidence decay.', 'base_bbox': base_bbox_full, 'conf_delta': -0.02},
        {'name': 'linear', 'desc': 'Linear step, stable conf.', 'base_bbox': base_bbox_full, 'dx': 0.01, 'dy': 0.008},
        {'name': 'linear_conf_decay', 'desc': 'Fast x drift with confidence drop.', 'base_bbox': base_bbox_full, 'dx': 0.003, 'dy': 0.0, 'conf_delta': -0.01},
        {'name': 'nonlinear_accel', 'desc': 'Accelerating motion both axes.', 'base_bbox': base_bbox_full, 'accel_x': 0.0001, 'accel_y': 0.00005},
        {'name': 'nonlinear_curve', 'desc': 'Curve: steady x, decelerating y.', 'base_bbox': base_bbox_full, 'dx': 0.002, 'accel_y': -0.0003},
    ]


def _load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load weights if a checkpoint path is provided.

    Args:
        model: Model to populate.
        checkpoint_path: Path to a checkpoint containing a ``model`` state dict.

    Raises:
        FileNotFoundError: If the checkpoint path is set but does not exist.
        KeyError: If the checkpoint file misses the ``model`` key.
    """
    if not checkpoint_path:
        logger.info('No checkpoint provided, running with randomly initialized weights.')
        return

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint "{ckpt_path}" not found.')

    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' not in state_dict:
        raise KeyError(f'Checkpoint "{ckpt_path}" does not contain "model".')
    model.load_state_dict(state_dict['model'])
    logger.info('Loaded checkpoint from %s', ckpt_path)


def _generate_bbox_sequence(config: dict, clip_length: int) -> torch.Tensor:
    """Generate a synthetic bbox sequence from a deterministic config.

    Args:
        config: Configuration specifying motion and confidence behavior.
        clip_length: Number of timesteps to synthesize.

    Returns:
        Tensor of shape ``(clip_length, 5)`` containing ``[x, y, w, h, conf]``.
    """
    base_bbox = torch.tensor(config.get('base_bbox', [0.3, 0.2, 0.12, 0.18, 0.95]), dtype=torch.float32)

    t = torch.linspace(0, clip_length - 1, clip_length, dtype=torch.float32)
    dx = config.get('dx', 0.0)
    dy = config.get('dy', 0.0)
    accel_x = config.get('accel_x', 0.0)
    accel_y = config.get('accel_y', 0.0)
    dw = config.get('dw', 0.0)
    dh = config.get('dh', 0.0)

    x_shift = dx * t + 0.5 * accel_x * (t ** 2)
    y_shift = dy * t + 0.5 * accel_y * (t ** 2)
    w_shift = dw * t
    h_shift = dh * t

    conf_delta = config.get('conf_delta', 0.0)
    conf_amp = config.get('conf_amp', 0.0)
    conf_freq = config.get('conf_freq', 0.0)
    conf_shift = conf_delta * t
    if conf_amp != 0.0:
        conf_shift = conf_shift + conf_amp * torch.sin(2 * torch.pi * conf_freq * t / max(1, clip_length - 1))

    deltas = torch.stack(
        [
            x_shift,
            y_shift,
            w_shift,
            h_shift,
            conf_shift,
        ],
        dim=1,
    )

    sequence = base_bbox.unsqueeze(0) + deltas
    return sequence


def _build_dummy_clip(
    feature_names: set[SupportedFeatures],
    clip_length: int,
    config: dict,
    negative_bbox_shifts: list[list[float]],
) -> VideoClipData:
    """Build synthetic observed/unobserved data for bbox-only inference.

    Args:
        feature_names: Model feature names as ``SupportedFeatures``.
        clip_length: Number of observed frames to include.
        config: Motion and confidence configuration.
        negative_bbox_shift: Shift applied to the negative bbox (relative to the anchor).

    Returns:
        ``VideoClipData`` ready for downstream transforms.

    Raises:
        AssertionError: If bbox features are not requested by the model.
    """
    assert SupportedFeatures.BBOX in feature_names, 'Dummy runner expects bbox-only model.'

    # Tracks: index 0 is the positive track; the rest are negatives.
    n_neg = len(negative_bbox_shifts)
    n_tracks = 1 + n_neg
    observed_features = PredictionBBoxFeatureExtractor.initialize_features(
        feature_names=feature_names,
        n_tracks=n_tracks,
        temporal_length=clip_length,
    )
    observed_ts = torch.arange(clip_length, dtype=torch.long).unsqueeze(0).repeat(n_tracks, 1)
    observed_mask = torch.ones(n_tracks, clip_length, dtype=torch.bool)
    observed_mask[0, :] = False  # only the first track is active

    bbox_sequence = _generate_bbox_sequence(config, clip_length + 1)
    for t in range(clip_length):
        data: Dict[str, torch.Tensor | float] = {
            'bbox_xywh': bbox_sequence[t, :4],
            'bbox_conf': float(bbox_sequence[t, 4]),
        }
        PredictionBBoxFeatureExtractor.set_features(
            feature_names=feature_names,
            features=observed_features,
            object_index=0,
            clip_index=t,
            data=data,
        )

    unobserved_features = PredictionBBoxFeatureExtractor.initialize_features(
        feature_names=feature_names,
        n_tracks=n_tracks,
        temporal_length=1,
    )
    # Anchor taken from last simulated bbox (unobserved step)
    anchor_xywh = bbox_sequence[-1, :4]
    anchor_conf = float(bbox_sequence[-1, 4])

    # Positive detection (anchor)
    PredictionBBoxFeatureExtractor.set_features(
        feature_names=feature_names,
        features=unobserved_features,
        object_index=0,
        clip_index=0,
        data={
            'bbox_xywh': anchor_xywh,
            'bbox_conf': anchor_conf,
        },
    )
    # Negative detections relative to anchor
    for neg_idx, neg_shift in enumerate(negative_bbox_shifts):
        neg_shift_full = neg_shift if len(neg_shift) == 5 else [*neg_shift, 0.0]
        neg_bbox_xywh = anchor_xywh + torch.tensor(neg_shift_full[:4], dtype=torch.float32)
        neg_conf = float(anchor_conf + neg_shift_full[4])
        PredictionBBoxFeatureExtractor.set_features(
            feature_names=feature_names,
            features=unobserved_features,
            object_index=1 + neg_idx,
            clip_index=0,
            data={
                'bbox_xywh': neg_bbox_xywh,
                'bbox_conf': neg_conf,
            },
        )

    unobserved_ts = torch.tensor([clip_length] * n_tracks, dtype=torch.long)
    unobserved_mask = torch.zeros(n_tracks, dtype=torch.bool)

    return VideoClipData(
        observed=VideoClipPart(
            ids=None,
            ts=observed_ts,
            mask=observed_mask,
            features=observed_features,
        ),
        unobserved=VideoClipPart(
            ids=None,
            ts=unobserved_ts,
            mask=unobserved_mask,
            features={k: v[:, 0] for k, v in unobserved_features.items()},
        ),
    )


def _prepare_transform(cfg: GlobalConfig) -> Transform:
    """Return the dataset transform or an identity fallback.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Transform callable to apply to ``VideoClipData``.
    """
    try:
        transform = cfg.dataset.build_transform()
        logger.info('Using configured transform "%s".', transform.name)
        return transform
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning('Falling back to IdentityTransform because transform creation failed: %s', exc)
        return IdentityTransform()


def _run_dummy_inference(
    model: torch.nn.Module,
    transform: Transform,
    device: str,
    clip_length: int,
    base_bbox: list[float],
    negative_bbox_shifts: list[list[float]],
) -> None:
    """Run inference for deterministic synthetic sequences.

    Args:
        model: TDCP/TDSP model instance.
        transform: Transform to apply before inference.
        device: Target device for tensors.
        clip_length: Length of the synthetic clip.
        base_bbox: Base bbox used for positive samples.
        negative_bbox_shift: Relative shift applied to create negative samples.

    Raises:
        ValueError: If model type is unsupported.
    """
    if len(base_bbox) not in (4, 5):
        raise ValueError('base_bbox must have length 4 or 5.')
    if any(len(shift) not in (4, 5) for shift in negative_bbox_shifts):
        raise ValueError('All negative_bbox_shifts must have length 4 or 5.')

    base_bbox_full = base_bbox if len(base_bbox) == 5 else [*base_bbox, 0.95]

    feature_names = {SupportedFeatures(name) for name in model.feature_names}
    configs = _get_default_configs(base_bbox_full)

    pos_scores: list[float] = []
    neg_max_scores: list[float] = []
    labels: list[str] = []
    descs: list[str] = []
    successes: list[bool] = []
    rank_matrix: list[list[bool]] = []
    thresh_matrix: list[list[bool]] = []
    threshold_successes: list[bool] = []

    if isinstance(model, MultiModalTDSP):
        pos_thresh = 0.3
        neg_thresh = 0.3
    elif isinstance(model, MultiModalTDCP):
        pos_thresh = 0.05
        neg_thresh = 0.05
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')

    for config in configs:
        labels.append(config['name'])
        descs.append(config['desc'])
        pos_score, neg_scores = _run_single_config(
            model=model,
            transform=transform,
            device=device,
            feature_names=feature_names,
            clip_length=clip_length,
            config=config,
            negative_bbox_shifts=negative_bbox_shifts,
        )
        pos_scores.append(pos_score)
        neg_max = max(neg_scores) if neg_scores else float('-inf')
        neg_max_scores.append(neg_max)
        successes.append(pos_score > neg_max)
        rank_row = [pos_score > n for n in neg_scores]
        thresh_row = [(pos_score > pos_thresh) and (n < neg_thresh) for n in neg_scores]
        rank_matrix.append(rank_row)
        thresh_matrix.append(thresh_row)
        threshold_successes.append(all(thresh_row))

    if pos_scores:
        print('Per-config scores (pos / max-neg, rank_pass=pos>max_neg, thresh_pass):')
        for lbl, desc, pos, neg, succ_rank, succ_thr in zip(labels, descs, pos_scores, neg_max_scores, successes, threshold_successes, strict=False):
            rank_status = 'PASS' if succ_rank else 'FAIL'
            thr_status = 'PASS' if succ_thr else 'FAIL'
            print(f'  {lbl}: pos={pos:.4f}, neg={neg:.4f}, rank={rank_status}, thresh={thr_status}  | {desc}')
        avg_pos = sum(pos_scores) / len(pos_scores)
        avg_neg = sum(neg_max_scores) / len(neg_max_scores)
        success_rate = sum(successes) / len(successes)
        threshold_rate = sum(threshold_successes) / len(threshold_successes)
        print(f'Average positive score across {len(pos_scores)} configs: {avg_pos:.4f}')
        print(f'Average max-negative score across {len(neg_max_scores)} configs: {avg_neg:.4f}')
        print(f'Success rate (pos>neg): {success_rate*100:.2f}%')
        print(f'Success rate (threshold test): {threshold_rate*100:.2f}%')
        failed = [lbl for lbl, succ in zip(labels, successes, strict=False) if not succ]
        if failed:
            print(f'Configs failing (pos<=neg): {failed}')
        failed_thr = [lbl for lbl, succ in zip(labels, threshold_successes, strict=False) if not succ]
        if failed_thr:
            print(f'Configs failing threshold test: {failed_thr}')

        def _print_matrix(title: str, matrix: list[list[bool]]) -> None:
            if not matrix or not matrix[0]:
                return
            cols = [f'n{i}' for i in range(len(matrix[0]))]
            print(f'\n{title}')
            header = 'config'.ljust(18) + ' '.join(c.rjust(4) for c in cols)
            print(header)
            for lbl, row in zip(labels, matrix, strict=False):
                marks = ' '.join(('.' if v else 'X').rjust(4) for v in row)
                print(lbl.ljust(18) + marks)

        _print_matrix('Rank test matrix (.:pass, X:fail)', rank_matrix)
        _print_matrix('Threshold test matrix (.:pass, X:fail)', thresh_matrix)


def _run_translation_invariance_experiment(
    model: torch.nn.Module,
    transform: Transform,
    device: str,
    clip_length: int,
    base_bbox: list[float],
    negative_bbox_shifts: list[list[float]],
    n_translations: int = 1000,
    seed: int = 42,
    translation_range: float = 0.5,
) -> None:
    """Evaluate translation invariance by sampling many coordinate shifts.

    Args:
        model: TDCP/TDSP model.
        transform: Data transform.
        device: Target device.
        clip_length: Clip length.
        base_bbox: Base bbox for the anchor.
        negative_bbox_shift: Relative negative shift.
        n_translations: Number of random translations.
        seed: RNG seed for reproducibility.
        translation_range: Max absolute translation applied to x/y.

    Raises:
        ValueError: If model type is unsupported.
    """
    if len(base_bbox) not in (4, 5):
        raise ValueError('base_bbox must have length 4 or 5.')
    if any(len(shift) not in (4, 5) for shift in negative_bbox_shifts):
        raise ValueError('All negative_bbox_shifts must have length 4 or 5.')

    base_bbox_full = base_bbox if len(base_bbox) == 5 else [*base_bbox, 0.95]
    rng = torch.Generator().manual_seed(seed)
    translations = (torch.rand((n_translations, 2), generator=rng) * 2 * translation_range) - translation_range

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    feature_names = {SupportedFeatures(name) for name in model.feature_names}

    configs = _get_default_configs(base_bbox_full)

    for config in tqdm(configs, desc='Configs', leave=False):
        for idx in tqdm(range(n_translations), desc='Translations', leave=False):
            dx, dy = translations[idx]
            translated_bbox = [
                config['base_bbox'][0] + float(dx),
                config['base_bbox'][1] + float(dy),
                config['base_bbox'][2],
                config['base_bbox'][3],
                config['base_bbox'][4],
            ]
            shifted_config = {**config, 'base_bbox': translated_bbox, 'name': f"{config['name']}_t{idx}"}
            pos, neg = _run_single_config(
                model=model,
                transform=transform,
                device=device,
                feature_names=feature_names,
                clip_length=clip_length,
                config=shifted_config,
                negative_bbox_shifts=negative_bbox_shifts,
            )
            pos_scores.append(pos)
            neg_scores.append(max(neg))

    pos_tensor = torch.tensor(pos_scores)
    neg_tensor = torch.tensor(neg_scores)
    pos_mean = pos_tensor.mean().item()
    pos_var = pos_tensor.var(unbiased=False).item()
    neg_mean = neg_tensor.mean().item()
    neg_var = neg_tensor.var(unbiased=False).item()

    total_runs = len(configs) * n_translations
    print(f'Translation invariance (configs={len(configs)}, runs={total_runs}, seed={seed}, range={translation_range}):')
    print(f'  Positive: mean={pos_mean:.4f}, var={pos_var:.8f}')
    print(f'  Negative: mean={neg_mean:.4f}, var={neg_var:.8f}')

    output_dir = Path('intermediate_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'translation_invariance.png'
    plt.figure(figsize=(8, 4))
    bins = max(10, total_runs // 20)
    plt.hist(pos_scores, bins=bins, alpha=0.6, label='positive', color='tab:blue')
    plt.hist(neg_scores, bins=bins, alpha=0.6, label='negative', color='tab:orange')
    plt.title('Translation invariance score distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'  Saved histogram to {plot_path}')


def _run_clip_length_robustness_experiment(
    model: torch.nn.Module,
    transform: Transform,
    device: str,
    base_bbox: list[float],
    negative_bbox_shifts: list[list[float]],
    clip_lengths: list[int] | None = None,
) -> None:
    """Evaluate robustness across different clip lengths.

    Args:
        model: TDCP/TDSP model instance.
        transform: Transform to apply before inference.
        device: Target device for tensors.
        base_bbox: Base bbox for the anchor.
        negative_bbox_shifts: Relative shifts for negative samples.
        clip_lengths: Clip lengths to test. Defaults to [10, 20, 30, 40, 50].

    Raises:
        ValueError: If bbox inputs have invalid lengths.
    """
    if len(base_bbox) not in (4, 5):
        raise ValueError('base_bbox must have length 4 or 5.')
    if any(len(shift) not in (4, 5) for shift in negative_bbox_shifts):
        raise ValueError('All negative_bbox_shifts must have length 4 or 5.')

    if clip_lengths is None:
        clip_lengths = [10, 20, 30, 40, 50]

    base_bbox_full = base_bbox if len(base_bbox) == 5 else [*base_bbox, 0.95]
    configs = _get_default_configs(base_bbox_full)
    feature_names = {SupportedFeatures(name) for name in model.feature_names}

    pos_means: list[float] = []
    neg_means: list[float] = []

    print('Clip length robustness:')
    for clip_len in tqdm(clip_lengths, desc='Clip lengths', leave=False):
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for config in configs:
            pos, neg = _run_single_config(
                model=model,
                transform=transform,
                device=device,
                feature_names=feature_names,
                clip_length=clip_len,
                config=config,
                negative_bbox_shifts=negative_bbox_shifts,
            )
            pos_scores.append(pos)
            neg_scores.append(max(neg))

        pos_tensor = torch.tensor(pos_scores)
        neg_tensor = torch.tensor(neg_scores)
        pos_mean = pos_tensor.mean().item()
        neg_mean = neg_tensor.mean().item()
        pos_means.append(pos_mean)
        neg_means.append(neg_mean)
        print(f'  clip_len={clip_len}: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}')

    output_dir = Path('intermediate_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'clip_length_robustness.png'
    plt.figure(figsize=(8, 4))
    plt.plot(clip_lengths, pos_means, marker='o', label='positive', color='tab:blue')
    plt.plot(clip_lengths, neg_means, marker='o', label='negative', color='tab:orange')
    plt.title('Clip length robustness')
    plt.xlabel('Clip length')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'  Saved clip-length plot to {plot_path}')


def _run_single_config(
    model: torch.nn.Module,
    transform: Transform,
    device: str,
    feature_names: set[SupportedFeatures],
    clip_length: int,
    config: dict,
    negative_bbox_shifts: list[list[float]],
) -> tuple[float, list[float]]:
    """Run one config and return positive and list of negative scores."""
    clip = _build_dummy_clip(feature_names, clip_length, config, negative_bbox_shifts)
    transformed = transform(clip)
    transformed.apply(lambda x: x.unsqueeze(0).to(device))

    if isinstance(model, MultiModalTDCP):
        track_feats, det_feats, _, _ = model(
            transformed.observed.features,
            transformed.observed.mask,
            transformed.unobserved.features,
            transformed.unobserved.mask,
        )
        track_feats = F.normalize(track_feats, dim=-1)
        det_feats = F.normalize(det_feats, dim=-1)
        similarity = torch.einsum('bnk,bmk->bnm', track_feats, det_feats)
        sim = similarity[0].detach().cpu()  # shape (tracks, dets)
        pos = float(sim[0, 0])
        neg_scores = [float(sim[0, 1 + i]) for i in range(len(negative_bbox_shifts))]
        return pos, neg_scores

    if isinstance(model, MultiModalTDSP):
        logits, _ = model(
            transformed.observed.features,
            transformed.observed.mask,
            transformed.unobserved.features,
            transformed.unobserved.mask,
        )
        probas = torch.sigmoid(logits)[0].detach().cpu()  # shape (tracks, dets)
        pos = float(probas[0, 0])
        neg_scores = [float(probas[0, 1 + i]) for i in range(len(negative_bbox_shifts))]
        return pos, neg_scores

    raise ValueError(f'Unsupported model type: {type(model)}')


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    """Entry point running dummy bbox-only inference."""
    torch.set_printoptions(precision=3, sci_mode=None)
    device = cfg.resources.accelerator

    model = cfg.build_model()
    model.to(device)
    model.eval()
    _load_checkpoint_if_available(model, cfg.eval.checkpoint)

    # clip_length = getattr(cfg.eval.tracker, 'remember_threshold', 4)
    clip_length = 50
    base_bbox = getattr(cfg.eval.tracker, 'base_bbox', [0.30, 0.20, 0.05, 0.10, 0.95])
    STEP = 0.1
    negative_bbox_shifts = getattr(
        cfg.eval.tracker,
        'negative_bbox_shifts',
        [
            [STEP, STEP, STEP, STEP, 0.0],
            [STEP, 0.0, 0.0, 0.0, 0.0],
            [0.0, STEP, 0.0, 0.0, 0.0],
            [STEP/2, STEP/2, 0.0, 0.0, -5 * STEP],
        ],
    )
    transform = _prepare_transform(cfg)
    _run_dummy_inference(model, transform, device, clip_length, base_bbox, negative_bbox_shifts)
    translation_cfg = {
        'enable': True,
        'n_translations': 100,
        'seed': 42,
        'translation_range': 0.5,
    }
    if translation_cfg['enable']:
        _run_translation_invariance_experiment(
            model=model,
            transform=transform,
            device=device,
            clip_length=clip_length,
            base_bbox=base_bbox,
            negative_bbox_shifts=negative_bbox_shifts,
            n_translations=translation_cfg['n_translations'],
            seed=translation_cfg['seed'],
            translation_range=translation_cfg['translation_range'],
        )
    clip_len_cfg = {
        'enable': True,
        'clip_lengths': [5 * (i + 1) for i in range(clip_length // 5)],
    }
    if clip_len_cfg['enable']:
        _run_clip_length_robustness_experiment(
            model=model,
            transform=transform,
            device=device,
            base_bbox=base_bbox,
            negative_bbox_shifts=negative_bbox_shifts,
            clip_lengths=clip_len_cfg['clip_lengths'],
        )


if __name__ == '__main__':
    main()
