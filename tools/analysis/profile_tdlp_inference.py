#!/usr/bin/env python3
"""Profile TDLP inference components to identify bottlenecks as N (number of tracks) scales.

Supports two profiling modes:
1. Model component profiling (default): Times each model sub-component with synthetic GPU tensors.
2. Association pipeline profiling (+profile_association=true): Times the full tracking association
   pipeline including preprocessing, transforms, device transfer, model forward, and matching.

Usage:
    # Model component profiling
    python tools/analysis/profile_tdlp_inference.py --config-path=/work/configs/tdlp_v2 --config-name=exp04

    # Association pipeline profiling
    python tools/analysis/profile_tdlp_inference.py --config-path=/work/configs/tdlp_v2 --config-name=exp04 +profile_association=true
"""
import copy
import csv
import math
import random
import time
from typing import Callable, Dict, List, Tuple

import hydra
from hydra.utils import instantiate
from motrack.library.cv.bbox import BBox, PredBBox
from motrack.tracker.matching.utils import hungarian
from motrack.tracker.tracklet import Tracklet, TrackletState
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

import tdlp.config_parser.core  # noqa: F401 — registers the_global_config in ConfigStore
from tdlp.architectures.tdlp.core import MultiModalTDSP
from tdlp.common.project import CONFIGS_PATH
from tdlp.tracker.online import TDLPOnlineTracker

MODEL_COMPONENTS = [
    'feature_encoding',
    'track_encoding',
    'projection',
    'interaction_encoding',
    'mm_linear_agg',
    'mm_similarity_head',
    'pf_similarity_head',
    'full_forward',
]

ASSOCIATION_COMPONENTS = [
    'convert_data',
    'transform',
    'to_device',
    'model_forward',
    'postprocess',
    'hungarian',
    'full_association',
]


def timed_forward(
    fn: Callable,
    n_warmup: int = 30,
    n_measure: int = 100,
) -> Dict[str, float]:
    """Time a GPU callable with warmup and averaging using CUDA events."""
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return {'mean_ms': mean, 'std_ms': std}


def timed_cpu(
    fn: Callable,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> Dict[str, float]:
    """Time a CPU callable with warmup and averaging using perf_counter."""
    for _ in range(n_warmup):
        fn()

    times: List[float] = []
    for _ in range(n_measure):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return {'mean_ms': mean, 'std_ms': std}


def create_mock_tracklets(
    n_tracks: int,
    clip_length: int,
    frame_index: int,
) -> Tuple[List[Tracklet], List[PredBBox], List[dict]]:
    """Create mock tracklets with realistic history and detections."""
    tracklets: List[Tracklet] = []
    for i in range(n_tracks):
        x, y = random.uniform(50, 1870), random.uniform(50, 1030)
        w, h = random.uniform(30, 200), random.uniform(50, 300)
        conf = random.uniform(0.5, 0.99)
        xywh = [x, y, w, h]

        bbox = PredBBox.create(BBox.from_xywh(*xywh), label='pedestrian', conf=conf)
        start_frame = max(0, frame_index - clip_length + 1)
        t = Tracklet(
            bbox=bbox, frame_index=start_frame, max_history=clip_length,
            _id=i, state=TrackletState.ACTIVE,
            frame_data={'bbox_xywh': xywh, 'bbox_conf': conf},
        )

        # Add history frames with small random motion (up to frame before current)
        for fi in range(start_frame + 1, frame_index):
            x += random.uniform(-5, 5)
            y += random.uniform(-5, 5)
            xywh = [x, y, w, h]
            new_bbox = PredBBox.create(BBox.from_xywh(*xywh), label='pedestrian', conf=conf)
            t.update(new_bbox, fi, state=TrackletState.ACTIVE,
                     frame_data={'bbox_xywh': xywh, 'bbox_conf': conf})
        tracklets.append(t)

    # Create detections (same count as tracks for square cost matrix)
    detections: List[PredBBox] = []
    objects_data: List[dict] = []
    for i in range(n_tracks):
        x, y = random.uniform(50, 1870), random.uniform(50, 1030)
        w, h = random.uniform(30, 200), random.uniform(50, 300)
        conf = random.uniform(0.5, 0.99)
        xywh = [x, y, w, h]
        detections.append(PredBBox.create(BBox.from_xywh(*xywh), label='pedestrian', conf=conf))
        objects_data.append({'bbox_xywh': xywh, 'bbox_conf': conf})

    return tracklets, detections, objects_data


def profile_association(
    tracker: TDLPOnlineTracker,
    N: int,
    clip_length: int,
    device: str = 'cuda:0',
    n_warmup: int = 10,
    n_measure: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Profile each step of the association pipeline."""
    results: Dict[str, Dict[str, float]] = {}
    frame_index = clip_length + 50  # Ensure enough history

    # Pre-generate mock data for all iterations (avoid measuring data generation)
    all_data = [create_mock_tracklets(N, clip_length, frame_index) for _ in range(n_warmup + n_measure)]

    idx = [0]
    def next_data():
        d = all_data[idx[0] % len(all_data)]
        idx[0] += 1
        return d

    # 1. convert_data
    def convert_data_fn():
        tracklets, _, objects_data = next_data()
        tracker._convert_data(tracklets, objects_data, frame_index)

    idx[0] = 0
    results['convert_data'] = timed_cpu(convert_data_fn, n_warmup, n_measure)

    # 2. transform (run on pre-converted data)
    # Pre-convert a batch of data for transform profiling
    sample_tracklets, _, sample_objects_data = all_data[0]
    sample_converted = tracker._convert_data(sample_tracklets, sample_objects_data, frame_index)

    def transform_fn():
        data_copy = copy.deepcopy(sample_converted)
        tracker._transform(data_copy)

    results['transform'] = timed_cpu(transform_fn, n_warmup, n_measure)

    # 3. to_device
    sample_transformed = copy.deepcopy(sample_converted)
    sample_transformed = tracker._transform(sample_transformed)

    def to_device_fn():
        data_copy = copy.deepcopy(sample_transformed)
        data_copy.apply(lambda x: x.unsqueeze(0).to(device))

    results['to_device'] = timed_cpu(to_device_fn, n_warmup, n_measure)

    # 4. model_forward (GPU timing)
    sample_gpu = copy.deepcopy(sample_transformed)
    sample_gpu.apply(lambda x: x.unsqueeze(0).to(device))

    n_tracks = N

    def model_forward_fn():
        with torch.no_grad():
            tracker._model(
                {k: v.clone() for k, v in sample_gpu.observed.features.items()},
                sample_gpu.observed.mask,
                {k: v.clone() for k, v in sample_gpu.unobserved.features.items()},
                sample_gpu.unobserved.mask,
            )

    results['model_forward'] = timed_forward(model_forward_fn, n_warmup, n_measure)

    # 5. postprocess (sigmoid + cost matrix)
    with torch.no_grad():
        sample_logits, _ = tracker._model(
            {k: v.clone() for k, v in sample_gpu.observed.features.items()},
            sample_gpu.observed.mask,
            {k: v.clone() for k, v in sample_gpu.unobserved.features.items()},
            sample_gpu.unobserved.mask,
        )
    sample_logits = sample_logits.detach()

    def postprocess_fn():
        probas = torch.sigmoid(sample_logits).cpu().numpy()
        cost_matrix = 1 - probas[0, :n_tracks, :n_tracks]
        cost_matrix[cost_matrix > 0.5] = np.inf

    results['postprocess'] = timed_cpu(postprocess_fn, n_warmup, n_measure)

    # 6. hungarian
    probas = torch.sigmoid(sample_logits).cpu().numpy()
    sample_cost_matrix = 1 - probas[0, :n_tracks, :n_tracks]
    sample_cost_matrix[sample_cost_matrix > 0.5] = np.inf

    def hungarian_fn():
        hungarian(sample_cost_matrix.copy())

    results['hungarian'] = timed_cpu(hungarian_fn, n_warmup, n_measure)

    # 7. full_association (end-to-end)
    def full_association_fn():
        tracklets, _, objects_data = next_data()
        with torch.no_grad():
            tracker._association(tracklets, objects_data, frame_index, sim_threshold=0.5)

    idx[0] = 0
    results['full_association'] = timed_cpu(full_association_fn, n_warmup, n_measure)

    return results


def profile_components(
    model: MultiModalTDSP,
    N: int,
    M: int,
    T: int = 50,
    device: str = 'cuda:0',
    n_warmup: int = 5,
    n_measure: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Profile each model component separately at given N, M."""
    results: Dict[str, Dict[str, float]] = {}
    tdcp = model._mm_tdcp._tdcps['bbox']

    # Infer dimensions from model
    hidden_dim = tdcp._track_encoder._hidden_dim
    mm_dim = model._mm_tdcp._mm_dim

    # 1. Feature encoding (static + motion encoders)
    track_x = torch.randn(1, N, T, 10, device=device)
    det_x = torch.randn(1, M, 5, device=device)

    def feature_enc_fn():
        half = track_x.shape[-1] // 2
        tdcp._static_encoder(det_x)
        track_static = tdcp._static_encoder(track_x[..., :half])
        _ = track_static + tdcp._motion_encoder(track_x[..., half:])

    results['feature_encoding'] = timed_forward(feature_enc_fn, n_warmup, n_measure)

    # 2. Track encoding (transformer over temporal dim)
    track_encoded = torch.randn(1, N, T, hidden_dim, device=device)
    track_mask = torch.zeros(1, N, T, dtype=torch.bool, device=device)

    def track_enc_fn():
        tdcp._track_encoder(track_encoded, track_mask)

    results['track_encoding'] = timed_forward(track_enc_fn, n_warmup, n_measure)

    # 3. Projection
    track_features = torch.randn(1, N, hidden_dim, device=device)

    def proj_fn():
        tdcp._projector(track_features)

    results['projection'] = timed_forward(proj_fn, n_warmup, n_measure)

    # 4. Object interaction encoding
    tracks_proj = torch.randn(1, N, hidden_dim, device=device)
    dets_enc = torch.randn(1, M, hidden_dim, device=device)
    track_mask_2d = torch.zeros(1, N, dtype=torch.bool, device=device)
    det_mask_1d = torch.zeros(1, M, dtype=torch.bool, device=device)

    def interaction_fn():
        tdcp._object_interaction_encoder(tracks_proj, track_mask_2d, dets_enc, det_mask_1d)

    results['interaction_encoding'] = timed_forward(interaction_fn, n_warmup, n_measure)

    # 5. MM linear + aggregation
    mm_linear = model._mm_tdcp._mm_linear_layers['bbox']
    feat_h = torch.randn(1, N, hidden_dim, device=device)

    def mm_linear_fn():
        mm_linear(feat_h)

    results['mm_linear_agg'] = timed_forward(mm_linear_fn, n_warmup, n_measure)

    # 6. MM similarity head
    mm_tracks = torch.randn(1, N, mm_dim, device=device)
    mm_dets = torch.randn(1, M, mm_dim, device=device)

    def mm_sph_fn():
        model._mm_sph(mm_tracks, mm_dets)

    results['mm_similarity_head'] = timed_forward(mm_sph_fn, n_warmup, n_measure)

    # 7. Per-feature similarity head
    pf_tracks = torch.randn(1, N, hidden_dim, device=device)
    pf_dets = torch.randn(1, M, hidden_dim, device=device)

    def pf_sph_fn():
        model._sphs['bbox'](pf_tracks, pf_dets)

    results['pf_similarity_head'] = timed_forward(pf_sph_fn, n_warmup, n_measure)

    # 8. Full forward pass
    full_track_data = torch.randn(1, N, T, 10, device=device)
    full_track_mask = torch.zeros(1, N, T, dtype=torch.bool, device=device)
    full_det_data = torch.randn(1, M, 5, device=device)
    full_det_mask = torch.zeros(1, M, dtype=torch.bool, device=device)

    def full_fn():
        model({'bbox': full_track_data.clone()}, full_track_mask, {'bbox': full_det_data.clone()}, full_det_mask)

    results['full_forward'] = timed_forward(full_fn, n_warmup, n_measure)

    return results


def print_results(
    all_results: Dict[int, Dict[str, Dict[str, float]]],
    sweep_values: List[int],
    components: List[str],
    sweep_label: str = 'N',
) -> None:
    """Print timing table and percentage breakdown."""
    total_key = components[-1]  # last component is the total
    col_width = 14
    header = f"{'Component':<25}" + ''.join(f"{sweep_label + '=' + str(v):>{col_width}}" for v in sweep_values)
    sep = '-' * len(header)

    print('\n' + '=' * len(header))
    print('Mean time (ms) per component')
    print('=' * len(header))
    print(header)
    print(sep)
    for comp in components:
        row = f'{comp:<25}'
        for v in sweep_values:
            r = all_results[v].get(comp)
            if r is None:
                row += f'{"OOM":>{col_width}}'
            else:
                row += f"{r['mean_ms']:>{col_width - 4}.2f}\u00b1{r['std_ms']:<3.1f}"
            row = row.ljust(25 + col_width * (sweep_values.index(v) + 1))
        print(row)

    print(f'\n% of {total_key}')
    print(sep)
    print(header)
    print(sep)
    for comp in components[:-1]:
        row = f'{comp:<25}'
        for v in sweep_values:
            r = all_results[v].get(comp)
            total = all_results[v].get(total_key)
            if r is None or total is None:
                row += f'{"OOM":>{col_width}}'
            else:
                pct = r['mean_ms'] / total['mean_ms'] * 100 if total['mean_ms'] > 0 else 0
                row += f'{pct:>{col_width}.1f}%'
        print(row)


def write_csv(
    all_results: Dict[int, Dict[str, Dict[str, float]]],
    n_values: List[int],
    components: List[str],
    output_path: str,
) -> None:
    """Write results to CSV."""
    total_key = components[-1]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'N', 'mean_ms', 'std_ms', 'pct_of_total'])
        for n in n_values:
            total = all_results[n].get(total_key, {}).get('mean_ms', 0)
            for comp in components:
                r = all_results[n].get(comp)
                if r is None:
                    writer.writerow([comp, n, 'OOM', 'OOM', 'OOM'])
                else:
                    pct = r['mean_ms'] / total * 100 if total > 0 else 0
                    writer.writerow([comp, n, f"{r['mean_ms']:.3f}", f"{r['std_ms']:.3f}", f'{pct:.1f}'])
    print(f'\nResults saved to {output_path}')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    n_values = list(cfg.get('n_values', [5, 10, 20, 30, 50, 100]))
    clip_length = int(cfg.dataset.clip_length)
    t_values = list(cfg.t_values) if cfg.get('t_values') else None
    fixed_n = cfg.get('fixed_n', None)
    n_warmup = cfg.get('n_warmup', 30)
    n_measure = cfg.get('n_measure', 100)
    device = cfg.get('device', 'cuda:0')
    output_csv = cfg.get('output_csv', None)
    profile_assoc = cfg.get('profile_association', False)

    assert torch.cuda.is_available(), 'CUDA is required for profiling'

    print(f'Building model from config on {device}...')
    model = instantiate(cfg.model_config)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    print(f'Clip length: {clip_length}')

    if profile_assoc:
        # Association pipeline profiling
        print('\n=== Association Pipeline Profiling ===')
        transform = instantiate(cfg.dataset.transform)
        tracker = TDLPOnlineTracker(
            transform=transform,
            model=model,
            device=device,
            clip_length=clip_length,
        )

        all_results: Dict[int, Dict[str, Dict[str, float]]] = {}
        for n in n_values:
            print(f'\nProfiling association N=M={n}, T={clip_length} ...')
            try:
                all_results[n] = profile_association(
                    tracker, n, clip_length, device, n_warmup, n_measure,
                )
                total = all_results[n]['full_association']['mean_ms']
                print(f'  Full association: {total:.2f} ms')
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM at N={n} — skipping')
                all_results[n] = {}
                torch.cuda.empty_cache()

        print_results(all_results, n_values, ASSOCIATION_COMPONENTS)
        if output_csv:
            write_csv(all_results, n_values, ASSOCIATION_COMPONENTS, output_csv)
    else:
        # Model component profiling
        sweep_t = t_values is not None
        if sweep_t:
            assert fixed_n is not None, '+fixed_n is required when using +t_values'

        all_results: Dict[int, Dict[str, Dict[str, float]]] = {}

        if sweep_t:
            sweep_values = t_values
            sweep_label = 'T'
            for t in sweep_values:
                print(f'\nProfiling N=M={fixed_n}, T={t} ...')
                try:
                    with torch.no_grad():
                        all_results[t] = profile_components(
                            model, fixed_n, fixed_n, t, device, n_warmup, n_measure,
                        )
                    total = all_results[t]['full_forward']['mean_ms']
                    print(f'  Full forward: {total:.2f} ms')
                except torch.cuda.OutOfMemoryError:
                    print(f'  OOM at T={t} — skipping')
                    all_results[t] = {}
                    torch.cuda.empty_cache()
        else:
            sweep_values = n_values
            sweep_label = 'N'
            for n in sweep_values:
                print(f'\nProfiling N=M={n}, T={clip_length} ...')
                try:
                    with torch.no_grad():
                        all_results[n] = profile_components(
                            model, n, n, clip_length, device, n_warmup, n_measure,
                        )
                    total = all_results[n]['full_forward']['mean_ms']
                    print(f'  Full forward: {total:.2f} ms')
                except torch.cuda.OutOfMemoryError:
                    print(f'  OOM at N={n} — skipping')
                    all_results[n] = {}
                    torch.cuda.empty_cache()

        print_results(all_results, sweep_values, MODEL_COMPONENTS, sweep_label)
        if output_csv:
            write_csv(all_results, sweep_values, MODEL_COMPONENTS, output_csv)


if __name__ == '__main__':
    main()
