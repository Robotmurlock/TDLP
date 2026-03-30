#!/usr/bin/env python3
"""Profile TDLP inference components to identify bottlenecks as N (number of tracks) scales.

Builds the MultiModalTDSP model programmatically (no config/checkpoint needed),
creates synthetic inputs at varying N=M, and times each component using CUDA events.

Usage:
    python tools/analysis/profile_tdlp_inference.py --n-values 5 10 20 40 80 160
"""
import argparse
import csv
import math
import sys
from typing import Any, Callable, Dict, List, Optional

import torch

from tdlp.architectures.tdlp.aggregators import tdcp_aggregator_factory
from tdlp.architectures.tdlp.core import (
    MultiModalTDCP,
    MultiModalTDSP,
    build_tdcp_model,
)
from tdlp.architectures.tdlp.similarity_prediction import similarity_head_factory

COMPONENTS = [
    'feature_encoding',
    'track_encoding',
    'projection',
    'interaction_encoding',
    'mm_linear_agg',
    'mm_similarity_head',
    'pf_similarity_head',
    'full_forward',
]


def build_profiling_model(
    device: str = 'cuda:0',
    similarity_head_type: str = 'mlp',
    similarity_head_params: Optional[Dict[str, Any]] = None,
) -> MultiModalTDSP:
    """Build MultiModalTDSP matching history/DanceTrack/tdlp_bboxes_mmdet.yaml."""
    extra_params = similarity_head_params or {}

    tdcp = build_tdcp_model(
        feature_encoder_type='motion',
        feature_encoder_params={'input_dim': 5},
        hidden_dim=512,
        dropout=0.1,
        track_encoder_n_heads=8,
        track_encoder_n_layers=4,
        track_encoder_ffn_dim=1024,
        track_encoder_enable_motion_encoder=True,
        projector_intermediate_dim=512,
        interaction_encoder_enable=True,
        interaction_encoder_n_heads=8,
        interaction_encoder_n_layers=4,
        interaction_encoder_ffn_dim=1024,
    )

    aggregator = tdcp_aggregator_factory('sum', {}, n_features=1)

    mm_tdcp = MultiModalTDCP(
        tdcps={'bbox': tdcp},
        mm_dim=1024,
        aggregator=aggregator,
        object_interaction_encoder=None,
    )

    per_feature_sph = similarity_head_factory(similarity_head_type, input_dim=512, hidden_dim=512, **extra_params)
    mm_sph = similarity_head_factory(similarity_head_type, input_dim=1024, hidden_dim=512, **extra_params)

    model = MultiModalTDSP(
        mm_tdcp=mm_tdcp,
        sphs={'bbox': per_feature_sph},
        mm_sph=mm_sph,
    )
    model.to(device).eval()
    return model


def timed_forward(
    fn: Callable,
    n_warmup: int = 5,
    n_measure: int = 20,
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

    # 1. Feature encoding (static + motion encoders)
    track_x = torch.randn(1, N, T, 10, device=device)
    det_x = torch.randn(1, M, 5, device=device)

    def feature_enc_fn():
        half = track_x.shape[-1] // 2
        det_f = tdcp._static_encoder(det_x)
        track_static = tdcp._static_encoder(track_x[..., :half])
        _ = track_static + tdcp._motion_encoder(track_x[..., half:])

    results['feature_encoding'] = timed_forward(feature_enc_fn, n_warmup, n_measure)

    # 2. Track encoding (transformer over temporal dim)
    track_encoded = torch.randn(1, N, T, 512, device=device)
    track_mask = torch.zeros(1, N, T, dtype=torch.bool, device=device)

    def track_enc_fn():
        tdcp._track_encoder(track_encoded, track_mask)

    results['track_encoding'] = timed_forward(track_enc_fn, n_warmup, n_measure)

    # 3. Projection
    track_features = torch.randn(1, N, 512, device=device)

    def proj_fn():
        tdcp._projector(track_features)

    results['projection'] = timed_forward(proj_fn, n_warmup, n_measure)

    # 4. Object interaction encoding
    tracks_proj = torch.randn(1, N, 512, device=device)
    dets_enc = torch.randn(1, M, 512, device=device)
    track_mask_2d = torch.zeros(1, N, dtype=torch.bool, device=device)
    det_mask_1d = torch.zeros(1, M, dtype=torch.bool, device=device)

    def interaction_fn():
        tdcp._object_interaction_encoder(tracks_proj, track_mask_2d, dets_enc, det_mask_1d)

    results['interaction_encoding'] = timed_forward(interaction_fn, n_warmup, n_measure)

    # 5. MM linear + aggregation
    mm_linear = model._mm_tdcp._mm_linear_layers['bbox']
    feat_512 = torch.randn(1, N, 512, device=device)

    def mm_linear_fn():
        mm_linear(feat_512)

    results['mm_linear_agg'] = timed_forward(mm_linear_fn, n_warmup, n_measure)

    # 6. MM similarity head (input_dim=1024)
    mm_tracks = torch.randn(1, N, 1024, device=device)
    mm_dets = torch.randn(1, M, 1024, device=device)

    def mm_sph_fn():
        model._mm_sph(mm_tracks, mm_dets)

    results['mm_similarity_head'] = timed_forward(mm_sph_fn, n_warmup, n_measure)

    # 7. Per-feature similarity head (input_dim=512)
    pf_tracks = torch.randn(1, N, 512, device=device)
    pf_dets = torch.randn(1, M, 512, device=device)

    def pf_sph_fn():
        model._sphs['bbox'](pf_tracks, pf_dets)

    results['pf_similarity_head'] = timed_forward(pf_sph_fn, n_warmup, n_measure)

    # 8. Full forward pass
    # Note: MultiModalTDCP.forward mutates input dicts (overwrites features with encoded values),
    # so we must create fresh copies each call.
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
    sweep_label: str = 'N',
) -> None:
    """Print timing table and percentage breakdown."""
    col_width = 14
    header = f"{'Component':<25}" + ''.join(f"{sweep_label + '=' + str(v):>{col_width}}" for v in sweep_values)
    sep = '-' * len(header)

    print('\n' + '=' * len(header))
    print('Mean time (ms) per component')
    print('=' * len(header))
    print(header)
    print(sep)
    for comp in COMPONENTS:
        row = f'{comp:<25}'
        for v in sweep_values:
            r = all_results[v].get(comp)
            if r is None:
                row += f'{"OOM":>{col_width}}'
            else:
                row += f"{r['mean_ms']:>{col_width - 4}.2f}\u00b1{r['std_ms']:<3.1f}"
            row = row.ljust(25 + col_width * (sweep_values.index(v) + 1))
        print(row)

    print(f'\n{"% of full forward":}')
    print(sep)
    print(header)
    print(sep)
    for comp in COMPONENTS[:-1]:
        row = f'{comp:<25}'
        for v in sweep_values:
            r = all_results[v].get(comp)
            total = all_results[v].get('full_forward')
            if r is None or total is None:
                row += f'{"OOM":>{col_width}}'
            else:
                pct = r['mean_ms'] / total['mean_ms'] * 100 if total['mean_ms'] > 0 else 0
                row += f'{pct:>{col_width}.1f}%'
        print(row)


def write_csv(
    all_results: Dict[int, Dict[str, Dict[str, float]]],
    n_values: List[int],
    output_path: str,
) -> None:
    """Write results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'N', 'mean_ms', 'std_ms', 'pct_of_total'])
        for n in n_values:
            total = all_results[n].get('full_forward', {}).get('mean_ms', 0)
            for comp in COMPONENTS:
                r = all_results[n].get(comp)
                if r is None:
                    writer.writerow([comp, n, 'OOM', 'OOM', 'OOM'])
                else:
                    pct = r['mean_ms'] / total * 100 if total > 0 else 0
                    writer.writerow([comp, n, f"{r['mean_ms']:.3f}", f"{r['std_ms']:.3f}", f'{pct:.1f}'])
    print(f'\nResults saved to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile TDLP inference bottlenecks')
    parser.add_argument('--n-values', nargs='+', type=int, default=[5, 10, 20, 40, 80, 160],
                        help='Number of tracks (N=M) to sweep')
    parser.add_argument('--clip-length', type=int, default=50, help='Temporal clip length T (fixed when sweeping N)')
    parser.add_argument('--t-values', nargs='+', type=int, default=None,
                        help='Sweep clip lengths T instead of N. Requires --fixed-n.')
    parser.add_argument('--fixed-n', type=int, default=None,
                        help='Fixed N=M when sweeping T')
    parser.add_argument('--n-warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--n-measure', type=int, default=20, help='Measurement iterations')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')
    parser.add_argument('--output-csv', type=str, default=None, help='Optional CSV output path')
    parser.add_argument('--similarity-head-type', type=str, default='mlp',
                        help='Similarity head type (mlp, compact_mlp)')
    parser.add_argument('--similarity-head-proj-dim', type=int, default=None,
                        help='Projection dim for compact_mlp head')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA is required for profiling'

    sweep_t = args.t_values is not None
    if sweep_t:
        assert args.fixed_n is not None, '--fixed-n is required when using --t-values'

    sph_params = {}
    if args.similarity_head_proj_dim is not None:
        sph_params['proj_dim'] = args.similarity_head_proj_dim

    print(f'Building model on {args.device} (similarity_head={args.similarity_head_type})...')
    model = build_profiling_model(args.device, args.similarity_head_type, sph_params or None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    all_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    if sweep_t:
        sweep_values = args.t_values
        sweep_label = 'T'
        for t in sweep_values:
            print(f'\nProfiling N=M={args.fixed_n}, T={t} ...')
            try:
                with torch.no_grad():
                    all_results[t] = profile_components(
                        model, args.fixed_n, args.fixed_n, t, args.device,
                        args.n_warmup, args.n_measure,
                    )
                total = all_results[t]['full_forward']['mean_ms']
                print(f'  Full forward: {total:.2f} ms')
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM at T={t} — skipping')
                all_results[t] = {}
                torch.cuda.empty_cache()
    else:
        sweep_values = args.n_values
        sweep_label = 'N'
        for n in sweep_values:
            print(f'\nProfiling N=M={n}, T={args.clip_length} ...')
            try:
                with torch.no_grad():
                    all_results[n] = profile_components(
                        model, n, n, args.clip_length, args.device,
                        args.n_warmup, args.n_measure,
                    )
                total = all_results[n]['full_forward']['mean_ms']
                print(f'  Full forward: {total:.2f} ms')
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM at N={n} — skipping')
                all_results[n] = {}
                torch.cuda.empty_cache()

    print_results(all_results, sweep_values, sweep_label)

    if args.output_csv:
        write_csv(all_results, sweep_values, args.output_csv)


if __name__ == '__main__':
    main()
