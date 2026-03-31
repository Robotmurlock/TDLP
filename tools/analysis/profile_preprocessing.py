#!/usr/bin/env python3
"""Profile the CPU-side preprocessing pipeline (convert_data + transforms) in detail.

Breaks down convert_data into sub-steps and profiles each transform individually
to identify the dominant bottleneck in the association preprocessing.

Usage:
    python tools/analysis/profile_preprocessing.py --config-path=/work/configs/tdlp_v2 --config-name=exp04
    python tools/analysis/profile_preprocessing.py --config-path=/work/configs/tdlp_v2 --config-name=exp04 '+n_values=[5,10,20,30,50,100]'
"""
import copy
import math
import random
import time
from typing import Callable, Dict, List, Set, Tuple

import hydra
from hydra.utils import instantiate
from motrack.library.cv.bbox import BBox, PredBBox
from motrack.tracker.tracklet import Tracklet, TrackletState
from omegaconf import DictConfig, OmegaConf
import torch

import tdlp.config_parser.core  # noqa: F401
from tdlp.common.project import CONFIGS_PATH
from tdlp.datasets.dataset.common.data import VideoClipData, VideoClipPart
from tdlp.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import (
    PredictionBBoxFeatureExtractor,
    SupportedFeatures,
)
from tdlp.datasets.dataset.transform import ComposeTransform


def timed_cpu(
    fn: Callable,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> Dict[str, float]:
    """Time a CPU callable with warmup and averaging."""
    for _ in range(n_warmup):
        fn()

    times: List[float] = []
    for _ in range(n_measure):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return {'mean_ms': mean, 'std_ms': std}


def create_mock_tracklets(
    n_tracks: int,
    clip_length: int,
    frame_index: int,
) -> Tuple[List[Tracklet], List[dict]]:
    """Create mock tracklets with full history and matching detections."""
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
        for fi in range(start_frame + 1, frame_index):
            x += random.uniform(-5, 5)
            y += random.uniform(-5, 5)
            xywh = [x, y, w, h]
            new_bbox = PredBBox.create(BBox.from_xywh(*xywh), label='pedestrian', conf=conf)
            t.update(new_bbox, fi, state=TrackletState.ACTIVE,
                     frame_data={'bbox_xywh': xywh, 'bbox_conf': conf})
        tracklets.append(t)

    objects_data: List[dict] = []
    for _ in range(n_tracks):
        x, y = random.uniform(50, 1870), random.uniform(50, 1030)
        w, h = random.uniform(30, 200), random.uniform(50, 300)
        conf = random.uniform(0.5, 0.99)
        objects_data.append({'bbox_xywh': [x, y, w, h], 'bbox_conf': conf})

    return tracklets, objects_data


def profile_convert_data(
    tracklets: List[Tracklet],
    objects_data: List[dict],
    feature_names: Set[SupportedFeatures],
    clip_length: int,
    frame_index: int,
    n_warmup: int,
    n_measure: int,
) -> Dict[str, Dict[str, float]]:
    """Profile convert_data sub-steps."""
    results: Dict[str, Dict[str, float]] = {}
    N = max(len(tracklets), len(objects_data))
    time_offset = frame_index - clip_length

    # 1. Tensor initialization (observed)
    def init_observed_fn():
        torch.zeros(N, clip_length, dtype=torch.long)
        torch.ones(N, clip_length, dtype=torch.bool)
        PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N, temporal_length=clip_length,
        )

    results['init_observed_tensors'] = timed_cpu(init_observed_fn, n_warmup, n_measure)

    # 2. Fill observed features (the main loop)
    observed_ts = torch.zeros(N, clip_length, dtype=torch.long)
    observed_temporal_mask = torch.ones(N, clip_length, dtype=torch.bool)
    observed_features = PredictionBBoxFeatureExtractor.initialize_features(
        feature_names=feature_names, n_tracks=N, temporal_length=clip_length,
    )

    def fill_observed_fn():
        # Reset tensors
        observed_ts.zero_()
        observed_temporal_mask.fill_(True)
        for k in observed_features:
            observed_features[k].zero_()

        for t_i, tracklet in enumerate(tracklets):
            for frame_info in tracklet.history:
                hist_frame_index = frame_info.frame_index
                data = frame_info.data
                relative_index = hist_frame_index - time_offset
                if relative_index < 0:
                    continue

                PredictionBBoxFeatureExtractor.set_features(
                    feature_names=feature_names,
                    features=observed_features,
                    object_index=t_i,
                    clip_index=relative_index,
                    data=data,
                )
                observed_ts[t_i, relative_index] = hist_frame_index
                observed_temporal_mask[t_i, relative_index] = False

    results['fill_observed_loop'] = timed_cpu(fill_observed_fn, n_warmup, n_measure)

    # 2a. Isolate set_features cost: just the torch.tensor() calls in set_features
    sample_data = tracklets[0].history[0].data
    def set_features_single_fn():
        PredictionBBoxFeatureExtractor.set_features(
            feature_names=feature_names,
            features=observed_features,
            object_index=0,
            clip_index=0,
            data=sample_data,
        )

    results['set_features_single'] = timed_cpu(set_features_single_fn, n_warmup, n_measure)

    # 2b. Just the Python iteration overhead (without set_features)
    def iterate_history_fn():
        for t_i, tracklet in enumerate(tracklets):
            for frame_info in tracklet.history:
                relative_index = frame_info.frame_index - time_offset
                if relative_index < 0:
                    continue
                _ = frame_info.data

    results['iterate_history_only'] = timed_cpu(iterate_history_fn, n_warmup, n_measure)

    # 3. Unobserved tensor init + fill
    def init_fill_unobserved_fn():
        unobserved_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N, temporal_length=1,
        )
        unobserved_ts = torch.zeros(N, dtype=torch.long)
        unobserved_temporal_mask = torch.ones(N, dtype=torch.bool)
        unobserved_ts[:len(objects_data)] = frame_index
        unobserved_temporal_mask[:len(objects_data)] = False

        for d_i, data in enumerate(objects_data):
            PredictionBBoxFeatureExtractor.set_features(
                feature_names=feature_names,
                features=unobserved_features,
                object_index=d_i,
                clip_index=0,
                data=data,
            )
        {k: v[:, 0] for k, v in unobserved_features.items()}

    results['init_fill_unobserved'] = timed_cpu(init_fill_unobserved_fn, n_warmup, n_measure)

    # 4. Full convert_data
    def full_convert_data_fn():
        N_ = max(len(tracklets), len(objects_data))
        obs_ts = torch.zeros(N_, clip_length, dtype=torch.long)
        obs_mask = torch.ones(N_, clip_length, dtype=torch.bool)
        obs_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N_, temporal_length=clip_length,
        )

        t_offset = frame_index - clip_length
        for t_i, tracklet in enumerate(tracklets):
            for frame_info in tracklet.history:
                rel_idx = frame_info.frame_index - t_offset
                if rel_idx < 0:
                    continue
                PredictionBBoxFeatureExtractor.set_features(
                    feature_names=feature_names, features=obs_features,
                    object_index=t_i, clip_index=rel_idx, data=frame_info.data,
                )
                obs_ts[t_i, rel_idx] = frame_info.frame_index
                obs_mask[t_i, rel_idx] = False

        unobs_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N_, temporal_length=1,
        )
        unobs_ts = torch.zeros(N_, dtype=torch.long)
        unobs_mask = torch.ones(N_, dtype=torch.bool)
        unobs_ts[:len(objects_data)] = frame_index
        unobs_mask[:len(objects_data)] = False
        for d_i, data in enumerate(objects_data):
            PredictionBBoxFeatureExtractor.set_features(
                feature_names=feature_names, features=unobs_features,
                object_index=d_i, clip_index=0, data=data,
            )
        unobs_features = {k: v[:, 0] for k, v in unobs_features.items()}

        VideoClipData(
            observed=VideoClipPart(ids=None, ts=obs_ts, mask=obs_mask, features=obs_features),
            unobserved=VideoClipPart(ids=None, ts=unobs_ts, mask=unobs_mask, features=unobs_features),
        )

    results['full_convert_data'] = timed_cpu(full_convert_data_fn, n_warmup, n_measure)

    return results


def profile_transforms(
    transform: ComposeTransform,
    sample_data: VideoClipData,
    n_warmup: int,
    n_measure: int,
) -> Dict[str, Dict[str, float]]:
    """Profile each transform in the ComposeTransform individually."""
    results: Dict[str, Dict[str, float]] = {}

    # Profile each transform individually
    for t in transform._transforms:
        name = t._name

        def transform_fn(t=t):
            data_copy = copy.deepcopy(sample_data)
            t.apply(data_copy)

        results[f'transform_{name}'] = timed_cpu(transform_fn, n_warmup, n_measure)

    # Profile full transform pipeline
    def full_transform_fn():
        data_copy = copy.deepcopy(sample_data)
        transform(data_copy)

    results['full_transform'] = timed_cpu(full_transform_fn, n_warmup, n_measure)

    # Profile deepcopy overhead alone (to understand how much is copy vs compute)
    def deepcopy_fn():
        copy.deepcopy(sample_data)

    results['deepcopy_overhead'] = timed_cpu(deepcopy_fn, n_warmup, n_measure)

    return results


def print_section(
    title: str,
    all_results: Dict[int, Dict[str, Dict[str, float]]],
    n_values: List[int],
    components: List[str],
    total_key: str,
) -> None:
    """Print a profiling results section."""
    col_width = 14
    header = f"{'Component':<30}" + ''.join(f"{'N=' + str(v):>{col_width}}" for v in n_values)
    sep = '-' * len(header)

    print(f'\n{"=" * len(header)}')
    print(f'{title} — Mean time (ms)')
    print(f'{"=" * len(header)}')
    print(header)
    print(sep)
    for comp in components:
        row = f'{comp:<30}'
        for v in n_values:
            r = all_results[v].get(comp)
            if r is None:
                row += f'{"N/A":>{col_width}}'
            else:
                row += f"{r['mean_ms']:>{col_width - 4}.2f}±{r['std_ms']:<3.1f}"
            row = row.ljust(30 + col_width * (n_values.index(v) + 1))
        print(row)

    if total_key:
        print(f'\n% of {total_key}')
        print(sep)
        print(header)
        print(sep)
        for comp in components:
            if comp == total_key:
                continue
            row = f'{comp:<30}'
            for v in n_values:
                r = all_results[v].get(comp)
                total = all_results[v].get(total_key)
                if r is None or total is None:
                    row += f'{"N/A":>{col_width}}'
                else:
                    pct = r['mean_ms'] / total['mean_ms'] * 100 if total['mean_ms'] > 0 else 0
                    row += f'{pct:>{col_width}.1f}%'
            print(row)


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    n_values = list(cfg.get('n_values', [5, 10, 20, 30, 50, 100]))
    clip_length = int(cfg.dataset.clip_length)
    n_warmup = cfg.get('n_warmup', 10)
    n_measure = cfg.get('n_measure', 100)

    feature_names = {SupportedFeatures.BBOX}
    frame_index = clip_length + 50

    print(f'Clip length: {clip_length}')
    print(f'Feature names: {feature_names}')

    # Build transform from config
    transform = instantiate(cfg.dataset.transform)
    print(f'Transform pipeline: {[t._name for t in transform._transforms]}')

    convert_data_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    transform_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for n in n_values:
        print(f'\nProfiling N={n} ...')
        tracklets, objects_data = create_mock_tracklets(n, clip_length, frame_index)

        # Profile convert_data sub-steps
        convert_data_results[n] = profile_convert_data(
            tracklets, objects_data, feature_names, clip_length, frame_index,
            n_warmup, n_measure,
        )
        print(f'  convert_data: {convert_data_results[n]["full_convert_data"]["mean_ms"]:.2f} ms')

        # Build sample data for transform profiling
        N = max(len(tracklets), len(objects_data))
        time_offset = frame_index - clip_length
        obs_ts = torch.zeros(N, clip_length, dtype=torch.long)
        obs_mask = torch.ones(N, clip_length, dtype=torch.bool)
        obs_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N, temporal_length=clip_length,
        )
        for t_i, tracklet in enumerate(tracklets):
            for frame_info in tracklet.history:
                rel_idx = frame_info.frame_index - time_offset
                if rel_idx < 0:
                    continue
                PredictionBBoxFeatureExtractor.set_features(
                    feature_names=feature_names, features=obs_features,
                    object_index=t_i, clip_index=rel_idx, data=frame_info.data,
                )
                obs_ts[t_i, rel_idx] = frame_info.frame_index
                obs_mask[t_i, rel_idx] = False

        unobs_features = PredictionBBoxFeatureExtractor.initialize_features(
            feature_names=feature_names, n_tracks=N, temporal_length=1,
        )
        unobs_ts = torch.zeros(N, dtype=torch.long)
        unobs_mask = torch.ones(N, dtype=torch.bool)
        unobs_ts[:len(objects_data)] = frame_index
        unobs_mask[:len(objects_data)] = False
        for d_i, data in enumerate(objects_data):
            PredictionBBoxFeatureExtractor.set_features(
                feature_names=feature_names, features=unobs_features,
                object_index=d_i, clip_index=0, data=data,
            )
        unobs_features = {k: v[:, 0] for k, v in unobs_features.items()}

        sample_data = VideoClipData(
            observed=VideoClipPart(ids=None, ts=obs_ts, mask=obs_mask, features=obs_features),
            unobserved=VideoClipPart(ids=None, ts=unobs_ts, mask=unobs_mask, features=unobs_features),
        )

        # Profile transforms
        transform_results[n] = profile_transforms(transform, sample_data, n_warmup, n_measure)
        print(f'  transform: {transform_results[n]["full_transform"]["mean_ms"]:.2f} ms')

    # Print results
    convert_data_components = [
        'init_observed_tensors',
        'fill_observed_loop',
        'set_features_single',
        'iterate_history_only',
        'init_fill_unobserved',
        'full_convert_data',
    ]
    print_section('convert_data breakdown', convert_data_results, n_values,
                  convert_data_components, 'full_convert_data')

    transform_components = [f'transform_{t._name}' for t in transform._transforms]
    transform_components += ['deepcopy_overhead', 'full_transform']
    print_section('Transform breakdown', transform_results, n_values,
                  transform_components, 'full_transform')

    # Summary
    print('\n\nSummary: set_features cost analysis')
    print('=' * 60)
    for n in n_values:
        fill_loop = convert_data_results[n]['fill_observed_loop']['mean_ms']
        iterate_only = convert_data_results[n]['iterate_history_only']['mean_ms']
        single_call = convert_data_results[n]['set_features_single']['mean_ms']
        n_calls = n * (clip_length - 1)  # approx total set_features calls
        estimated_set_features = single_call * n_calls
        actual_overhead = fill_loop - iterate_only
        print(f'  N={n}: fill_loop={fill_loop:.2f}ms, iterate_only={iterate_only:.2f}ms, '
              f'set_features_overhead={actual_overhead:.2f}ms ({actual_overhead/fill_loop*100:.0f}%), '
              f'estimated_from_single={estimated_set_features:.2f}ms ({n_calls} calls × {single_call:.3f}ms)')


if __name__ == '__main__':
    main()
