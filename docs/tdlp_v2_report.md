# TDLP v2 Inference Profiling Report

## 1. Component Profiling vs Number of Objects (N=M, T=50)

### exp01: MLP Similarity Head (baseline)

Config: `configs/tdlp_v2/default.yaml` (hidden_dim=512, mm_dim=1024, track_encoder: 4 layers/8 heads/1024 FFN, interaction_encoder: 4 layers/8 heads/1024 FFN, B=1, 20.8M params)

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| feature_encoding | 0.14 (7.8%) | 0.18 (8.0%) | 0.24 (6.8%) | 0.34 (6.8%) | 0.53 (6.3%) | 0.98 (5.3%) |
| track_encoding | 0.99 (55.7%) | 1.57 (69.6%) | 2.44 (68.8%) | 3.66 (72.5%) | 5.69 (67.5%) | 10.96 (59.5%) |
| projection | 0.03 (1.8%) | 0.04 (1.7%) | 0.04 (1.1%) | 0.04 (0.8%) | 0.04 (0.5%) | 0.05 (0.3%) |
| interaction_encoding | 0.58 (32.5%) | 0.58 (25.7%) | 0.60 (16.8%) | 0.60 (11.8%) | 0.71 (8.4%) | 1.05 (5.7%) |
| mm_linear_agg | 0.02 (1.0%) | 0.02 (0.7%) | 0.02 (0.5%) | 0.02 (0.4%) | 0.02 (0.3%) | 0.03 (0.2%) |
| mm_similarity_head | 0.10 (5.4%) | 0.14 (6.2%) | 0.23 (6.4%) | 0.37 (7.4%) | 0.92 (10.9%) | 3.53 (19.2%) |
| pf_similarity_head | 0.09 (5.0%) | 0.10 (4.5%) | 0.16 (4.4%) | 0.22 (4.3%) | 0.53 (6.3%) | 1.96 (10.7%) |
| **full_forward** | **1.78** | **2.26** | **3.55** | **5.05** | **8.43** | **18.43** |

All times in milliseconds. Percentages are relative to full_forward.

### exp02: Compact MLP Similarity Head (proj_dim=128)

Uses only `|z1-z2|` as pair embedding (instead of `[z1, z2, |z1-z2|]`) with a learned projection `E->d` before pair creation. Pair tensor goes from `(B, N, M, 3E)` to `(B, N, M, d)`.

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| feature_encoding | 0.14 (8.1%) | 0.17 (7.7%) | 0.24 (7.3%) | 0.34 (7.3%) | 0.53 (7.3%) | 0.96 (7.0%) |
| track_encoding | 0.92 (51.2%) | 1.49 (67.8%) | 2.49 (74.4%) | 3.67 (78.7%) | 5.68 (78.9%) | 10.90 (78.7%) |
| projection | 0.04 (2.0%) | 0.04 (1.8%) | 0.04 (1.2%) | 0.04 (0.8%) | 0.04 (0.6%) | 0.05 (0.4%) |
| interaction_encoding | 0.61 (34.0%) | 0.60 (27.4%) | 0.59 (17.7%) | 0.60 (13.0%) | 0.70 (9.8%) | 1.03 (7.5%) |
| mm_linear_agg | 0.02 (1.0%) | 0.02 (0.8%) | 0.02 (0.6%) | 0.02 (0.4%) | 0.02 (0.3%) | 0.03 (0.2%) |
| mm_similarity_head | 0.11 (5.9%) | 0.10 (4.6%) | 0.10 (3.0%) | 0.12 (2.5%) | 0.19 (2.6%) | 0.52 (3.7%) |
| pf_similarity_head | 0.10 (5.7%) | 0.11 (4.8%) | 0.10 (3.0%) | 0.12 (2.5%) | 0.19 (2.6%) | 0.51 (3.7%) |
| **full_forward** | **1.79** | **2.20** | **3.35** | **4.67** | **7.19** | **13.84** |

### exp01 vs exp02: Similarity Head Speedup

| N | mlp mm+pf (ms) | compact_mlp mm+pf (ms) | Head speedup | Full forward speedup |
|---|---|---|---|---|
| 5 | 0.19 | 0.21 | 1.0x | 1.0x |
| 10 | 0.24 | 0.21 | 1.1x | 1.03x |
| 20 | 0.39 | 0.20 | 2.0x | 1.06x |
| 30 | 0.59 | 0.24 | 2.5x | 1.08x |
| 50 | 1.45 | 0.38 | 3.8x | 1.17x |
| 100 | 5.49 | 1.03 | 5.3x | 1.33x |

## 2. Component Profiling vs Clip Length (N=M=10, varying T)

| Component | T=30 | T=60 | T=90 | T=120 | T=150 |
|---|---|---|---|---|---|
| feature_encoding | 0.15 (7.6%) | 0.20 (8.2%) | 0.22 (6.9%) | 0.28 (7.0%) | 0.34 (6.8%) |
| track_encoding | 1.14 (58.3%) | 1.76 (72.8%) | 2.41 (74.7%) | 3.14 (78.9%) | 4.15 (82.0%) |
| projection | 0.04 (2.0%) | 0.04 (1.6%) | 0.04 (1.2%) | 0.04 (1.0%) | 0.04 (0.8%) |
| interaction_encoding | 0.61 (31.0%) | 0.59 (24.4%) | 0.59 (18.4%) | 0.58 (14.5%) | 0.59 (11.7%) |
| mm_linear_agg | 0.02 (0.9%) | 0.02 (0.7%) | 0.02 (0.5%) | 0.02 (0.4%) | 0.02 (0.4%) |
| mm_similarity_head | 0.15 (7.8%) | 0.14 (5.8%) | 0.14 (4.5%) | 0.14 (3.6%) | 0.15 (2.9%) |
| pf_similarity_head | 0.11 (5.5%) | 0.10 (4.3%) | 0.10 (3.3%) | 0.10 (2.6%) | 0.11 (2.1%) |
| **full_forward** | **1.95** | **2.42** | **3.22** | **3.98** | **5.06** |

## 3. Track Encoder Comparison: Transformer vs TCN (N=30, compact_mlp, proj_dim=128)

### Track encoder only (ms)

| Encoder | T=50 | T=150 | T scaling (150/50) |
|---|---|---|---|
| Transformer 4-layer | 3.96 | 11.45 | 2.9x |
| Transformer 2-layer | 2.72 | 7.14 | 2.6x |
| TCN 6-block (dilations 1,2,4,8,16,32) | 6.20 | 13.44 | 2.2x |

### Full forward (ms)

| Encoder | T=50 | T=150 |
|---|---|---|
| Transformer 4-layer (exp02) | 6.77 | 17.74 |
| TCN 6-block (exp04) | 8.45 | 11.56 |

### Observations

- **Transformer 2-layer** is the fastest at both T=50 and T=150 for the track encoder alone
- **TCN** has better T-scaling (2.2x vs 2.9x) due to O(T) vs O(T^2) complexity, but higher constant factor from many small Conv1d ops vs fused attention kernels
- **TCN full forward wins at T=150** (11.56 vs 17.74 ms) due to reduced GPU contention from the lighter encoder
- The crossover point where TCN beats transformer is around T=100-150
- For DanceTrack (T=50, N~10-30), the 2-layer transformer is the most efficient choice

## 4. exp04: Small Model (hidden_dim=128, mm_dim=256, 2-layer encoders)

Config: `configs/tdlp_v2/exp04.yaml` (hidden_dim=128, mm_dim=256, track_encoder: 2 layers/8 heads/512 FFN, interaction_encoder: 2 layers/8 heads/512 FFN, B=1, 1.0M params)

### Model Component Profiling (N=M, T=50)

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| feature_encoding | 0.11 (9.5%) | 0.11 (9.6%) | 0.12 (9.7%) | 0.12 (9.8%) | 0.14 (9.4%) | 0.23 (8.3%) |
| track_encoding | 0.38 (31.6%) | 0.37 (31.8%) | 0.45 (37.8%) | 0.62 (49.4%) | 0.85 (57.4%) | 1.47 (53.4%) |
| projection | 0.03 (2.7%) | 0.03 (2.8%) | 0.04 (2.9%) | 0.04 (3.0%) | 0.03 (2.3%) | 0.03 (1.2%) |
| interaction_encoding | 0.33 (27.8%) | 0.33 (28.4%) | 0.32 (26.7%) | 0.34 (27.2%) | 0.33 (22.3%) | 0.32 (11.8%) |
| mm_linear_agg | 0.01 (1.0%) | 0.01 (1.0%) | 0.01 (1.0%) | 0.01 (1.0%) | 0.01 (0.9%) | 0.01 (0.4%) |
| mm_similarity_head | 0.09 (8.0%) | 0.09 (7.8%) | 0.09 (7.3%) | 0.10 (8.0%) | 0.17 (11.7%) | 0.51 (18.3%) |
| pf_similarity_head | 0.10 (8.2%) | 0.09 (7.6%) | 0.09 (7.5%) | 0.09 (7.1%) | 0.13 (8.5%) | 0.35 (12.5%) |
| **full_forward** | **1.19** | **1.18** | **1.20** | **1.25** | **1.48** | **2.75** |

### exp01 vs exp04: Model Forward Speedup

| N | exp01 (ms) | exp04 (ms) | Speedup |
|---|---|---|---|
| 5 | 1.77 | 1.19 | 1.5x |
| 10 | 2.28 | 1.18 | 1.9x |
| 20 | 3.57 | 1.20 | 3.0x |
| 30 | 5.04 | 1.25 | 4.0x |
| 50 | 8.38 | 1.48 | 5.7x |
| 100 | 18.48 | 2.75 | 6.7x |

## 5. Association Pipeline Profiling (N=M, T=50)

Profiles the full tracking association pipeline: data conversion, transforms, device transfer, model forward, postprocessing, and Hungarian matching. Uses synthetic tracklets with full clip-length history.

### exp04: Small Model

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| convert_data | 2.13 (47.6%) | 4.28 (54.3%) | 8.39 (66.8%) | 12.81 (74.3%) | 21.60 (80.6%) | 42.01 (81.8%) |
| transform | 0.49 (10.8%) | 0.66 (8.3%) | 1.06 (8.5%) | 1.45 (8.4%) | 2.37 (8.8%) | 4.92 (9.6%) |
| to_device | 0.12 (2.6%) | 0.13 (1.6%) | 0.14 (1.1%) | 0.14 (0.8%) | 0.14 (0.5%) | 0.20 (0.4%) |
| model_forward | 1.22 (27.1%) | 1.23 (15.6%) | 1.25 (9.9%) | 1.24 (7.2%) | 1.57 (5.8%) | 2.80 (5.4%) |
| postprocess | 0.01 (0.3%) | 0.01 (0.2%) | 0.01 (0.1%) | 0.01 (0.1%) | 0.02 (0.1%) | 0.03 (0.1%) |
| hungarian | 0.01 (0.2%) | 0.01 (0.1%) | 0.01 (0.1%) | 0.02 (0.1%) | 0.05 (0.2%) | 0.20 (0.4%) |
| **full_association** | **4.49** | **7.88** | **12.55** | **17.25** | **26.79** | **51.39** |

### exp01: Baseline Model

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| convert_data | 2.16 (32.1%) | 4.35 (50.1%) | 8.85 (59.8%) | 13.13 (62.8%) | 21.59 (59.4%) | 42.93 (61.5%) |
| transform | 0.46 (6.8%) | 0.81 (9.4%) | 1.02 (6.9%) | 1.40 (6.7%) | 2.51 (6.9%) | 4.66 (6.7%) |
| to_device | 0.12 (1.7%) | 0.12 (1.4%) | 0.13 (0.9%) | 0.14 (0.6%) | 0.15 (0.4%) | 0.17 (0.2%) |
| model_forward | 1.79 (26.6%) | 2.37 (27.3%) | 3.62 (24.5%) | 5.05 (24.2%) | 8.37 (23.0%) | 18.45 (26.5%) |
| postprocess | 0.01 (0.2%) | 0.01 (0.2%) | 0.01 (0.1%) | 0.01 (0.1%) | 0.02 (0.0%) | 0.04 (0.1%) |
| hungarian | 0.01 (0.1%) | 0.01 (0.1%) | 0.02 (0.1%) | 0.08 (0.4%) | 0.25 (0.7%) | 1.63 (2.3%) |
| **full_association** | **6.73** | **8.68** | **14.79** | **20.91** | **36.31** | **69.75** |

### exp01 vs exp04: Full Association Speedup

| N | exp01 (ms) | exp04 (ms) | Association speedup | Model-only speedup |
|---|---|---|---|---|
| 5 | 6.73 | 4.49 | 1.5x | 1.5x |
| 10 | 8.68 | 7.88 | 1.1x | 1.9x |
| 20 | 14.79 | 12.55 | 1.2x | 3.0x |
| 30 | 20.91 | 17.25 | 1.2x | 4.0x |
| 50 | 36.31 | 26.79 | 1.4x | 5.7x |
| 100 | 69.75 | 51.39 | 1.4x | 6.7x |

### Observations

- **`convert_data` is the dominant bottleneck**, consuming 48-82% of association time. It scales linearly with N due to Python loops over tracklet histories (`N tracks × T frames` iterations).
- **Model forward speedup is largely masked** by preprocessing: exp04 achieves 3x model speedup at N=20, but only 1.2x end-to-end association speedup because `convert_data` is identical for both models.
- **`transform` is the second bottleneck** at ~7-10%, with `FeatureFODStandardization` containing a Python for-loop over N tracks (when `fod_time_scaled=true`).
- **`to_device`**, **`postprocess`**, and **`hungarian`** are negligible (<2% combined) for typical N values.
- The preprocessing overhead (`convert_data` + `transform`) is **purely CPU-bound Python** — optimizing this (e.g., vectorized tensor construction, caching track histories as tensors, moving FOD computation to GPU) would yield larger speedups than further model compression.

## 6. Preprocessing Bottleneck Analysis (CPU profiling, exp04)

### convert_data breakdown

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| init_observed_tensors | 0.00 (0.2%) | 0.00 (0.1%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.01 (0.0%) |
| fill_observed_loop | 2.09 (98.8%) | 4.14 (98.5%) | 8.36 (100%) | 12.54 (100%) | 20.70 (96.3%) | 41.54 (100%) |
| init_fill_unobserved | 0.03 (1.5%) | 0.05 (1.3%) | 0.10 (1.2%) | 0.14 (1.2%) | 0.23 (1.1%) | 0.46 (1.1%) |
| **full_convert_data** | **2.11** | **4.21** | **8.32** | **12.53** | **21.49** | **41.55** |

`fill_observed_loop` (the nested Python loop over N tracks × T frames calling `set_features`) accounts for ~99% of `convert_data`. Within that loop:
- **Iteration overhead** (just traversing tracklet history): 0.01-0.10ms — negligible
- **`set_features` calls**: ~100% of the loop cost. Each call creates a `torch.tensor()` from a Python list and writes it into a pre-allocated tensor via indexed assignment. At N=20, T=50: ~980 calls × 0.005ms/call ≈ 4.7ms estimated, vs 8.3ms actual (the gap is per-call overhead accumulating over many calls).

### Transform breakdown

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| BBoxXYWHtoXYXY | 0.08 (17.2%) | 0.08 (11.6%) | 0.08 (7.5%) | 0.09 (5.9%) | 0.15 (6.1%) | 0.12 (2.5%) |
| BBoxMinMaxScaling | 0.24 (51.3%) | 0.19 (25.8%) | 0.23 (20.5%) | 0.23 (15.4%) | 0.32 (13.4%) | 0.39 (8.3%) |
| FeatureFODStandardization | 0.31 (66.0%) | 0.48 (65.9%) | 0.82 (73.3%) | 1.17 (77.2%) | 2.35 (98.3%) | 4.41 (95.2%) |
| **full_transform** | **0.47** | **0.73** | **1.12** | **1.51** | **2.39** | **4.64** |

Note: Individual transform percentages exceed 100% because each is measured with its own `deepcopy` overhead (0.07-0.12ms).

- **`FeatureFODStandardization`** dominates the transform cost (66-95%), scaling linearly with N due to a Python for-loop over tracks when `fod_time_scaled=true`.
- `BBoxXYWHtoXYXY` and `BBoxMinMaxScaling` are ~constant and negligible at large N.

### Root cause summary

The preprocessing bottleneck has a single root cause: **per-element Python loops creating `torch.tensor()` objects**.

| Bottleneck | Cost at N=20 | Cause |
|---|---|---|
| `set_features` in `fill_observed_loop` | ~8.3ms | `torch.tensor([...])` called N×T=980 times |
| `FeatureFODStandardization` | ~0.8ms | Python for-loop over N tracks for time-scaled FOD |
| Everything else | ~0.2ms | Negligible |

### Optimizations applied

Two optimizations were implemented:

1. **Batch `set_features` per track** (`tdlp/tracker/online.py`): Collect all frame values per track into a list, then make one `torch.tensor()` call per track (N calls) instead of per frame (N×T calls).
2. **Vectorize FOD computation** (`tdlp/datasets/dataset/transform/bbox.py`): Replace the Python for-loop over N tracks in `FeatureFODStandardization` with batched tensor ops using `expand_as` and masked indexing.

## 7. Association Pipeline After Optimization (N=M, T=50)

### exp04: Small Model (optimized)

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| convert_data | 0.22 (10.5%) | 0.41 (17.3%) | 0.80 (26.5%) | 1.23 (34.9%) | 2.04 (42.4%) | 4.08 (38.3%) |
| transform | 0.30 (14.4%) | 0.32 (13.6%) | 0.37 (12.1%) | 0.43 (12.3%) | 0.50 (10.4%) | 0.71 (6.7%) |
| to_device | 0.12 (5.7%) | 0.13 (5.4%) | 0.13 (4.3%) | 0.14 (3.9%) | 0.15 (3.1%) | 0.16 (1.5%) |
| model_forward | 1.24 (59.9%) | 1.24 (52.4%) | 1.23 (40.7%) | 1.27 (36.2%) | 1.57 (32.7%) | 2.79 (26.2%) |
| postprocess | 0.01 (0.7%) | 0.01 (0.5%) | 0.01 (0.4%) | 0.01 (0.4%) | 0.02 (0.3%) | 0.03 (0.3%) |
| hungarian | 0.01 (0.4%) | 0.01 (0.4%) | 0.02 (0.7%) | 0.07 (2.1%) | 0.25 (5.2%) | 1.43 (13.5%) |
| **full_association** | **2.07** | **2.36** | **3.03** | **3.52** | **4.82** | **10.65** |

### exp01: Baseline Model (optimized)

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| convert_data | 0.22 (7.1%) | 0.42 (11.6%) | 0.82 (16.0%) | 1.29 (16.3%) | 2.12 (16.9%) | 4.21 (16.7%) |
| transform | 0.31 (10.1%) | 0.40 (11.0%) | 0.38 (7.4%) | 0.44 (5.6%) | 0.51 (4.1%) | 0.84 (3.4%) |
| to_device | 0.12 (3.9%) | 0.14 (3.7%) | 0.13 (2.5%) | 0.13 (1.7%) | 0.14 (1.1%) | 0.19 (0.7%) |
| model_forward | 1.81 (58.7%) | 2.28 (62.3%) | 3.57 (69.5%) | 5.05 (64.1%) | 8.37 (66.7%) | 18.45 (73.2%) |
| postprocess | 0.01 (0.4%) | 0.01 (0.3%) | 0.01 (0.3%) | 0.01 (0.2%) | 0.01 (0.1%) | 0.02 (0.1%) |
| hungarian | 0.01 (0.3%) | 0.01 (0.3%) | 0.02 (0.4%) | 0.08 (1.0%) | 0.24 (1.9%) | 1.50 (5.9%) |
| **full_association** | **3.09** | **3.65** | **5.14** | **7.88** | **12.55** | **25.19** |

### Before vs After: Association Speedup

| N | exp04 before | exp04 after | exp04 speedup | exp01 before | exp01 after | exp01 speedup |
|---|---|---|---|---|---|---|
| 5 | 4.49 | 2.07 | **2.2x** | 6.73 | 3.09 | **2.2x** |
| 10 | 7.88 | 2.36 | **3.3x** | 8.68 | 3.65 | **2.4x** |
| 20 | 12.55 | 3.03 | **4.1x** | 14.79 | 5.14 | **2.9x** |
| 30 | 17.25 | 3.52 | **4.9x** | 20.91 | 7.88 | **2.7x** |
| 50 | 26.79 | 4.82 | **5.6x** | 36.31 | 12.55 | **2.9x** |
| 100 | 51.39 | 10.65 | **4.8x** | 69.75 | 25.19 | **2.8x** |

### Observations

- **Model forward is now the dominant cost** for both models at typical N values, accounting for 40-73% of association time (previously 5-27% for exp04).
- **`convert_data` dropped from 48-82% to 10-42%** of association time thanks to batched tensor creation (one `torch.tensor()` call per track instead of per frame).
- **`transform` dropped from 7-10% to ~constant 0.3-0.7ms** thanks to vectorized FOD computation.
- At N=20 (typical DanceTrack), exp04 association is now **3.03ms** (down from 12.55ms), making it feasible for real-time tracking at 30fps (33ms budget).
