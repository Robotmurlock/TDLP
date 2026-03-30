# TDLP v2 Inference Profiling Report

Config: `history/DanceTrack/tdlp_bboxes_mmdet.yaml` (hidden_dim=512, mm_dim=1024, track_encoder: 4 layers/8 heads/1024 FFN, interaction_encoder: 4 layers/8 heads/1024 FFN, B=1)

## 1. Component Profiling vs Number of Objects (N=M, T=50)

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| feature_encoding | 0.15 (8.6%) | 0.18 (7.8%) | 0.24 (6.8%) | 0.34 (6.8%) | 0.53 (6.3%) | 0.97 (5.3%) |
| track_encoding | 0.99 (57.2%) | 1.57 (68.6%) | 2.51 (70.6%) | 3.63 (72.1%) | 5.68 (67.8%) | 11.01 (60.1%) |
| projection | 0.04 (2.0%) | 0.04 (1.6%) | 0.04 (1.0%) | 0.04 (0.8%) | 0.04 (0.5%) | 0.05 (0.3%) |
| interaction_encoding | 0.61 (35.3%) | 0.58 (25.4%) | 0.58 (16.4%) | 0.61 (12.0%) | 0.71 (8.4%) | 1.04 (5.7%) |
| mm_linear_agg | 0.02 (1.1%) | 0.02 (0.8%) | 0.02 (0.5%) | 0.02 (0.4%) | 0.02 (0.3%) | 0.03 (0.2%) |
| mm_similarity_head | 0.10 (5.8%) | 0.14 (6.3%) | 0.22 (6.3%) | 0.37 (7.3%) | 0.91 (10.9%) | 3.53 (19.3%) |
| pf_similarity_head | 0.09 (5.3%) | 0.10 (4.5%) | 0.15 (4.3%) | 0.22 (4.4%) | 0.54 (6.4%) | 1.96 (10.7%) |
| **full_forward** | **1.74** | **2.28** | **3.55** | **5.04** | **8.37** | **18.33** |

All times in milliseconds. Percentages are relative to full_forward.

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

## 3. Compact Similarity Head (TDSPCompactMLPHead, proj_dim=128)

Uses only `|z1-z2|` as pair embedding (instead of `[z1, z2, |z1-z2|]`) with a learned projection `E->d` before pair creation. Pair tensor goes from `(B, N, M, 3E)` to `(B, N, M, d)`.

| Component | N=5 | N=10 | N=20 | N=30 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| feature_encoding | 0.14 (7.6%) | 0.17 (7.9%) | 0.24 (7.2%) | 0.34 (7.3%) | 0.53 (7.3%) | 0.96 (6.9%) |
| track_encoding | 0.93 (50.2%) | 1.49 (68.1%) | 2.53 (74.7%) | 3.67 (79.0%) | 5.69 (78.8%) | 10.93 (78.9%) |
| projection | 0.04 (1.9%) | 0.04 (1.8%) | 0.04 (1.2%) | 0.04 (0.8%) | 0.04 (0.6%) | 0.05 (0.4%) |
| interaction_encoding | 0.61 (33.1%) | 0.60 (27.2%) | 0.62 (18.2%) | 0.61 (13.1%) | 0.71 (9.8%) | 1.04 (7.5%) |
| mm_linear_agg | 0.02 (0.9%) | 0.02 (0.8%) | 0.02 (0.5%) | 0.02 (0.4%) | 0.02 (0.3%) | 0.03 (0.2%) |
| mm_similarity_head | 0.10 (5.5%) | 0.10 (4.5%) | 0.10 (3.0%) | 0.11 (2.5%) | 0.19 (2.6%) | 0.52 (3.7%) |
| pf_similarity_head | 0.10 (5.3%) | 0.10 (4.5%) | 0.10 (2.9%) | 0.11 (2.5%) | 0.18 (2.5%) | 0.51 (3.7%) |
| **full_forward** | **1.85** | **2.19** | **3.38** | **4.65** | **7.21** | **13.86** |

### Similarity Head Speedup (compact_mlp proj=128 vs mlp)

| N | mlp mm+pf (ms) | compact_mlp mm+pf (ms) | Head speedup | Full forward speedup |
|---|---|---|---|---|
| 5 | 0.19 | 0.20 | 1.0x | 1.0x |
| 10 | 0.25 | 0.20 | 1.3x | 1.04x |
| 20 | 0.37 | 0.20 | 1.9x | 1.05x |
| 30 | 0.59 | 0.22 | 2.7x | 1.09x |
| 50 | 1.45 | 0.37 | 3.9x | 1.16x |
| 100 | 5.49 | 1.03 | 5.3x | 1.33x |
