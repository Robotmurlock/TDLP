# TDLP v2 Inference Profiling Report

Config: `history/DanceTrack/tdlp_bboxes_mmdet.yaml` (hidden_dim=512, mm_dim=1024, track_encoder: 4 layers/8 heads/1024 FFN, interaction_encoder: 4 layers/8 heads/1024 FFN, B=1)

## 1. Component Profiling vs Number of Objects (N=M, T=50)

### exp01: MLP Similarity Head (baseline)

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
