# TDLP

Official implementation of the paper **Learning Association via Track–Detection Matching for Multi-Object Tracking**, delivering state-of-the-art results on DanceTrack, SportsMOT, SoccerNet, and BEE24.

## Paper
**Title:** Learning Association via Track–Detection Matching for Multi-Object Tracking

Multi-object tracking aims to maintain object identities over time by associating detections across video frames. Two dominant paradigms exist in literature: tracking-by-detection methods, which are computationally efficient but rely on handcrafted association heuristics, and end-to-end approaches, which learn association from data at the cost of higher computational complexity. We propose Track–Detection Link Prediction (TDLP), a tracking-by-detection method that performs per-frame association via link prediction between tracks and detections, i.e., by predicting the correct continuation of each track at every frame. TDLP is architecturally designed primarily for geometric features such as bounding boxes, while optionally incorporating additional cues, including pose and appearance. Unlike heuristic-based methods, TDLP learns association directly from data without handcrafted rules, while remaining modular and computationally efficient compared to end-to-end trackers. Extensive experiments on multiple benchmarks demonstrate that TDLP consistently surpasses state-of-the-art performance across both tracking-by-detection and end-to-end methods. Finally, we provide a detailed analysis comparing link prediction with metric learning–based association and show that link prediction is more effective, particularly when handling heterogeneous features such as detection bounding boxes.

## Repository Layout
- `configs/` – Hydra defaults, dataset/model/train/eval presets, path config.
- `tdlp/` – Core library (architectures, datasets, trainer, utils, config parser).
- `tools/` – Entry points: `train.py`, `inference.py`, analysis scripts.
- `history/` – Saved experiment settings for reproducibility.
- `requirements.txt`, `pyproject.toml` – Python package definitions.

## Requirements
- Python 3.12+.
- [uv](https://docs.astral.sh/uv/) for training.
- YOLOX for inference (`TODO`: remove this dependency).

## Training

First download the datasets (DanceTrack, SportsMOT, etc.) and place them in the appropriate directories (or update the existing configs).

Run with the default stack (see `configs/default.yaml`):
```bash
uv run tools/train.py --config-name=<name>
```

Use `torchrun` (`uv run torchrun ...`) for multi-GPU training.

Outputs (checkpoints, TensorBoard logs, metrics) are written under `path.master/<dataset>/<experiment_name>/`.

## Inference

**Note:** At the moment, due to YOLOX and uv incompatibility, you need to clone the YOLOX repo and create a separate Python virtual environment with the required packages to run inference.

Use `tools/inference.py` to generate tracker outputs with trained models and detector features:
```bash
python tools/inference.py --config-name=<name>
```

## Evaluation

Use [TrackEval](https://github.com/JonathonLuiten/TrackEval) in order to evaluate tracker performance. You
can also use the [forked](https://github.com/Robotmurlock/TrackEval) repo with fixed minor numpy errors.

## TODO
- [ ] Publish configs, models and checkpoints.
- [ ] Refactor code (move tracker).
- [ ] Add online inference support.
- [ ] Remove YOLOX dependency.
- [ ] Add Ultralytics YOLO support.
- [ ] Create package.
