# TDLP

Official implementation of the paper [Learning Association via Track–Detection Matching for Multi-Object Tracking](https://arxiv.org/html/2512.22105v1), delivering state-of-the-art results on DanceTrack, SportsMOT, SoccerNet, and BEE24.

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

## Results

| Dataset    | Method     | OD         | HOTA | Weights |
|:-----------|:-----------|:----------:|:----:|:--------|
| DanceTrack | TDLP-bbox  | [DanceTrack](https://github.com/DanceTrack/DanceTrack) | 67.8 | [tdlp_dancetrack_bboxes.pt](https://drive.google.com/file/d/17OoRFR-vdxAbnN9v7moQ4ICr4yfPG0P5/view?usp=drive_link) |
| DanceTrack | TDLP       | [DanceTrack](https://github.com/DanceTrack/DanceTrack) | 70.1 | [tdlp_dancetrack.pt](https://drive.google.com/file/d/1JltyjvIqOy64xE87u1eoTauKbxOAMAN0/view?usp=drive_link) |
| SportsMOT  | TDLP-bbox  | [SportsMOT](https://github.com/MCG-NJU/SportsMOT)  | 74.8 | [tdlp_sportsmot_bboxes.pt](https://drive.google.com/file/d/1XHh3ZJYBmqHUL2pQIRCpZ4VaK42llv5m/view?usp=drive_link) |
| SportsMOT  | TDLP       | [SportsMOT](https://github.com/MCG-NJU/SportsMOT)  | 81.9 | [tdlp_sportsmot.pt](https://drive.google.com/file/d/1__3W_Tm5yqcFrRhx1vebZXWceBe7OBLR/view?usp=drive_link) |
| BEE24      | TDLP-bbox  | [TOPICTrack](https://github.com/holmescao/TOPICTrack) | 51.9 | [tdlp_bee24_bboxes.pt](https://drive.google.com/file/d/1jBeiuKTkoyM10SOVmApYu7BaJR4rRLOO/view?usp=drive_link) |
| MOT17      | TDLP       | [ByteTrack](https://github.com/FoundationVision/ByteTrack)  | 60.6 | [tdlp_mot17.pt](https://drive.google.com/file/d/1prWA9wk5ZAkfgsVgYFTH3ftvTNg61ZB3/view?usp=drive_link) |
| SportsMOT  | TDLP-bbox  | [SportsMOT](https://github.com/MCG-NJU/SportsMOT)  | 52.2 | [tdlp_sportsmot_bboxes.pt](https://drive.google.com/file/d/1XHh3ZJYBmqHUL2pQIRCpZ4VaK42llv5m/view?usp=drive_link) |
| SportsMOT  | TDLP       | [SportsMOT](https://github.com/MCG-NJU/SportsMOT)  | 56.3 | [tdlp_sportsmot.pt](https://drive.google.com/file/d/1__3W_Tm5yqcFrRhx1vebZXWceBe7OBLR/view?usp=drive_link) |

Notes:
- OD: Object Detection model and weights.
- TDLP-bbox only used object detection outputs as input features for association while TDLP additionally exploits appearance and human pose points features.
- All checkpoints can be found [here](https://drive.google.com/drive/folders/1ZT7iofkbHzl6_8HU7WsVtuITiIbq_ABU?usp=sharing)


## TODO

- [x] Publish configs, 
- [x] Publish model checkpoints.
- [ ] Refactor code (move tracker).
- [ ] Add online inference support.
- [ ] Remove YOLOX dependency.
- [ ] Add Ultralytics YOLO support.
- [ ] Create package.

## Citation

```bibtex
@misc{tdlp,
  title         = {Learning Association via Track-Detection Matching for Multi-Object Tracking},
  author        = {Momir Ad{\v{z}}emovi{\'c}},
  year          = {2025},
  eprint        = {2512.22105},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2512.22105}
}
```