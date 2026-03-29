"""
Run object detection on a single image (or directory of images) and write
results to a JSON file.

Usage
-----
Single image:
    python tools/helpers/od_inference.py \\
        --config configs/eval/object_detection/mmdet_yolox_dancetrack.yaml \\
        --input path/to/image.jpg \\
        --output detections.json

Directory of images:
    python tools/helpers/od_inference.py \\
        --config configs/eval/object_detection/mmdet_yolox_dancetrack.yaml \\
        --input path/to/frames/ \\
        --output detections.json

Output JSON schema
------------------
Single image:
    [{"bbox_xyxy": [x1, y1, x2, y2], "class": 0, "conf": 0.85}, ...]

Directory:
    {"frame_000001.jpg": [...], "frame_000002.jpg": [...], ...}
"""
import argparse
import json
import os
import sys

# Ensure project root is on the path when running the script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2  # noqa: E402
import yaml  # noqa: E402
import tdlp.object_detection  # noqa: F401, E402 — registers mmdet_yolox detector
from motrack.object_detection.factory import object_detection_inference_factory  # noqa: E402


_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _run_on_image(detector, image_path: str) -> list:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')
    bboxes, classes, confidences = detector.predict_with_postprocess(image)
    return [
        {
            'bbox_xyxy': bbox.tolist(),
            'class': int(cls),
            'conf': float(conf),
        }
        for bbox, cls, conf in zip(bboxes, classes, confidences)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description='OD inference to JSON')
    parser.add_argument('--config', required=True, help='Path to OD yaml config')
    parser.add_argument('--input', required=True, help='Image file or directory')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    detector = object_detection_inference_factory(
        name=cfg['type'],
        params=cfg['params'],
    )

    if os.path.isdir(args.input):
        results = {}
        files = sorted(
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        for fname in files:
            fpath = os.path.join(args.input, fname)
            results[fname] = _run_on_image(detector, fpath)
            print(f'  {fname}: {len(results[fname])} detections')
    else:
        results = _run_on_image(detector, args.input)
        print(f'{len(results)} detections')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
