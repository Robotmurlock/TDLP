"""
Visualize bounding-box detections produced by od_inference.py.

Usage
-----
Single image:
    python tools/helpers/od_visualize.py \\
        --input path/to/image.jpg \\
        --detections detections.json \\
        --output vis.jpg

Directory of frames → video:
    python tools/helpers/od_visualize.py \\
        --input path/to/frames/ \\
        --detections detections.json \\
        --output vis.mp4 \\
        --fps 20
"""
import argparse
import json
import os

import cv2
import numpy as np
from motrack.library.cv.video_writer import MP4Writer

_COLOUR = (0, 255, 0)
_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5


def _draw(image: np.ndarray, detections: list) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det['bbox_xyxy']
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(vis, (px1, py1), (px2, py2), _COLOUR, _THICKNESS)
        label = f"cls{det['class']} {det['conf']:.2f}"
        cv2.putText(vis, label, (px1, max(py1 - 4, 0)), _FONT, _FONT_SCALE,
                    _COLOUR, 1, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize OD detections')
    parser.add_argument('--input', required=True, help='Image file or directory of frames')
    parser.add_argument('--detections', required=True, help='JSON produced by od_inference.py')
    parser.add_argument('--output', required=True, help='Output image (.jpg/.png) or video (.mp4)')
    parser.add_argument('--fps', type=float, default=20.0, help='FPS for video output (default: 20)')
    args = parser.parse_args()

    with open(args.detections, 'r') as f:
        data = json.load(f)

    if os.path.isdir(args.input):
        if not isinstance(data, dict):
            raise ValueError('For directory mode the detections JSON must be a dict keyed by filename.')

        fnames = sorted(data.keys())
        if not fnames:
            raise ValueError('No frames found in detections JSON.')

        with MP4Writer(args.output, fps=int(args.fps)) as writer:
            for fname in fnames:
                fpath = os.path.join(args.input, fname)
                image = cv2.imread(fpath)
                if image is None:
                    print(f'Warning: could not read {fpath}, skipping')
                    continue
                writer.write(_draw(image, data[fname]))
        print(f'Video saved → {args.output}  ({len(fnames)} frames @ {args.fps} fps)')

    else:
        image = cv2.imread(args.input)
        if image is None:
            raise ValueError(f'Could not read image: {args.input}')
        if not isinstance(data, list):
            raise ValueError('For single-image mode the detections JSON must be a list.')
        cv2.imwrite(args.output, _draw(image, data))
        print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
