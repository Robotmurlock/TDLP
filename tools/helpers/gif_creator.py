#!/usr/bin/env python3
"""CLI utility to create a GIF from frames of an MP4 video."""

import argparse
import os
from typing import List

import cv2
import imageio.v2 as imageio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Build and parse command line arguments for GIF creation.

    Returns:
        argparse.Namespace: Parsed arguments for the GIF creator CLI.
    """
    parser = argparse.ArgumentParser(
        description="Create a GIF from frames of an MP4 video."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input MP4 video path.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="Start frame index (inclusive) to include in the GIF.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="End frame index (inclusive) to include in the GIF.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Destination GIF file path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale factor to resize frames (e.g., 0.5 halves width/height).",
    )
    return parser.parse_args()


def _validate_video_range(
    capture: cv2.VideoCapture, start_index: int, end_index: int
) -> None:
    """Validate requested frame range against the video.

    Args:
        capture (cv2.VideoCapture): Open video capture object.
        start_index (int): Requested start frame (inclusive).
        end_index (int): Requested end frame (inclusive).

    Raises:
        ValueError: If the range is invalid or exceeds the video length.
    """
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError("Video contains no frames.")

    if start_index < 0 or end_index < start_index:
        raise ValueError("end_index must be >= start_index and start_index >= 0.")

    if end_index >= frame_count:
        raise ValueError(
            f"Requested end_index {end_index} exceeds last frame {frame_count - 1}."
        )


def _read_frames(
    capture: cv2.VideoCapture, start_index: int, end_index: int, scale: float
) -> List:
    """Read frames from an open capture within the provided range.

    Args:
        capture (cv2.VideoCapture): Open video capture object.
        start_index (int): Start frame index (inclusive).
        end_index (int): End frame index (inclusive).
        scale (float): Uniform resize factor applied to each frame.

    Returns:
        List: Frames converted to RGB, ordered from start_index to end_index.

    Raises:
        RuntimeError: If a frame cannot be read.
    """
    frames: List = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

    for frame_idx in tqdm(
        range(start_index, end_index + 1), desc="Reading frames", unit="frame"
    ):
        success, frame_bgr = capture.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {frame_idx}.")
        if scale != 1.0:
            new_w = max(1, int(frame_bgr.shape[1] * scale))
            new_h = max(1, int(frame_bgr.shape[0] * scale))
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return frames


def create_gif_from_video(
    input_path: str,
    start_index: int,
    end_index: int,
    output_path: str,
    frame_duration: float = 0.1,
    scale: float = 1.0,
) -> None:
    """Create a GIF from a segment of an MP4 video.

    Args:
        input_path (str): Path to the source MP4 video.
        start_index (int): Start frame index (inclusive).
        end_index (int): End frame index (inclusive).
        output_path (str): Destination GIF path.
        frame_duration (float): Duration per frame in seconds.
        scale (float): Uniform scale factor for resizing frames.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video cannot be opened or a frame cannot be read.
        ValueError: If the requested frame range is invalid or scale is non-positive.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    _validate_video_range(capture, start_index, end_index)
    frames = _read_frames(capture, start_index, end_index, scale)
    capture.release()

    imageio.mimsave(output_path, frames, duration=frame_duration)
    print(f"GIF written to: {output_path}")


def main() -> None:
    """Parse arguments and orchestrate GIF creation."""
    args = parse_args()
    create_gif_from_video(
        args.input_path,
        args.start_index,
        args.end_index,
        args.output_path,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()

