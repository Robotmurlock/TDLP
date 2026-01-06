#!/usr/bin/env python3
"""CLI utility to create an MP4 from a directory of sequential JPG images."""

import argparse
import glob
import os
import cv2
from tqdm import tqdm


def parse_args():
    """Build and parse command line arguments for video creation.

    Returns:
        argparse.Namespace: Parsed arguments with image_dir, output, and fps.
    """
    parser = argparse.ArgumentParser(
        description="Create MP4 video from a directory of JPG images."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input images (e.g. 00000001.jpg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MP4 file path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    return parser.parse_args()


def create_video_from_dir(image_dir: str, output_path: str, fps: int) -> None:
    """Create an MP4 video from sequential JPG images in a directory.

    Args:
        image_dir (str): Directory containing input images (e.g. 00000001.jpg).
        output_path (str): Destination path for the generated MP4 file.
        fps (int): Frames per second for the output video.

    Raises:
        RuntimeError: If images cannot be found, read, or have mismatched sizes.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if not image_paths:
        raise RuntimeError("No JPG images found in the directory.")

    # Read first image to get resolution
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")

    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    for img_path in tqdm(image_paths, desc="Writing frames", unit="frame"):
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if img.shape[:2] != (height, width):
            raise RuntimeError(
                f"Image {img_path} has different resolution."
            )

        writer.write(img)

    writer.release()
    print(f"Video written to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    create_video_from_dir(args.image_dir, args.output, args.fps)
