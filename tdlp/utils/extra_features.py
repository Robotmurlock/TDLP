"""
Read and write extra feature pickle files.
This is used to store features for offline (tools/inference.py) tracking.
"""
import os
import pickle
from pathlib import Path
from typing import List


class ExtraFeaturesWriter:
    """
    Write extra feature pickle files.
    """
    def __init__(self, path: str):
        """
        Args:
            path: Path to the extra features directory.
        """
        self._path = path

    def write(self, scene_name: str, frame_index: int, data: List[dict]) -> None:
        """
        Write extra feature pickle files.

        Args:
            scene_name: Name of the scene.
            frame_index: Index of the frame.
            data: List of dictionaries containing the extra features.
        """
        scene_path = os.path.join(self._path, scene_name)
        Path(scene_path).mkdir(parents=True, exist_ok=True)
        frame_path = os.path.join(scene_path, f'{frame_index:06d}.pkl')
        with open(frame_path, 'wb') as f:
            pickle.dump(data, f)


class ExtraFeaturesReader:
    """
    Read extra feature pickle files.
    """
    def __init__(self, path: str):
        """
        Args:
            path: Path to the extra features directory.
        """
        self._path = path

    def read(self, scene_name: str, frame_index: int) -> List[dict]:
        """
        Read extra feature pickle files.

        Args:
            scene_name: Name of the scene.
            frame_index: Index of the frame.
        """
        scene_path = os.path.join(self._path, scene_name)
        frame_path = os.path.join(scene_path, f'{frame_index:06d}.pkl')
        with open(frame_path, 'rb') as f:
            return pickle.load(f)
