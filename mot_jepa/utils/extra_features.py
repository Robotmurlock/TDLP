import pickle
import os
from pathlib import Path
from typing import List


class ExtraFeaturesWriter:
    def __init__(self, path: str):
        self._path = path

    def write(self, scene_name: str, frame_index: int, data: List[dict]) -> None:
        scene_path = os.path.join(self._path, scene_name)
        Path(scene_path).mkdir(parents=True, exist_ok=True)
        frame_path = os.path.join(scene_path, f'{frame_index:06d}.png')
        with open(frame_path, 'wb') as f:
            pickle.dump(data, f)


class ExtraFeaturesReader:
    def __init__(self, path: str):
        self._path = path

    def read(self, scene_name: str, frame_index: int) -> List[dict]:
        scene_path = os.path.join(self._path, scene_name)
        frame_path = os.path.join(scene_path, f'{frame_index:06d}.png')
        with open(frame_path, 'rb') as f:
            return pickle.load(f)