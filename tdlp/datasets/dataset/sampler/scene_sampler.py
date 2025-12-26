import math
import random
from typing import Dict, Iterator, List, Sequence, Hashable

from torch.utils.data import Sampler
from tdlp.datasets.dataset.mot import MOTClipDataset


def _group_indices_by_scene(scene_ids: Sequence[str]) -> Dict[str, List[int]]:
    by_scene: Dict[str, List[int]] = {}
    for indices, s in enumerate(scene_ids):
        by_scene.setdefault(s, []).append(indices)
    return by_scene


class SceneBatchSamplerNoRepeat(Sampler[List[int]]):
    """BatchSampler: N scenes × up to M frames/scene, no frame repeats.

    Each batch contains frames chunked per scene (distinct scenes per batch).
    All dataset indices are yielded exactly once per epoch.
    The final batch can be smaller than N*M (and some scenes may contribute < M).

    Args:
        scene_ids: scene id per dataset index (len == len(dataset)).
        n_scenes: number of distinct scenes per batch (N).
        n_frames: number of frames per scene in a batch (M).
        shuffle_scenes: shuffle scene order each epoch.
        shuffle_frames: shuffle frames *within* each scene each epoch.
        seed: base RNG seed.
    """
    def __init__(
        self,
        scene_ids: List[str],
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        shuffle_frames: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._scene_ids = scene_ids
        self._n_scenes = n_scenes
        self._n_frames = n_frames
        self._shuffle_scenes = shuffle_scenes
        self._shuffle_frames = shuffle_frames
        self._seed = seed
        self._iteration = 0
        self._by_scene = _group_indices_by_scene(scene_ids)

        # Precompute an upper-bound on __len__
        total_chunks = 0
        for indices in self._by_scene.values():
            total_chunks += math.ceil(len(indices) / self._n_frames)
        self._len = math.ceil(total_chunks / self._n_scenes)

    @classmethod
    def from_dataset(
        cls,
        dataset: MOTClipDataset,
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        shuffle_frames: bool = True,
        seed: int = 0,
    ):
        return cls(
            scene_ids=dataset.scene_names_per_frame,
            n_scenes=n_scenes,
            n_frames=n_frames,
            shuffle_scenes=shuffle_scenes,
            shuffle_frames=shuffle_frames,
            seed=seed,
        )

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self._seed + self._iteration)
        self._iteration += 1

        # Build per-scene chunks (size M, last may be smaller)
        per_scene_chunks: Dict[str, List[List[int]]] = {}
        scene_keys = list(self._by_scene.keys())
        if self._shuffle_scenes:
            rng.shuffle(scene_keys)

        for s in scene_keys:
            indices = self._by_scene[s][:]
            if self._shuffle_frames:
                rng.shuffle(indices)
            chunks = [indices[i:i + self._n_frames] for i in range(0, len(indices), self._n_frames)]
            per_scene_chunks[s] = chunks

        # Round-robin: at each step, take up to N scenes that still have chunks
        active = [s for s in scene_keys if per_scene_chunks[s]]
        while active:
            if self._shuffle_scenes:
                rng.shuffle(active)
            batch_scenes = active[: self._n_scenes]
            batch: List[int] = []
            depleted: List[str] = []
            for s in batch_scenes:
                chunk = per_scene_chunks[s].pop(0)
                batch.extend(chunk)
                if not per_scene_chunks[s]:
                    depleted.append(s)
            # remove depleted scenes
            active = [s for s in active if s not in depleted]
            yield batch


class SceneBatchSamplerWithRepeat(Sampler[List[int]]):
    """BatchSampler: N scenes × exactly M frames/scene, repeats allowed.

    Guarantees all frames are seen at least once per epoch. If scenes don't
    divide evenly, some frames will repeat to keep batches full.

    Args:
        scene_ids: scene id per dataset index (len == len(dataset)).
        n_scenes: number of distinct scenes per batch (N).
        n_frames: number of frames per scene in a batch (M).
        shuffle_scenes: shuffle scene order each epoch.
        shuffle_frames: shuffle frames *within* each scene each epoch.
        seed: base RNG seed.
    """
    def __init__(
        self,
        scene_ids: List[str],
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        shuffle_frames: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._scene_ids = scene_ids
        self._n_scenes = n_scenes
        self._n_frames = n_frames
        self._shuffle_scenes = shuffle_scenes
        self._shuffle_frames = shuffle_frames
        self._seed = seed
        self._iteration = 0
        self._by_scene = _group_indices_by_scene(scene_ids)

        # How many scene-slots do we minimally need to cover all frames?
        self._scene_needed = {
            s: math.ceil(len(indices) / self._n_frames) for s, indices in self._by_scene.items()
        }
        total_slots = sum(self._scene_needed.values())
        # Number of batches = ceil(total_slots / N)
        self._num_batches = math.ceil(total_slots / self._n_scenes)
        # Total slots actually used (padding to multiple of N)
        self._total_slots_padded = self._num_batches * self._n_scenes

    @classmethod
    def from_dataset(
        cls,
        dataset: MOTClipDataset,
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        shuffle_frames: bool = True,
        seed: int = 0,
    ):
        return cls(
            scene_ids=dataset.scene_names_per_frame,
            n_scenes=n_scenes,
            n_frames=n_frames,
            shuffle_scenes=shuffle_scenes,
            shuffle_frames=shuffle_frames,
            seed=seed,
        )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self._seed + self._iteration)
        self._iteration += 1

        # Shuffle frames within each scene and set pointers
        per_scene_order: Dict[str, List[int]] = {}
        ptr: Dict[Hashable, int] = {}
        for s, indices in self._by_scene.items():
            order = indices[:]
            if self._shuffle_frames:
                rng.shuffle(order)
            per_scene_order[s] = order
            ptr[s] = 0

        # Build a multiset of scene assignments: each scene appears `needed` times
        scene_assignments: List[str] = []
        for s, k in self._scene_needed.items():
            scene_assignments.extend([s] * k)

        # Pad to multiple of N with extra scenes (these will cause repeats)
        if len(scene_assignments) < self._total_slots_padded:
            pool = list(self._by_scene.keys())
            while len(scene_assignments) < self._total_slots_padded:
                scene_assignments.append(rng.choice(pool))

        if self._shuffle_scenes:
            rng.shuffle(scene_assignments)

        # Emit batches: N scenes per batch, M frames per scene (cyclic)
        for b in range(self._num_batches):
            batch: List[int] = []
            start = b * self._n_scenes
            for s in scene_assignments[start : start + self._n_scenes]:
                order = per_scene_order[s]
                n = len(order)
                # Take M frames, wrapping as needed
                take: List[int] = []
                for _ in range(self._n_frames):
                    take.append(order[ptr[s] % n])
                    ptr[s] += 1
                batch.extend(take)
            yield batch


class OneSceneWithRangeSampler(Sampler[List[int]]):
    """BatchSampler: N scenes × exactly M consecutive frames/scene.

    Each batch contains N different scenes with M consecutive frames per scene.
    Frames within each scene must be consecutive (no gaps).
    Scenes can be sampled multiple times across different batches.

    Args:
        scene_ids: scene id per dataset index (len == len(dataset)).
        n_scenes: number of distinct scenes per batch (N).
        n_frames: number of consecutive frames per scene in a batch (M).
        shuffle_scenes: shuffle scene order each epoch.
        seed: base RNG seed.
    """
    def __init__(
        self,
        scene_ids: List[str],
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._scene_ids = scene_ids
        self._n_scenes = n_scenes
        self._n_frames = n_frames
        self._shuffle_scenes = shuffle_scenes
        self._seed = seed
        self._iteration = 0
        self._by_scene = _group_indices_by_scene(scene_ids)

        # Filter out scenes that don't have enough consecutive frames
        self._valid_scenes = {}
        for scene_id, indices in self._by_scene.items():
            if len(indices) >= self._n_frames:
                # Sort indices to ensure we can find consecutive sequences
                sorted_indices = sorted(indices)
                self._valid_scenes[scene_id] = sorted_indices

        # Calculate number of batches based on total possible samples
        # Each scene can contribute multiple samples (consecutive frame ranges)
        total_samples = 0
        for scene_id, indices in self._valid_scenes.items():
            # Number of possible consecutive ranges of length n_frames
            num_ranges = len(indices) - self._n_frames + 1
            total_samples += num_ranges
        
        self._num_batches = math.ceil(total_samples / self._n_scenes)

    @classmethod
    def from_dataset(
        cls,
        dataset: MOTClipDataset,
        n_scenes: int,
        n_frames: int,
        shuffle_scenes: bool = True,
        seed: int = 0,
    ):
        return cls(
            scene_ids=dataset.scene_names_per_frame,
            n_scenes=n_scenes,
            n_frames=n_frames,
            shuffle_scenes=shuffle_scenes,
            seed=seed,
        )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self._seed + self._iteration)
        self._iteration += 1

        # Generate all possible consecutive frame ranges for each scene
        all_samples = []
        for scene_id, indices in self._valid_scenes.items():
            for start_idx in range(len(indices) - self._n_frames + 1):
                consecutive_frames = indices[start_idx:start_idx + self._n_frames]
                all_samples.append((scene_id, consecutive_frames))

        # Shuffle all samples
        rng.shuffle(all_samples)

        # Group samples into batches, ensuring each scene appears only once per batch
        batch_samples = []
        used_scenes_in_batch = set()
        
        for scene_id, frames in all_samples:
            # If we already have this scene in current batch, skip
            if scene_id in used_scenes_in_batch:
                continue
                
            batch_samples.append((scene_id, frames))
            used_scenes_in_batch.add(scene_id)
            
            # When we have enough scenes for a batch, yield it
            if len(batch_samples) >= self._n_scenes:
                batch: List[int] = []
                for _, frames in batch_samples:
                    batch.extend(frames)
                yield batch
                
                # Reset for next batch
                batch_samples = []
                used_scenes_in_batch = set()
        
        # Yield remaining samples as final batch if any
        if batch_samples:
            batch: List[int] = []
            for _, frames in batch_samples:
                batch.extend(frames)
            yield batch


def run_test_batch_sampler_without_repeat() -> None:
    scene_ids = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'D', 'D']
    batch_sampler = SceneBatchSamplerNoRepeat(
        scene_ids=scene_ids,
        n_scenes=3,
        n_frames=2
    )

    sampled_scene_ids = [[scene_ids[b_i] for b_i in batch] for batch in list(batch_sampler)]
    print(sampled_scene_ids)


def run_test_batch_sampler_with_repeat() -> None:
    scene_ids = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'D', 'D']
    batch_sampler = SceneBatchSamplerWithRepeat(
        scene_ids=scene_ids,
        n_scenes=3,
        n_frames=2
    )

    sampled_scene_ids = [[scene_ids[b_i] for b_i in batch] for batch in list(batch_sampler)]
    print(sampled_scene_ids)


def run_test_one_scene_with_range_sampler() -> None:
    """Test the OneSceneWithRangeSampler with consecutive frame sampling."""
    # Create test data with consecutive indices per scene
    scene_ids = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D']
    batch_sampler = OneSceneWithRangeSampler(
        scene_ids=scene_ids,
        n_scenes=2,
        n_frames=3
    )

    print("OneSceneWithRangeSampler test:")
    print(f"Scene distribution: {dict(_group_indices_by_scene(scene_ids))}")
    print(f"Valid scenes: {list(batch_sampler._valid_scenes.keys())}")
    print(f"Total batches: {len(batch_sampler)}")
    
    for i, batch in enumerate(batch_sampler):
        batch_scene_ids = [scene_ids[b_i] for b_i in batch]
        print(f"Batch {i}: indices={batch}, scenes={batch_scene_ids}")
        
        # Verify consecutive frames within each scene
        scene_groups = {}
        for idx, scene_id in zip(batch, batch_scene_ids):
            if scene_id not in scene_groups:
                scene_groups[scene_id] = []
            scene_groups[scene_id].append(idx)
        
        for scene_id, indices in scene_groups.items():
            # Check if indices are consecutive
            sorted_indices = sorted(indices)
            is_consecutive = all(sorted_indices[i] == sorted_indices[i-1] + 1 
                               for i in range(1, len(sorted_indices)))
            print(f"  Scene {scene_id}: indices={sorted_indices}, consecutive={is_consecutive}")


if __name__ == '__main__':
    run_test_batch_sampler_without_repeat()
    run_test_batch_sampler_with_repeat()
    run_test_one_scene_with_range_sampler()
