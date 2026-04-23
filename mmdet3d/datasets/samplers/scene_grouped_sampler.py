"""Sampler for recurrent BEV memory training.

Yields indices arranged so that, when the DataLoader chunks them into
batches of ``batch_size``, each *position within the batch* advances
through one scene's frames in temporal order. Concretely, with
``batch_size=B``:

    iter 0:  [scene_A frame_0, scene_B frame_0, ..., scene_X frame_0]
    iter 1:  [scene_A frame_1, scene_B frame_1, ..., scene_X frame_1]
    ...
    iter K:  [scene_A frame_K, scene_B frame_K, ..., scene_X frame_K]

When a scene exhausts at batch position b, that position picks up a new
scene from the pool, which causes the model's recurrent BEV cache for
that batch slot to miss (cold start) — the intended behavior.

This is the data-side prerequisite for BEVFormer/UniAD-style recurrent
``prev_bev`` memory: without it, consecutive batches at the same batch
slot have no temporal relation and the cache is meaningless.
"""

import math
from collections import defaultdict, deque
from typing import Iterator, List, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.logging import MMLogger
from torch.utils.data import Sampler

from mmdet3d.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SceneGroupedSampler(Sampler):
    """Group-by-scene round-robin sampler for recurrent temporal training.

    Args:
        dataset: Source dataset. Each ``dataset.data_list[i]`` (or its
            ``get_data_info(i)``) must contain ``scene_token`` (str) and
            ``timestamp`` (numeric, sortable within a scene).
        batch_size: Per-rank batch size B. Must equal the DataLoader's
            ``batch_size``. Determines how many parallel scene tracks the
            sampler maintains.
        shuffle_scenes: Whether to shuffle scene order each epoch.
            Within a scene, frames are *always* yielded in timestamp
            order. Defaults to True.
        sync_length_strategy: How to reconcile per-rank length mismatch.
            ``'max'`` pads lighter ranks by repeating earlier samples;
            ``'min'`` truncates heavier ranks to the lightest full-batch
            shard; ``'none'`` keeps each rank's native length and allows
            the last batch to be short. Defaults to ``'max'``.
        seed: Base RNG seed. If None, uses a synchronized random seed.
            Defaults to None.

    Notes:
        * DDP: scenes are sharded across ranks deterministically. Each
          rank operates on its own scene subset; no cross-rank sync is
          required for the recurrent cache (each rank holds its own).
        * Length: ``sync_length_strategy='max'`` keeps DDP iteration
          count uniform by padding. ``'min'`` keeps it uniform by
          truncation. ``'none'`` preserves native shard length for eval.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 shuffle_scenes: bool = True,
                 sync_length_strategy: str = 'max',
                 seed: Optional[int] = None) -> None:
        if batch_size < 1:
            raise ValueError(f'batch_size must be >= 1, got {batch_size}')
        if sync_length_strategy not in ('max', 'min', 'none'):
            raise ValueError(
                'sync_length_strategy must be one of '
                f"('max', 'min', 'none'), got {sync_length_strategy!r}")

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_scenes = shuffle_scenes
        self.sync_length_strategy = sync_length_strategy
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        # Group sample indices by scene_token, sorting within scene by
        # timestamp. We read raw infos via data_list (no I/O) when
        # available, falling back to get_data_info for wrapped datasets.
        scene_to_pairs: dict = defaultdict(list)
        for idx in range(len(dataset)):
            info = self._get_raw_info(dataset, idx)
            scene = info.get('scene_token', '')
            timestamp = info.get('timestamp', idx)
            scene_to_pairs[scene].append((idx, timestamp))

        if '' in scene_to_pairs and len(scene_to_pairs) == 1:
            MMLogger.get_current_instance().warning(
                'SceneGroupedSampler: no scene_token found in any sample; '
                'all frames fall into one bucket and recurrent memory will '
                'effectively see one giant scene. Check pkl converter.')

        for tok in scene_to_pairs:
            scene_to_pairs[tok].sort(key=lambda pair: pair[1])

        # DDP shard scenes deterministically: rank r owns scenes
        # all_tokens[r::world_size]. Sorting by token gives reproducible
        # assignment across runs.
        all_tokens = sorted(scene_to_pairs.keys())
        my_tokens = all_tokens[rank::world_size]
        self._scene_indices: dict = {
            tok: [pair[0] for pair in scene_to_pairs[tok]]
            for tok in my_tokens
        }

        per_rank_totals = [
            sum(len(scene_to_pairs[t]) for t in all_tokens[r::world_size])
            for r in range(world_size)
        ]
        local_total = per_rank_totals[rank]
        if local_total == 0:
            raise RuntimeError(
                f'SceneGroupedSampler on rank {rank} has no frames.')
        if sync_length_strategy in ('max', 'min') and local_total < batch_size:
            raise RuntimeError(
                f'SceneGroupedSampler on rank {rank} has '
                f'{local_total} frames < batch_size={batch_size}; '
                f'cannot form a single batch. Check scene sharding or '
                f'reduce world_size.')

        if sync_length_strategy == 'max':
            # Equal across ranks so DDP doesn't deadlock on AllReduce.
            # Lighter ranks pad from the start in __iter__.
            max_per_rank = max(per_rank_totals)
            self._num_samples = (
                math.ceil(max_per_rank / batch_size) * batch_size)
            if self._num_samples == 0:
                raise RuntimeError(
                    f'SceneGroupedSampler: max per-rank total {max_per_rank} '
                    f'with batch_size={batch_size} yields zero samples.')
        elif sync_length_strategy == 'min':
            # Equal across ranks without repeating samples: heavier ranks
            # truncate their tail to the lightest shard rounded down to a
            # full batch. This is safer for recurrent prev_bev training.
            min_per_rank = min(per_rank_totals)
            self._num_samples = (
                math.floor(min_per_rank / batch_size) * batch_size)
            if self._num_samples == 0:
                raise RuntimeError(
                    f'SceneGroupedSampler: min per-rank total {min_per_rank} '
                    f'with batch_size={batch_size} yields zero samples.')
        else:
            # Evaluation mode: preserve native shard length, allow the last
            # batch to be smaller, and never repeat samples.
            self._num_samples = local_total

    @staticmethod
    def _get_raw_info(dataset, idx: int) -> dict:
        """Get the raw info dict without running the data pipeline."""
        if hasattr(dataset, 'data_list') and idx < len(dataset.data_list):
            return dataset.data_list[idx]
        # Fallback: parsed info (slower but always works).
        return dataset.get_data_info(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Per-epoch deterministic shuffle of scene order.
        keys = list(self._scene_indices.keys())
        if self.shuffle_scenes:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(keys), generator=g).tolist()
            keys = [keys[i] for i in order]

        scene_iter = iter(keys)
        tracks: List[deque] = [deque() for _ in range(self.batch_size)]

        # Prime each track with one scene.
        for b in range(self.batch_size):
            tok = next(scene_iter, None)
            if tok is not None:
                tracks[b].extend(self._scene_indices[tok])

        # Phase 1 — scene-streaming round-robin: at each round refill any
        # empty track with the next scene and emit one frame per track.
        # Stops as soon as some track cannot be refilled, leaving
        # leftover frames in the longer tracks.
        indices: List[int] = []
        while True:
            for b in range(self.batch_size):
                while not tracks[b]:
                    tok = next(scene_iter, None)
                    if tok is None:
                        break
                    tracks[b].extend(self._scene_indices[tok])
            if not all(tracks):
                break
            for b in range(self.batch_size):
                indices.append(tracks[b].popleft())

        # Phase 2 — drain residual frames left in non-empty tracks. These
        # are appended in track-major order; they break the per-position
        # scene-streaming invariant for the very last few batches, but the
        # detector's cache is keyed by scene_token so a frame whose
        # neighbour comes from a different scene simply misses the cache
        # and cold-starts. Without this drain, val would silently lose
        # ~20% of samples on small datasets, breaking metric coverage.
        for tr in tracks:
            indices.extend(tr)

        # Phase 3 — optionally pad or truncate to the configured target
        # length. `max` pads lighter ranks for train-DDP parity; `min`
        # truncates heavier ranks to avoid sample repetition; `none`
        # preserves the native shard exactly.
        if self.sync_length_strategy == 'max' and len(indices) < self._num_samples:
            need = self._num_samples - len(indices)
            indices.extend(indices[:need])
        elif self.sync_length_strategy == 'min' and len(indices) > self._num_samples:
            indices = indices[:self._num_samples]
        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples
