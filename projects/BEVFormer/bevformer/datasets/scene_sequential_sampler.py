"""Scene-wise sequential sampler for temporal evaluation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmdet3d.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SceneSequentialSampler(Sampler):
    """Sample complete scenes on each rank while preserving frame order.

    ``DefaultSampler(shuffle=False)`` shards evaluation data as
    ``rank, rank + world_size, ...``. That is fine for single-frame detectors,
    but it breaks online temporal models because adjacent frames in the same
    scene land on different ranks. This sampler assigns whole scenes to ranks,
    then yields frames in timestamp order inside each scene.

    Args:
        dataset: Dataset to sample from.
        shuffle: Whether to shuffle scenes before assignment. Defaults to
            False. Frames inside each scene are never shuffled.
        seed: Random seed for scene shuffling. Defaults to a synchronized seed.
        balance: Whether to greedily balance scene counts across ranks.
            Defaults to True.
    """

    def __init__(self,
                 dataset,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 balance: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = sync_random_seed() if seed is None else seed
        self.balance = balance
        self.epoch = 0

        self._scenes = self._build_scenes()
        self._num_samples = len(self._rank_indices(self._scenes))

    def _build_scenes(self) -> List[Tuple[str, List[int]]]:
        if hasattr(self.dataset, 'full_init'):
            self.dataset.full_init()

        scene_frames = OrderedDict()
        for idx in range(len(self.dataset)):
            info = self.dataset.get_data_info(idx)
            scene_token = info.get('scene_token', info.get('token', str(idx)))
            timestamp = info.get('timestamp', idx)
            scene_frames.setdefault(scene_token, []).append((timestamp, idx))

        scenes = []
        for scene_token, frames in scene_frames.items():
            frames.sort(key=lambda item: item[0])
            scenes.append((scene_token, [idx for _, idx in frames]))
        return scenes

    def _maybe_shuffle_scenes(
            self,
            scenes: Sequence[Tuple[str, List[int]]]
    ) -> List[Tuple[str, List[int]]]:
        scenes = list(scenes)
        if not self.shuffle:
            return scenes

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(scenes), generator=generator).tolist()
        return [scenes[i] for i in order]

    def _rank_indices(
            self,
            scenes: Sequence[Tuple[str, List[int]]]
    ) -> List[int]:
        scenes = self._maybe_shuffle_scenes(scenes)

        if self.world_size == 1:
            return [idx for _, indices in scenes for idx in indices]

        rank_scene_indices = [[] for _ in range(self.world_size)]
        rank_lengths = [0 for _ in range(self.world_size)]

        if self.balance:
            # Balance by whole scenes only; frame order inside each scene stays
            # untouched.
            scene_order = sorted(
                range(len(scenes)),
                key=lambda i: len(scenes[i][1]),
                reverse=True)
            for scene_idx in scene_order:
                target_rank = min(
                    range(self.world_size), key=lambda r: rank_lengths[r])
                rank_scene_indices[target_rank].append(scene_idx)
                rank_lengths[target_rank] += len(scenes[scene_idx][1])
        else:
            for scene_idx in range(len(scenes)):
                rank_scene_indices[scene_idx % self.world_size].append(
                    scene_idx)

        indices = []
        for scene_idx in rank_scene_indices[self.rank]:
            indices.extend(scenes[scene_idx][1])
        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._rank_indices(self._scenes))

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._num_samples = len(self._rank_indices(self._scenes))
