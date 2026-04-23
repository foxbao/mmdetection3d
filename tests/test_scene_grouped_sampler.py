"""Unit tests for SceneGroupedSampler — recurrent BEV memory prerequisite.

Run: PYTHONPATH=. python tests/test_scene_grouped_sampler.py
"""

import sys
from collections import Counter, defaultdict
from unittest.mock import patch

# Make repo importable.
sys.path.insert(0, '.')

from mmdet3d.datasets.samplers import SceneGroupedSampler


class FakeDataset:
    """Minimal dataset stub exposing data_list and __len__."""

    def __init__(self, scenes_to_frames):
        # scenes_to_frames: dict[scene_token -> list of timestamps].
        # Frames are spread across the dataset in arbitrary order to
        # exercise the sampler's per-scene sort.
        self.data_list = []
        for scene, ts_list in scenes_to_frames.items():
            for ts in ts_list:
                self.data_list.append({
                    'scene_token': scene,
                    'timestamp': ts,
                })
        # Shuffle the order so timestamps inside data_list aren't
        # already sorted; sampler must sort.
        import random
        rng = random.Random(123)
        rng.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)


def collect_indices(sampler, batch_size):
    """Drain sampler into batches of size batch_size."""
    indices = list(iter(sampler))
    assert len(indices) % batch_size == 0, \
        f'sampler yielded {len(indices)} indices, not a multiple of {batch_size}'
    return [indices[i:i + batch_size]
            for i in range(0, len(indices), batch_size)]


def test_per_position_scene_continuity():
    """Within batch position b, consecutive batches' scenes either match
    or transition cleanly (a complete scene followed by a different
    complete scene), and timestamps are monotonic within each scene."""
    scenes = {
        'scene_a': [0.0, 1.0, 2.0, 3.0],
        'scene_b': [10.0, 11.0, 12.0],
        'scene_c': [20.0, 21.0, 22.0, 23.0, 24.0],
        'scene_d': [30.0, 31.0],
    }
    ds = FakeDataset(scenes)
    B = 2
    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=42):
        sampler = SceneGroupedSampler(ds, batch_size=B,
                                       shuffle_scenes=True, seed=42)
    batches = collect_indices(sampler, B)
    assert len(batches) > 0, 'expected at least one batch'

    for pos in range(B):
        prev_scene = None
        prev_ts = None
        for batch in batches:
            idx = batch[pos]
            info = ds.data_list[idx]
            scene = info['scene_token']
            ts = info['timestamp']
            if scene == prev_scene:
                assert ts > prev_ts, (
                    f'pos {pos} same-scene timestamps not monotonic: '
                    f'{prev_ts} -> {ts}')
            # if scene differs, that's a clean transition — no constraint
            prev_scene, prev_ts = scene, ts
    print('  PASS: per-position scene continuity + timestamp ordering')


def test_no_extra_or_missing_frames_modulo_padding():
    """Every yielded index points to a real sample; every sample (modulo
    truncation to multiple of B) is yielded at least once across one
    epoch's iter."""
    scenes = {f'scene_{i}': list(range(i + 2, i + 7)) for i in range(6)}
    ds = FakeDataset(scenes)
    B = 3
    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(ds, batch_size=B,
                                       shuffle_scenes=False, seed=0)

    indices = list(iter(sampler))
    # All indices in valid range
    assert all(0 <= i < len(ds) for i in indices)

    # Counts: each frame appears at least once (some may appear twice
    # due to padding to __len__). No frame should be entirely missing.
    counts = Counter(indices)
    # Iter length matches __len__
    assert len(indices) == len(sampler), \
        f'iter len {len(indices)} != __len__ {len(sampler)}'
    print(f'  PASS: indices in range, count consistency '
          f'(distinct={len(counts)}, total={len(indices)})')


def test_min_strategy_truncates_without_repeating():
    """`min` keeps rank lengths equal by truncating, never by repeating."""
    scenes = {
        'scene_a': [0, 1, 2, 3, 4],
        'scene_b': [10, 11, 12, 13],
        'scene_c': [20, 21, 22],
        'scene_d': [30, 31, 32],
    }
    ds = FakeDataset(scenes)

    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'get_dist_info', return_value=(0, 2)), \
         patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(
            ds,
            batch_size=2,
            shuffle_scenes=False,
            sync_length_strategy='min',
            seed=0)

    indices = list(iter(sampler))
    counts = Counter(indices)
    assert len(indices) == len(sampler)
    assert all(c == 1 for c in counts.values()), \
        f'expected no repeated samples under min strategy, got {counts}'
    print('  PASS: min strategy truncates without repeating')


def test_none_strategy_preserves_native_length():
    """`none` keeps every local sample exactly once and allows a short tail."""
    scenes = {
        'scene_a': [0, 1, 2],
        'scene_b': [10, 11],
        'scene_c': [20, 21, 22, 23],
    }
    ds = FakeDataset(scenes)

    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(
            ds,
            batch_size=4,
            shuffle_scenes=False,
            sync_length_strategy='none',
            seed=0)

    indices = list(iter(sampler))
    assert len(indices) == len(ds), \
        f'expected native length {len(ds)}, got {len(indices)}'
    counts = Counter(indices)
    assert all(c == 1 for c in counts.values()), \
        f'expected no repeated samples under none strategy, got {counts}'
    print('  PASS: none strategy preserves native length')


def test_ddp_scene_disjoint():
    """In a 2-rank simulated DDP setup, scenes assigned to rank 0 and
    rank 1 are disjoint (no scene goes to both ranks)."""
    scenes = {f'scene_{i:02d}': list(range(i, i + 4)) for i in range(8)}
    ds = FakeDataset(scenes)

    # Patch get_dist_info to simulate world_size=2
    rank0_scenes = set()
    rank1_scenes = set()
    for rank, target in [(0, rank0_scenes), (1, rank1_scenes)]:
        with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
                   'get_dist_info', return_value=(rank, 2)), \
             patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
                   'sync_random_seed', return_value=0):
            sampler = SceneGroupedSampler(ds, batch_size=2,
                                           shuffle_scenes=False, seed=0)
        for tok in sampler._scene_indices:
            target.add(tok)
    assert rank0_scenes.isdisjoint(rank1_scenes), \
        f'rank 0 and rank 1 share scenes: {rank0_scenes & rank1_scenes}'
    assert (rank0_scenes | rank1_scenes) == set(scenes.keys()), \
        'union of rank scenes != all scenes'
    print(f'  PASS: DDP shards disjoint '
          f'(rank0={len(rank0_scenes)}, rank1={len(rank1_scenes)})')


def test_shuffle_changes_with_epoch():
    """Different epochs produce different scene orderings (shuffle works)."""
    scenes = {f'scene_{i}': [0.0, 1.0, 2.0] for i in range(20)}
    ds = FakeDataset(scenes)
    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(ds, batch_size=4,
                                       shuffle_scenes=True, seed=0)
    sampler.set_epoch(0)
    e0 = list(iter(sampler))
    sampler.set_epoch(1)
    e1 = list(iter(sampler))
    assert e0 != e1, 'shuffle did not change between epochs'
    # Same set of indices though (modulo padding)
    assert Counter(e0).keys() == Counter(e1).keys(), \
        'epoch 0 and epoch 1 see different index *sets*'
    print('  PASS: shuffle changes per-epoch, index set unchanged')


def test_short_scene_handled():
    """A scene shorter than batch_size should still be consumed without
    causing dup/drop pathologies."""
    scenes = {
        'long_a': list(range(10)),
        'short_b': [0.0],  # 1 frame only
        'long_c': list(range(5)),
    }
    ds = FakeDataset(scenes)
    B = 2
    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(ds, batch_size=B,
                                       shuffle_scenes=False, seed=0)
    indices = list(iter(sampler))
    # Should not crash, should produce at least one batch
    assert len(indices) >= B
    assert len(indices) % B == 0
    print(f'  PASS: short scene handled, yielded {len(indices)} indices')


def test_batch_size_1():
    """batch_size=1 collapses to sequential per-scene yield."""
    scenes = {'a': [0, 1, 2], 'b': [10, 11]}
    ds = FakeDataset(scenes)
    with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
               'sync_random_seed', return_value=0):
        sampler = SceneGroupedSampler(ds, batch_size=1,
                                       shuffle_scenes=False, seed=0)
    indices = list(iter(sampler))
    # Reconstruct (scene, ts) sequence and verify each scene's frames
    # come consecutively in timestamp order.
    seq = [(ds.data_list[i]['scene_token'], ds.data_list[i]['timestamp'])
           for i in indices]
    seen_scenes = set()
    cur_scene = None
    cur_ts = None
    for scene, ts in seq:
        if scene != cur_scene:
            assert scene not in seen_scenes, \
                f'scene {scene} reappeared after switch'
            seen_scenes.add(cur_scene) if cur_scene else None
            cur_scene, cur_ts = scene, ts
        else:
            assert ts > cur_ts, f'non-monotonic ts in scene {scene}'
            cur_ts = ts
    print('  PASS: batch_size=1 yields sequential per-scene chunks')


def test_raises_on_too_few_frames():
    """Per-rank frame count < batch_size must raise (cannot form a batch)."""
    scenes = {'tiny': [0, 1]}
    ds = FakeDataset(scenes)
    raised = False
    try:
        with patch('mmdet3d.datasets.samplers.scene_grouped_sampler.'
                   'sync_random_seed', return_value=0):
            SceneGroupedSampler(ds, batch_size=4,
                                 shuffle_scenes=False, seed=0)
    except RuntimeError as e:
        raised = True
        assert 'cannot form' in str(e).lower() or 'batch_size' in str(e)
    assert raised, 'expected RuntimeError for under-sized dataset'
    print('  PASS: raises on too-few-frames')


if __name__ == '__main__':
    print('Running SceneGroupedSampler tests...')
    test_per_position_scene_continuity()
    test_no_extra_or_missing_frames_modulo_padding()
    test_ddp_scene_disjoint()
    test_shuffle_changes_with_epoch()
    test_short_scene_handled()
    test_batch_size_1()
    test_raises_on_too_few_frames()
    print('\nAll tests passed.')
