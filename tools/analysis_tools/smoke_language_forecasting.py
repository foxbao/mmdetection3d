"""Run a tiny train-step smoke test for language-conditioned forecasting.

This intentionally builds the full KL temporal dataset first, then wraps it in
``torch.utils.data.Subset``. That preserves the token lookup table used by
``KlBEVFormerDataset`` to follow ``prev`` links, while still running only a few
samples.
"""

from __future__ import annotations

import argparse
from typing import List

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings
from torch.utils.data import DataLoader, Subset

from mmdet3d.registry import DATASETS, MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smoke-test language-conditioned forecasting training.')
    parser.add_argument(
        '--config',
        default=(
            'projects/BEVFormer/configs/'
            'bevformer_lidar_kl_temporal_transfusion_language_forecasting.py'),
        help='Config file to smoke-test.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint to load. Defaults to cfg.load_from if unset.')
    parser.add_argument(
        '--indices',
        nargs='*',
        type=int,
        default=None,
        help='Raw dataset indices to run. Defaults to first valid queue items.')
    parser.add_argument(
        '--max-iters', type=int, default=2, help='Number of train steps.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Smoke dataloader batch size.')
    parser.add_argument(
        '--device', default='cuda:0', help='Torch device, e.g. cuda:0 or cpu.')
    return parser.parse_args()


def import_custom_modules(cfg: Config) -> None:
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def first_valid_indices(dataset, count: int) -> List[int]:
    if not hasattr(dataset, 'full_init'):
        return list(range(count))
    dataset.full_init()
    if not hasattr(dataset, '_collect_queue_indices'):
        return list(range(min(count, len(dataset))))

    valid = []
    raw_len = getattr(dataset, 'raw_data_len', len(dataset))
    for idx in range(raw_len):
        if dataset._collect_queue_indices(idx) is not None:
            valid.append(idx)
            if len(valid) >= count:
                break
    if len(valid) < count:
        raise RuntimeError(
            f'Only found {len(valid)} valid temporal samples, need {count}.')
    return valid


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_custom_modules(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    device = torch.device(args.device)
    model = MODELS.build(cfg.model).to(device)
    checkpoint = args.checkpoint or cfg.get('load_from', None)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu', strict=False)
    model.train()

    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    indices = args.indices or first_valid_indices(dataset, args.max_iters)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pseudo_collate)

    print(f'Using indices: {indices[:args.max_iters]}')
    for iter_idx, data_batch in enumerate(dataloader, 1):
        optim_wrapper.zero_grad()
        log_vars = model.train_step(data_batch, optim_wrapper)
        scalars = {key: float(value) for key, value in log_vars.items()}
        prompts = [
            sample.metainfo.get('language_prompt', '<missing>')
            for sample in data_batch['data_samples']
        ]
        print(f'iter {iter_idx}: {scalars}')
        print(f'prompts: {prompts}')
        if iter_idx >= args.max_iters:
            break
    print('smoke_ok')


if __name__ == '__main__':
    main()
