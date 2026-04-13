import argparse
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export camera-only BEVFusion weights for fusion-LSS init')
    parser.add_argument(
        'src',
        help='source camera-only checkpoint, e.g. '
        'work_dirs/bevfusion_cam_swint_lss_kl/epoch_5.pth')
    parser.add_argument(
        'dst',
        help='output checkpoint for fusion init, e.g. '
        'work_dirs/bevfusion_cam_swint_lss_kl/epoch_5_cam_init_for_fusion_lss.pth'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    keep_prefixes = (
        'img_backbone.',
        'img_neck.',
        'view_transform.depthnet.',
        'view_transform.downsample.',
    )
    drop_exact_keys = {
        'view_transform.dx',
        'view_transform.bx',
        'view_transform.nx',
        'view_transform.frustum',
    }

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key in drop_exact_keys:
            continue
        if key.startswith(keep_prefixes):
            new_state_dict[key] = value

    out = {}
    if isinstance(ckpt, dict):
        # Keep lightweight metadata when present, but replace the state_dict.
        out.update({
            k: v
            for k, v in ckpt.items() if k != 'state_dict'
        })
    out['state_dict'] = new_state_dict

    torch.save(out, args.dst)

    print(f'saved {len(new_state_dict)} params to {args.dst}')
    for key in sorted(new_state_dict.keys())[:50]:
        print(key)


if __name__ == '__main__':
    main()
