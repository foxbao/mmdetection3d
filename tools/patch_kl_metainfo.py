"""One-shot: add `lidar_coord_frame` to existing KL pkl metainfo.

Existing KL pkls were generated before the explicit-frame change and lack
the field. Run this once to stamp them in-place so the dataset/sanity
asserts pass without a full re-run of `tools/create_data.py kl`.

Usage:
    python tools/patch_kl_metainfo.py PKL [PKL ...] [--frame RFU|FLU]
"""
import argparse

import mmengine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', nargs='+', help='pkl file(s) to patch')
    parser.add_argument('--frame', default='FLU', choices=['RFU', 'FLU'])
    args = parser.parse_args()

    for path in args.pkl:
        data = mmengine.load(path)
        meta = data.setdefault('metainfo', data.setdefault('metadata', {}))
        old = meta.get('lidar_coord_frame')
        meta['lidar_coord_frame'] = args.frame
        mmengine.dump(data, path)
        print(f'patched {path}: lidar_coord_frame {old!r} -> {args.frame!r}')


if __name__ == '__main__':
    main()
