#!/usr/bin/env python
"""Phase 0 audit for KL forecasting GT (Stage 3 → trajectory prediction).

Checks whether `gt_forecasting_locs` / `gt_forecasting_mask` added by
`tools/dataset_converters/add_forecasting.py` are present, shaped correctly,
temporally coherent, and semantically consistent with the LiDAR coord frame.

Usage:
    python3 tools/analysis_tools/audit_kl_forecasting.py \
        --pkl data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl
"""

import argparse
import pickle
from collections import Counter, defaultdict

import numpy as np


FORECAST_STEPS = 6


def _lidar2ego(frame):
    M = np.eye(4, dtype=np.float64)
    if frame == 'RFU':
        M[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    elif frame != 'FLU':
        raise ValueError(frame)
    return M


def recompute_displacement(cur_info, fut_info, tid, lidar2ego):
    """Independent re-derivation to cross-check add_forecasting.py output."""
    cur_inst = next((i for i in cur_info['instances']
                     if i.get('track_id', -1) == tid), None)
    fut_inst = next((i for i in fut_info['instances']
                     if i.get('track_id', -1) == tid), None)
    if cur_inst is None or fut_inst is None:
        return None

    e2g_cur = np.asarray(cur_info['ego2global'], dtype=np.float64)
    e2g_fut = np.asarray(fut_info['ego2global'], dtype=np.float64)
    l2g_cur = e2g_cur @ lidar2ego
    l2g_fut = e2g_fut @ lidar2ego

    cur_xy = np.asarray(cur_inst['bbox_3d'][:2])
    pos_fut_l = np.array([*fut_inst['bbox_3d'][:3], 1.0])
    pos_global = l2g_fut @ pos_fut_l
    pos_cur_l = np.linalg.inv(l2g_cur) @ pos_global
    return pos_cur_l[:2] - cur_xy


def audit(pkl_path):
    print(f'\n{"=" * 72}\n{pkl_path}\n{"=" * 72}')
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    infos = d['data_list']
    meta = d.get('metainfo', {})
    frame = meta.get('lidar_coord_frame', 'FLU')
    lidar2ego = _lidar2ego(frame)
    cats = {v: k for k, v in meta.get('categories', {}).items()}

    # 1. structural
    n = len(infos)
    scenes = {info['scene_token'] for info in infos}
    n_prev = sum(1 for i in infos if i.get('prev'))
    n_next = sum(1 for i in infos if i.get('next'))
    print(f'[1] frames={n}  scenes={len(scenes)}  lidar_coord_frame={frame}')
    print(f'    with prev: {n_prev} ({n_prev/n*100:.1f}%)'
          f'   with next: {n_next} ({n_next/n*100:.1f}%)')

    # 2. dt distribution
    token_to_info = {i['token']: i for i in infos}
    dts = []
    for info in infos:
        nxt_tok = info.get('next', '')
        if nxt_tok and nxt_tok in token_to_info:
            dts.append(token_to_info[nxt_tok]['timestamp'] - info['timestamp'])
    if dts:
        dts = np.asarray(dts)
        med = np.median(dts)
        print(f'[2] dt (s): median={med:.3f}  p10={np.percentile(dts, 10):.3f}'
              f'  p90={np.percentile(dts, 90):.3f}  horizon={FORECAST_STEPS*med:.2f}s')

    # 3. track_id coverage and lifespan
    track_frames = defaultdict(set)  # (scene, tid) -> {tokens}
    n_inst = 0
    n_tid_ok = 0
    for info in infos:
        scn = info['scene_token']
        for inst in info.get('instances', []):
            n_inst += 1
            tid = inst.get('track_id', -1)
            if tid >= 0:
                n_tid_ok += 1
                track_frames[(scn, tid)].add(info['token'])
    life = np.asarray([len(v) for v in track_frames.values()])
    print(f'[3] instances={n_inst}  with track_id: {n_tid_ok} '
          f'({n_tid_ok/n_inst*100:.1f}%)')
    print(f'    unique tracks: {len(track_frames)}   lifespan (frames):'
          f' median={np.median(life):.0f}  p90={np.percentile(life, 90):.0f}'
          f'  max={life.max()}')
    print(f'    tracks reaching full horizon ({FORECAST_STEPS+1}+ frames): '
          f'{(life >= FORECAST_STEPS + 1).sum()} '
          f'({(life >= FORECAST_STEPS + 1).mean()*100:.1f}%)')

    # 4. forecasting field presence and mask coverage
    shape_bad = 0
    missing = 0
    mask_counts = Counter()  # num_true -> count
    cls_disp = defaultdict(list)  # cls_id -> list of final-step |disp|
    for info in infos:
        for inst in info.get('instances', []):
            locs = inst.get('gt_forecasting_locs')
            mask = inst.get('gt_forecasting_mask')
            if locs is None or mask is None:
                missing += 1
                continue
            la = np.asarray(locs)
            ma = np.asarray(mask)
            if la.shape != (FORECAST_STEPS, 2) or ma.shape != (FORECAST_STEPS,):
                shape_bad += 1
                continue
            mask_counts[int(ma.sum())] += 1
            if ma[-1]:
                cls_disp[inst['bbox_label_3d']].append(float(np.linalg.norm(la[-1])))
    print(f'[4] forecasting: missing={missing}  shape_bad={shape_bad}')
    total = sum(mask_counts.values())
    print('    mask_true_steps distribution:')
    for k in range(FORECAST_STEPS + 1):
        c = mask_counts.get(k, 0)
        print(f'       {k}/{FORECAST_STEPS}: {c:>10}  ({c/total*100:5.1f}%)')

    # 5. per-class final-step displacement (should be ~0 for static classes)
    print('[5] final-step displacement |disp| by class  '
          '(mean / median / p95, N):')
    rows = []
    for cls_id, ds in cls_disp.items():
        a = np.asarray(ds)
        rows.append((cats.get(cls_id, str(cls_id)), a.mean(), np.median(a),
                     np.percentile(a, 95), len(a)))
    rows.sort(key=lambda r: -r[1])
    for name, mean, med, p95, n_ in rows:
        print(f'       {name:<22} {mean:6.2f} / {med:6.2f} / {p95:6.2f}  (N={n_})')

    # 6. independent cross-check on 50 random full-horizon tracks
    print('[6] semantic cross-check (recomputed vs stored, step=5):')
    rng = np.random.default_rng(0)
    candidates = []
    for info in infos:
        for inst in info.get('instances', []):
            ma = inst.get('gt_forecasting_mask')
            if ma is not None and np.all(ma):
                candidates.append((info['token'], inst['track_id']))
    if candidates:
        pick = rng.choice(len(candidates), size=min(50, len(candidates)),
                          replace=False)
        errs = []
        for p in pick:
            tok, tid = candidates[p]
            cur = token_to_info[tok]
            # walk next chain 6 steps
            nxt = cur
            ok = True
            for _ in range(FORECAST_STEPS):
                t = nxt.get('next', '')
                if not t or t not in token_to_info:
                    ok = False
                    break
                nxt = token_to_info[t]
            if not ok:
                continue
            recomp = recompute_displacement(cur, nxt, tid, lidar2ego)
            stored = np.asarray(next(i['gt_forecasting_locs'][-1]
                                     for i in cur['instances']
                                     if i['track_id'] == tid))
            if recomp is not None:
                errs.append(np.linalg.norm(recomp - stored))
        if errs:
            errs = np.asarray(errs)
            print(f'       N={len(errs)}  max_err={errs.max():.4f}m '
                  f' mean_err={errs.mean():.4f}m   '
                  f'({"PASS" if errs.max() < 1e-3 else "MISMATCH"})')

    # 7. scene-id collisions: is track_id unique within scene?
    scene_tid_box_ct = defaultdict(list)
    for info in infos:
        scn = info['scene_token']
        for inst in info.get('instances', []):
            tid = inst.get('track_id', -1)
            if tid >= 0:
                scene_tid_box_ct[(scn, tid, info['token'])].append(
                    inst['bbox_3d'][:3])
    dup = [k for k, v in scene_tid_box_ct.items() if len(v) > 1]
    print(f'[7] (scene,tid,frame) duplicates (track_id reused in same frame): '
          f'{len(dup)}')
    if dup[:3]:
        for k in dup[:3]:
            print(f'       {k}: {len(scene_tid_box_ct[k])} boxes')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', nargs='+', required=True)
    args = ap.parse_args()
    for p in args.pkl:
        audit(p)


if __name__ == '__main__':
    main()
