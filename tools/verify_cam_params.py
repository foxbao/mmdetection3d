"""
Camera parameter verification for KL dataset.

Checks:
1. Image paths exist and images load correctly
2. Camera intrinsics sanity (focal length, principal point)
3. LiDAR -> image projection alignment (LiDAR points should land on visible surfaces)
4. GT 3D box -> image projection (corners should surround the object)

Usage:
    python tools/verify_cam_params.py --data-root data/kl_8 --num-samples 5
"""
import argparse
import os
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['data_list']


def load_image(data_root, data_prefix_img, img_path):
    full = os.path.join(data_root, data_prefix_img, img_path)
    img = cv2.imread(full)
    if img is None:
        print(f"  [ERROR] Cannot load image: {full}")
    return img, full


def load_lidar(data_root, lidar_path):
    full = os.path.join(data_root, 'v1.0-trainval/samples', lidar_path)
    if not os.path.exists(full):
        print(f"  [ERROR] LiDAR file not found: {full}")
        return None
    pts = np.fromfile(full, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]  # xyz only


def get_box_corners_3d(box):
    """Get 8 corners of a 3D LiDAR box [x,y,z,dx,dy,dz,yaw]."""
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners_local @ R.T + np.array([cx, cy, cz])
    return corners  # (8, 3)


def project_points_to_image(pts_lidar, lidar2cam, cam2img, img_shape):
    """Project 3D LiDAR points onto image. Returns u, v, depth arrays."""
    N = len(pts_lidar)
    pts_h = np.hstack([pts_lidar, np.ones((N, 1))])
    pts_cam = (lidar2cam @ pts_h.T).T

    mask = pts_cam[:, 2] > 0.1
    if mask.sum() == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    pts_cam = pts_cam[mask]
    K = np.array(cam2img)[:3, :3]
    pts_img = (K @ pts_cam[:, :3].T).T
    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    d = pts_cam[:, 2]

    h, w = img_shape[:2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[in_img], v[in_img], d[in_img]


def draw_projected_lidar(img, u, v, d, max_depth=50.0):
    """Draw LiDAR points colored by depth: red=near, blue=far."""
    out = img.copy()
    if len(u) == 0:
        return out
    d_norm = np.clip(d / max_depth, 0, 1)
    for i in range(len(u)):
        r = int((1 - d_norm[i]) * 255)
        b = int(d_norm[i] * 255)
        cv2.circle(out, (int(u[i]), int(v[i])), 2, (b, 100, r), -1)
    return out


def draw_box_2d(img, corners_3d, lidar2cam, cam2img, color=(0, 255, 0)):
    """Project 8 3D box corners to 2D and draw edges."""
    out = img.copy()
    h, w = img.shape[:2]
    N = 8
    pts_h = np.hstack([corners_3d, np.ones((N, 1))])
    pts_cam = (lidar2cam @ pts_h.T).T

    # skip if all corners behind camera
    if np.all(pts_cam[:, 2] < 0.1):
        return out

    K = np.array(cam2img)[:3, :3]
    pts_img = (K @ pts_cam[:, :3].T).T
    us = pts_img[:, 0] / pts_img[:, 2]
    vs = pts_img[:, 1] / pts_img[:, 2]
    pts2d = [(int(np.clip(us[i], -1e4, 1e4)), int(np.clip(vs[i], -1e4, 1e4)))
             for i in range(8)]

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        pa, pb = pts2d[a], pts2d[b]
        # only draw if at least one endpoint is in image
        if ((-100 < pa[0] < w+100 and -100 < pa[1] < h+100) or
            (-100 < pb[0] < w+100 and -100 < pb[1] < h+100)):
            cv2.line(out, pa, pb, color, 2)
    return out


def check_intrinsics(cam_name, cam2img, img_shape):
    K = np.array(cam2img)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = img_shape[:2]

    issues = []
    if not (100 < fx < 2000):
        issues.append(f"fx={fx:.1f} out of range")
    if not (100 < fy < 2000):
        issues.append(f"fy={fy:.1f} out of range")
    if not (w * 0.1 < cx < w * 0.9):
        issues.append(f"cx={cx:.1f} not near center (w={w})")
    if not (h * 0.1 < cy < h * 0.9):
        issues.append(f"cy={cy:.1f} not near center (h={h})")

    status = "OK" if not issues else "WARN: " + "; ".join(issues)
    print(f"      intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
          f"[{w}x{h}] -> {status}")
    return len(issues) == 0


def verify_sample(info, data_root, out_dir, sample_idx):
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}  token={info.get('token','?')[:20]}...")

    lidar_path = info['lidar_points']['lidar_path']
    pts_lidar = load_lidar(data_root, lidar_path)
    if pts_lidar is None:
        print("  Skipping: LiDAR file missing")
        return

    # subsample for speed
    if len(pts_lidar) > 50000:
        idx = np.random.choice(len(pts_lidar), 50000, replace=False)
        pts_lidar_vis = pts_lidar[idx]
    else:
        pts_lidar_vis = pts_lidar
    print(f"  LiDAR: {len(pts_lidar)} pts  file={lidar_path}")

    # GT boxes
    instances = info.get('instances', [])
    valid_boxes = [(inst['bbox_3d'], inst['bbox_label_3d'])
                   for inst in instances
                   if inst.get('bbox_3d_isvalid', True) and
                      inst.get('num_lidar_pts', 1) > 0]
    print(f"  GT: {len(instances)} instances, {len(valid_boxes)} valid boxes")

    os.makedirs(out_dir, exist_ok=True)

    box_palette = [
        (0,255,0),(0,200,255),(255,100,0),(255,255,0),(0,0,255),
        (255,0,200),(100,255,0),(0,100,255),(200,200,0),(0,200,200)
    ]

    any_issue = False
    for cam_name, cam_info in info['images'].items():
        img_path = cam_info['img_path']
        img, full_path = load_image(data_root, 'v1.0-trainval/sample', img_path)
        if img is None:
            any_issue = True
            continue

        h, w = img.shape[:2]
        print(f"\n  [{cam_name}]  size={w}x{h}  .../{'/'.join(Path(img_path).parts[-3:])}")

        cam2img  = cam_info['cam2img']
        lidar2cam = np.array(cam_info['lidar2cam'], dtype=np.float64)

        # --- intrinsics check ---
        ok = check_intrinsics(cam_name, cam2img, img.shape)
        if not ok:
            any_issue = True

        # --- lidar2cam sanity ---
        R = lidar2cam[:3, :3]
        det = np.linalg.det(R)
        t = lidar2cam[:3, 3]
        det_ok = abs(det - 1.0) < 0.01
        print(f"      lidar2cam: det(R)={det:.5f}{'  OK' if det_ok else '  WARN: not ~1!'} "
              f"  t=[{t[0]:.2f} {t[1]:.2f} {t[2]:.2f}]m")
        if not det_ok:
            any_issue = True

        # --- project LiDAR ---
        u, v, d = project_points_to_image(pts_lidar_vis, lidar2cam, cam2img, img.shape)
        coverage = len(u) / max(len(pts_lidar_vis), 1) * 100
        median_d = float(np.median(d)) if len(d) > 0 else -1
        print(f"      LiDAR proj: {len(u)}/{len(pts_lidar_vis)} pts in view "
              f"({coverage:.1f}%)  median_depth={median_d:.1f}m")
        if coverage < 0.5:
            print(f"      [WARN] Suspiciously few LiDAR points projected! "
                  f"Possible extrinsic error.")
            any_issue = True
        if median_d < 0 or median_d > 200:
            print(f"      [WARN] Median depth {median_d:.1f}m is unreasonable!")
            any_issue = True

        # draw
        vis = draw_projected_lidar(img, u, v, d)

        # --- project GT boxes ---
        boxes_visible = 0
        for bi, (box, label) in enumerate(valid_boxes):
            corners = get_box_corners_3d(box)
            # rough check: box center in front of camera
            center_h = np.array([box[0], box[1], box[2], 1.0])
            center_cam = lidar2cam @ center_h
            if center_cam[2] > 0.5:  # in front
                color = box_palette[bi % len(box_palette)]
                vis = draw_box_2d(vis, corners, lidar2cam, cam2img, color)
                boxes_visible += 1

        print(f"      GT boxes visible: {boxes_visible}/{len(valid_boxes)}")

        # label
        cv2.putText(vis, cam_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"LiDAR pts: {len(u)}  boxes: {boxes_visible}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out_name = f"s{sample_idx:03d}_{cam_name}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis)

    if any_issue:
        print(f"\n  *** Issues found in sample {sample_idx} — check images! ***")
    else:
        print(f"\n  Sample {sample_idx}: all checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description='Verify KL camera params by projecting LiDAR+GT onto images')
    parser.add_argument('--data-root', default='data/kl_8')
    parser.add_argument('--pkl',       default='data/kl_8/kl_infos_val.pkl')
    parser.add_argument('--out-dir',   default='work_dirs/cam_verify')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--start-idx',   type=int, default=0)
    args = parser.parse_args()

    np.random.seed(42)

    print(f"Loading: {args.pkl}")
    data_list = load_pkl(args.pkl)
    print(f"Total samples: {len(data_list)}")

    indices = list(range(args.start_idx,
                         min(args.start_idx + args.num_samples, len(data_list))))

    for i in indices:
        verify_sample(data_list[i], args.data_root, args.out_dir, i)

    print(f"\n{'='*70}")
    print(f"Output images: {args.out_dir}/")
    print("Visual check guide:")
    print("  GOOD: Colored dots (LiDAR) land on correct surfaces in image")
    print("  GOOD: Box outlines surround actual objects")
    print("  BAD:  LiDAR dots floating in the air / wrong position")
    print("  BAD:  Boxes around wrong objects or empty space")


if __name__ == '__main__':
    main()
