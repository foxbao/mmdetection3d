# -*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.

import os
import os.path as osp
from pathlib import Path
import json
import numpy as np
from decimal import getcontext
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pyquaternion import Quaternion
from numba import njit, prange
import mmengine
import cv2
import torch

from tools.utils.visualize_tools import validate_img_box
from mmdet3d.structures import LiDARInstance3DBoxes


getcontext().prec = 30


def load_json_if_exists(path: Path):
    if path is None or not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)

kl_categories = ( "Pedestrian", 
                 "Car", 
                 "IGV-Full", 
                 "Truck", 
                 "Trailer-Empty", 
                 "Trailer-Full", 
                 "IGV-Empty", 
                 "Crane", 
                 "OtherVehicle", 
                 "Cone", 
                 "ContainerForklift", 
                 "Forklift", 
                 "Lorry", 
                 "ConstructionVehicle", 
                 "WheelCrane" )

# ------------------- 点云读取 -------------------
def read_pcd_with_intensity(pcd_path):
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('DATA'):
                break
    fields, size, type_ = None, None, None
    for line in header:
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            size = list(map(int, line.split()[1:]))
        elif line.startswith('TYPE'):
            type_ = line.split()[1:]
    if fields is None or size is None or type_ is None:
        raise ValueError("Invalid PCD header: missing FIELDS/SIZE/TYPE")
    if not len(fields) == len(size) == len(type_):
        raise ValueError("FIELDS/SIZE/TYPE length mismatch")
    def get_numpy_dtype(t, s):
        if t == 'F': return np.float32 if s==4 else np.float64
        elif t == 'U': return {1:np.uint8,2:np.uint16,4:np.uint32}.get(s)
        elif t == 'I': return {1:np.int8,2:np.int16,4:np.int32}.get(s)
        raise ValueError(f"Unsupported TYPE/SIZE combination: TYPE={t}, SIZE={s}")
    # print(fields)
    dtype = np.dtype([(f, get_numpy_dtype(t,s)) for f,t,s in zip(fields,type_,size)])
    data_offset = len('\n'.join(header)) + 1
    # ... 前面读取 data 的代码不变 ...validate_img_box
    data = np.fromfile(pcd_path, dtype=dtype, offset=data_offset)

    # 定义我们想要的字段
    required_fields = ['x', 'y', 'z', 'intensity']
    arrs = [data[f].astype(np.float32) for f in required_fields]

    # 检查 timestamp_2us 是否存在
    if 'timestamp_2us' in data.dtype.names:
        arrs.append(data['timestamp_2us'].astype(np.float32))
    else:
        # 如果不存在，手动创建一个全 0 列，长度与 data 一致
        arrs.append(np.zeros(data.shape[0], dtype=np.float32))

    all_data = np.vstack(arrs).T
    valid_mask = ~np.isnan(all_data).any(axis=1)
    return all_data[valid_mask]

def read_pc(pc_file):
    pc_file = Path(pc_file)
    if not pc_file.exists():
        raise FileNotFoundError(f"{pc_file} not exist")
    if pc_file.suffix=='.bin':
        dtype = np.dtype([('x',np.float32),('y',np.float32),('z',np.float32),
                          ('intensity',np.float32),('ring',np.float32),('timestamp_2us',np.float32)])
        data = np.fromfile(pc_file,dtype=dtype)
        points = np.vstack([data['x'],data['y'],data['z'],data['intensity']]).T
    elif pc_file.suffix=='.pcd':
        points = read_pcd_with_intensity(pc_file)
    else:
        raise ValueError(f"Unsupported file format: {pc_file.suffix}")
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    points = points[np.max(np.abs(points[:,:3]),axis=1)<1e3]
    return points

# ------------------- 坐标变换 -------------------
def get_transform_matrix(quat_list):
    t = np.array(quat_list[:3],dtype=np.float32)
    q = Quaternion(w=quat_list[6], x=quat_list[3], y=quat_list[4], z=quat_list[5]).normalised
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = q.rotation_matrix.astype(np.float32)
    T[:3,3] = t
    return T

# Numba 加速点云矩阵变换
@njit(parallel=False)
def transform_points_numba(points, rotation, translation):
    N = points.shape[0]
    out = np.empty_like(points)
    for i in prange(N):
        out[i, :3] = rotation @ points[i, :3] + translation
        out[i, 3:] = points[i, 3:]
    return out

def find_nearest_ts_index(sorted_ts,target_ts):
    idx = np.searchsorted(sorted_ts,target_ts)
    if idx==0: return 0
    elif idx>=len(sorted_ts): return len(sorted_ts)-1
    else:
        prev_diff = abs(sorted_ts[idx-1]-target_ts)
        next_diff = abs(sorted_ts[idx]-target_ts)
        return idx-1 if prev_diff<=next_diff else idx

def generate_token():
    import uuid
    return str(uuid.uuid4())

def get_class_name_from_type(anno_data):
    names = []
    for obj in anno_data:
        if obj['label'] in kl_categories:
            names.append(obj['label'])
        elif obj['subtype'] in kl_categories:
            names.append(obj['subtype'])
        else:
            names.append('Unknown')
            print("[Warning] Unknown category:", obj['label'], obj['subtype'])
    return names

def get_undist_image_path(img_path: Path):
    """
    把 .../camera/xxx_image/xxx.jpg
    变成 .../camera_undist/xxx_image/xxx.jpg
    """
    parts = list(img_path.parts)
    try:
        cam_idx = parts.index('camera')
    except ValueError:
        raise RuntimeError(f"'camera' not found in path: {img_path}")

    parts[cam_idx] = 'camera_undist'
    return Path(*parts)

def merge_lidar_points(frame_info, frame_id, merged_file):
    """
    返回:
        True  -> 成功（或者文件已存在）
        False -> 失败（时间对不上 / 文件缺失）
    """
    if merged_file.exists():
        return True

    merged_points = []

    for lidar_name in frame_info['used_lidars']:
        ts_array = frame_info['lidar_sorted_ts'][lidar_name]
        nearest_idx = find_nearest_ts_index(ts_array, frame_id)
        nearest_ts = ts_array[nearest_idx]

        if abs(nearest_ts - frame_id) > frame_info['max_diff']:
            return False

        lidar_file = frame_info['lidar_file_index'][lidar_name][nearest_ts]
        if not lidar_file.exists():
            return False

        points = read_pc(lidar_file)
        T = get_transform_matrix(frame_info['extrinsics_dict'][lidar_name])
        points_trans = transform_points_numba(points, T[:3, :3], T[:3, 3])

        # 我们的坐标系本身是x朝前，y朝左，z朝上，nus坐标系是x朝右，y朝前，z朝上，所以沿着z轴旋转90度
        # 所以新的坐标点，x=-y，y=x，z=z
        if frame_info['coord_transform']:
            pts = np.empty_like(points_trans)
            pts[:, 0] = -points_trans[:, 1]
            pts[:, 1] = points_trans[:, 0]
            pts[:, 2] = points_trans[:, 2]
            pts[:, 3:] = points_trans[:, 3:]
            merged_points.append(pts)
        else:
            merged_points.append(points_trans)

    if not merged_points:
        return False

    merged_points = np.vstack(merged_points).astype(np.float32)
    merged_points = merged_points[~np.isnan(merged_points).any(axis=1)]
    merged_points.tofile(merged_file)

    return True

def make_yaw_rotation(yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [ 0,            0,           1]
    ])
    return R

def process_cameras(frame_info, frame_id, scale=1.0 / 3.0):
    CAM_NAME_MAP = {
        'front': 'CAM_FRONT',
        'left_front': 'CAM_FRONT_LEFT',
        'left_rear': 'CAM_BACK_LEFT',
        'rear': 'CAM_BACK',
        'right_front': 'CAM_FRONT_RIGHT',
        'right_rear': 'CAM_BACK_RIGHT',
    }

    cams = {}

    for cam_name in frame_info['used_cameras']:
        if cam_name not in CAM_NAME_MAP:
            continue

        ts_array = frame_info['camera_sorted_ts'].get(cam_name)
        if ts_array is None:
            continue

        nearest_idx = find_nearest_ts_index(ts_array, frame_id)
        nearest_ts = ts_array[nearest_idx]

        if abs(nearest_ts - frame_id) > frame_info['max_diff']:
            continue

        img_path = frame_info['camera_file_index'][cam_name][nearest_ts]
        intrin = frame_info['camera_intrinsics_dict'][cam_name]

        # ---------- 原始 K / D ----------
        K = np.array([
            [intrin['fx'], 0.0, intrin['cx']],
            [0.0, intrin['fy'], intrin['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        D = np.array([
            intrin.get('k1', 0.0),
            intrin.get('k2', 0.0),
            intrin.get('k3', 0.0),
            intrin.get('k4', 0.0),
        ], dtype=np.float64)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'[WARNING] cv2.imread returned None for: {img_path}')
            continue

        h, w = img.shape[:2]

        # 目标尺寸
        target_w = int(w * scale)
        target_h = int(h * scale)

        # ---------- 1️⃣ fisheye new_K ----------
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )

        # ---------- 2️⃣ undistort ----------
        undist_img_path = get_undist_image_path(img_path)

        if not undist_img_path.exists():
            undist_img_path.parent.mkdir(parents=True, exist_ok=True)

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )

            undist_img = cv2.remap(
                img, map1, map2,
                interpolation=cv2.INTER_LINEAR,
                # borderMode=cv2.BORDER_CONSTANT
            )

            # ---------- 3️⃣ resize ----------
            undist_img = cv2.resize(
                undist_img,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )

            cv2.imwrite(str(undist_img_path), undist_img)

        # ---------- 4️⃣ resize 后的 K ----------
        K_resized = new_K.copy()
        K_resized[0, 0] *= scale
        K_resized[1, 1] *= scale
        K_resized[0, 2] *= scale
        K_resized[1, 2] *= scale
        
        # ---------------------------------------------
        # 1. 原始外参说明
        # ---------------------------------------------
        # 原始 extrinsic 是 camera -> 原始 LiDAR 的变换
        # 即：
        #   p_lidar_orig = extrinsic_cam_to_lidar_orig @ p_cam
        # 其中 extrinsic_cam_to_lidar_orig 为 4x4 齐次变换矩阵
        extrinsic_cam_to_lidar_orig = get_transform_matrix(
            frame_info['camera_extrinsics_dict'][cam_name]
        )  # shape: (4, 4)
        
        # ---------------------------------------------
        # 2. 坐标系差异说明
        # ---------------------------------------------
        # 原 LiDAR 坐标系: x->前, y->左, z->上
        # nuScenes 坐标系: x->右, y->前, z->上
        # 因此需要绕 Z 轴旋转 90° 来对齐
        T_lidar_orig_to_nus = np.eye(4)
        T_lidar_orig_to_nus[:3, :3] = make_yaw_rotation(90)

        # ---------------------------------------------
        # 3. 数学关系推导（重点，保留原推导）
        # ---------------------------------------------
        # 点云在两个 LiDAR 坐标系之间的关系：
        #   p_lidar_nus = T_lidar_orig_to_nus * p_lidar_orig
        #   p_lidar_orig  = T_lidar_orig_to_nus⁻¹ * p_lidar_nus
        #
        # 原始相机到 LiDAR 的关系：
        #   p_lidar_orig = extrinsic_cam_to_lidar_orig * p_cam
        #
        # 代入得到：
        #   p_lidar_nus = (T_lidar_orig_to_nus @ extrinsic_cam_to_lidar_orig) @ p_cam
        #
        # 因此修正后的相机到 LiDAR（nuScenes）外参为：
        #   extrinsic_cam_to_lidar_nus = T_lidar_orig_to_nus @ extrinsic_cam_to_lidar_orig
        #
        # 这就是为什么要对 extrinsic 做“左乘”旋转修正

        # ---------------------------------------------
        # 4. 左乘修正外参（非常关键）
        # ---------------------------------------------
        extrinsic_cam_to_lidar_nus = T_lidar_orig_to_nus @ extrinsic_cam_to_lidar_orig
        # ---------- cam_info ----------
        cam_info = {
            'data_path': str(undist_img_path),
            'type': CAM_NAME_MAP[cam_name],
            'sample_data_token': generate_token(),
            'timestamp': float(nearest_ts),

            'sensor2ego_translation': np.zeros(3, dtype=np.float32),
            'sensor2ego_rotation': np.array([1, 0, 0, 0], dtype=np.float32),
            'ego2global_translation': np.zeros(3, dtype=np.float32),
            'ego2global_rotation': np.array([1, 0, 0, 0], dtype=np.float32),

            'sensor2lidar_rotation': extrinsic_cam_to_lidar_nus[:3, :3].astype(np.float32),
            'sensor2lidar_translation': extrinsic_cam_to_lidar_nus[:3, 3].astype(np.float32),

            'cam_intrinsic': K_resized.astype(np.float32),
            'image_shape': (target_h, target_w),
            'camera_model': 'pinhole',
        }

        cams[CAM_NAME_MAP[cam_name]] = cam_info

    return cams

def recompute_num_lidar_pts(
    gt_boxes,
    lidar_path,
    device='cpu',
    origin=(0.5, 0.5, 0.5)
):
    """
    计算每个 GT box 内的点云数量。

    根据 device 参数选择后端：
    - 'cuda' / 'cpu': 使用 torch + LiDARInstance3DBoxes（批量化，适合 GPU 加速）
    - 'numpy': 纯 NumPy 逐 box 计算（多进程安全，无 torch 依赖）

    Args:
        gt_boxes (np.ndarray): (M, 7) [x,y,z,dx,dy,dz,yaw]
        lidar_path (str or Path): merged lidar bin (5-dim float32)
        device (str): 'cuda', 'cpu', or 'numpy'
        origin (tuple): box origin for LiDARInstance3DBoxes (torch backend only)
    Returns:
        num_lidar_pts (np.ndarray): (M,) int32
    """
    M = len(gt_boxes)
    if M == 0:
        return np.zeros((0,), dtype=np.int32)

    # ---------- load points ----------
    # merged bin is 5-dim (x,y,z,intensity,timestamp_2us) from read_pcd_with_intensity()
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points_xyz = points[:, :3]

    if device == 'numpy':
        # ---------- NumPy backend (multi-process safe, no torch) ----------
        num_lidar_pts = np.zeros((M,), dtype=np.int32)
        for i in range(M):
            cx, cy, cz, dx, dy, dz, yaw = gt_boxes[i]
            local_pts = points_xyz - np.array([cx, cy, cz], dtype=np.float32)
            c = np.cos(-yaw)
            s = np.sin(-yaw)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            local_xy = local_pts[:, :2] @ rot.T
            local_z = local_pts[:, 2]
            mask = (
                (np.abs(local_xy[:, 0]) <= dx / 2) &
                (np.abs(local_xy[:, 1]) <= dy / 2) &
                (np.abs(local_z) <= dz / 2)
            )
            num_lidar_pts[i] = int(mask.sum())
        return num_lidar_pts

    # ---------- Torch backend ----------
    pts = torch.from_numpy(points_xyz).float()
    boxes = torch.from_numpy(gt_boxes[:, :7]).float()

    if device == 'cuda':
        pts = pts.cuda(non_blocking=True)
        boxes = boxes.cuda(non_blocking=True)

    boxes3d = LiDARInstance3DBoxes(
        boxes,
        box_dim=7,
        origin=origin
    )

    # (N,) int tensor, value in [-1, M-1]
    point_box_ids = boxes3d.points_in_boxes_part(pts)

    num_lidar_pts = np.zeros((M,), dtype=np.int32)
    ids = point_box_ids.cpu().numpy()

    valid = ids >= 0
    box_ids, counts = np.unique(ids[valid], return_counts=True)
    num_lidar_pts[box_ids] = counts

    return num_lidar_pts

def process_gt_annotations(frame_info, frame_id, lidar_path, device='cuda'):
    """
    处理 GT：
    - 时间戳对齐
    - 构建 gt_boxes
    - 坐标系转换
    - 重新计算 num_lidar_pts（基于点云）
    - annotation filter（min_points_by_class）
    """

    # ---------- 时间对齐 ----------
    nearest_idx = find_nearest_ts_index(
        frame_info['label_ts_list'], frame_id
    )
    nearest_ts = frame_info['label_ts_list'][nearest_idx]

    if abs(nearest_ts - frame_id) > frame_info['max_diff']:
        return None

    label_path = frame_info['label_file_list'][nearest_idx]
    if not label_path.exists():
        print(f'[WARNING] label file not found: {label_path}')
        return None

    with open(label_path, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)

    if not anno_data:
        return None

    # ---------- 构建 GT box ----------
    locs = np.array([s['xyz'] for s in anno_data], dtype=np.float32)
    dims = np.array([s['lwh'] for s in anno_data], dtype=np.float32)
    rots = np.array(
        [s['rotation']['z'] for s in anno_data],
        dtype=np.float32
    ).reshape(-1, 1)

    gt_boxes = np.concatenate([locs, dims, rots], axis=1)

    # ---------- 坐标系转换 ----------
    # x前 y左 z上  ->  nus: x右 y前 z上
    if frame_info.get('coord_transform', False):
        gt_boxes[:, 0] = -locs[:, 1]
        gt_boxes[:, 1] =  locs[:, 0]
        gt_boxes[:, 2] =  locs[:, 2]
        gt_boxes[:, 6] = rots[:, 0] + np.pi / 2
        gt_boxes[:, 6] = (gt_boxes[:, 6] + np.pi) % (2 * np.pi) - np.pi

    # ---------- 重新计算 num_lidar_pts（按类别选择性） ----------
    # WheelCrane 的标注 num_lidar_pts 值不准确（标注工具 bug），需要通过点云重算。
    # 其他类别的标注值是可信的，直接使用 label 中的值以节省计算。
    num_lidar_pts = np.zeros(len(anno_data), dtype=np.int32)

    wheelcrane_indices = []
    wheelcrane_boxes = []

    for i, ann in enumerate(anno_data):
        cls_name = ann.get('subtype') or ann.get('name')

        if cls_name == 'WheelCrane':
            wheelcrane_indices.append(i)
            wheelcrane_boxes.append(gt_boxes[i])
        else:
            num_lidar_pts[i] = ann.get('num_lidar_pts', 0)

    if wheelcrane_boxes:
        wheelcrane_boxes = np.asarray(wheelcrane_boxes, dtype=np.float32)

        wc_num_pts = recompute_num_lidar_pts(
            gt_boxes=wheelcrane_boxes,
            lidar_path=lidar_path,
            device=device
        )

        for idx, pts in zip(wheelcrane_indices, wc_num_pts):
            num_lidar_pts[idx] = pts
    # ---------- GT annotation filter ----------
    gt_filter_cfg = frame_info.get('gt_annotation_filter', None)
    if gt_filter_cfg and gt_filter_cfg.get('enable', False):
        min_pts_by_cls = gt_filter_cfg.get('min_points_by_class', {})

        keep_mask = []
        for i, ann in enumerate(anno_data):
            cls_name = ann.get('subtype') or ann.get('name')
            min_req = min_pts_by_cls.get(cls_name, 0)
            keep_mask.append(num_lidar_pts[i] >= min_req)

        keep_mask = np.array(keep_mask, dtype=bool)

        if not keep_mask.any():
            return None

        # 同步过滤
        gt_boxes = gt_boxes[keep_mask]
        num_lidar_pts = num_lidar_pts[keep_mask]
        anno_data = [a for k, a in zip(keep_mask, anno_data) if k]

    # ---------- 构建 gt_dict ----------
    gt_dict = {
        'gt_boxes': gt_boxes,
        'gt_names': np.array(get_class_name_from_type(anno_data)),
        'gt_velocity': np.zeros((len(anno_data), 2), dtype=np.float32),
        'num_lidar_pts': num_lidar_pts.tolist(),
        'num_radar_pts': [0] * len(anno_data),
        'valid_flag': (num_lidar_pts > 0).tolist()
    }

    return gt_dict

def process_frame(frame_info):
    frame_id = float(frame_info['frame_stem'])

    # =================== 基础 info ===================
    info = {
        'lidar_path': None,   # 先占位，后面补
        'num_features': 4,
        'token': generate_token(),
        'sweeps': [],
        'cams': {},
        'lidar2ego_translation': [0, 0, 0],
        'lidar2ego_rotation': [1, 0, 0, 0],
        'ego2global_translation': [0, 0, 0],
        'ego2global_rotation': [1, 0, 0, 0],
        'timestamp': frame_id,
        # 👇 保留 frame_info 里 GT 需要的信息
        'frame_info': frame_info,
    }

    # =================== LiDAR ===================
    # save the merged lidar file in 'samples' folder
    merged_file = frame_info['save_path'] / f"{frame_info['frame_stem']}.bin"
    ok = merge_lidar_points(frame_info, frame_id, merged_file)
    if not ok:
        return None
    info['lidar_path'] = str(merged_file)

    # =================== Camera ===================
    cams = process_cameras(frame_info, frame_id, scale=1.0 / 3.0)
    if len(cams) != len(frame_info['used_cameras']):
        return None

    info['cams'] = cams

    return info


def find_lidar_parent_dirs(current_path):
    current_path = Path(current_path)
    found_parents = []
    
    try:
        # 遍历当前层级下的所有文件和文件夹
        for item in current_path.iterdir():
            # 判断是否为目录（支持软链接追踪）
            if item.is_dir():
                # 如果这个目录的名字叫 lidar
                if item.name == "lidar":
                    # 将它的父目录（即当前的 current_path）加入列表
                    found_parents.append(item.parent)
                else:
                    # 如果不是 lidar，则进入该子目录继续递归寻找
                    found_parents.extend(find_lidar_parent_dirs(item))
    except PermissionError:
        pass 
        
    return found_parents


def process_gt_for_infos(infos, device='cuda'):
    final_infos = []

    for info in tqdm(infos, desc="Processing GT"):
        frame_info = info.pop('frame_info')
        frame_id = info['timestamp']
        lidar_path = info['lidar_path']

        gt_info = process_gt_annotations(
            frame_info,
            frame_id,
            lidar_path=lidar_path,
            device=device,   # 👈 只在这里用 CUDA
        )

        if gt_info is None:
            continue

        info.update(gt_info)
        final_infos.append(info)

    return final_infos

# ------------------- 主函数 -------------------
def generate_frame_bin_parallel(data_root, info_prefix, version, coord_transform = True,max_diff=0.05,cfg=None):
    # info_prefix = 'kl'
    data_root = Path(data_root)
    sample_path = data_root/version/'sample'
    out_sample_path = data_root/version/'samples'
    
    gt_annotation_filter = None
    if cfg is not None and 'gt_annotation_filter' in cfg:
        gt_annotation_filter = cfg.gt_annotation_filter

    frame_info_list = []
    # lidar_dirs = [p.parent for p in sample_path.rglob("lidar") if p.is_dir()]
    lidar_dirs = find_lidar_parent_dirs(sample_path)
    lidar_dirs = list(set(lidar_dirs))
    # for scene_path in lidar_dirs:
    total_label_count = 0
    skip_no_label = 0
    skip_no_extrinsics = 0
    for scene_path in tqdm(lidar_dirs, desc="Scanning scenes", unit="scene"):
        label_path_scene = Path(str(scene_path).replace('sample','label'))
        if not label_path_scene.exists():
            skip_no_label += 1
            print(f'[SKIP] no label dir: {label_path_scene}')
            continue
        # ---------- 标定文件路径 ----------
        extrinsics_path = scene_path / 'extrinsics.json'
        if not extrinsics_path.exists():
            # 统计被跳过的 label 数量
            skipped_labels = len([p for p in label_path_scene.iterdir() if p.is_file()])
            total_label_count += skipped_labels
            skip_no_extrinsics += 1
            print(f'[SKIP] no extrinsics: {scene_path.name}, has {skipped_labels} labels, total={total_label_count}')
            continue  # lidar 外参是必须的
        
        parent_path = scene_path.parent
        camera_extrinsics_path = parent_path / 'camera_extrinsics.json'
        intrinsics_path = parent_path / 'intrinsics.json'
        
        # ---------- 读取标定 ----------
        extrinsics = load_json_if_exists(extrinsics_path)
        camera_extrinsics = load_json_if_exists(camera_extrinsics_path)
        camera_intrinsics = load_json_if_exists(intrinsics_path)
        
        # ---------- lidar 外参 ----------
        used_lidars = []
        extrinsics_dict = {}

        lidar_prefix = 'Tx_baselink_lidar_'
        for sensor_name, quat in extrinsics.items():
            if lidar_prefix not in sensor_name:
                continue
            sensor_name_clean = sensor_name.split(lidar_prefix)[-1]
            used_lidars.append(sensor_name_clean)
            extrinsics_dict[sensor_name_clean] = quat
            
        # ---------- camera 外参 & 内参（可选） ----------
        used_cameras = []
        camera_extrinsics_dict = {}
        camera_intrinsics_dict = {}

        camera_prefix = 'Tx_baselink_camera_'

        for sensor_name, quat in camera_extrinsics.items():
            if camera_prefix not in sensor_name:
                continue
            cam_name = sensor_name.split(camera_prefix)[-1]
            used_cameras.append(cam_name)
            camera_extrinsics_dict[cam_name] = quat

        camera_intrinsic_prefix='camera_'
        for sensor_name, intr in camera_intrinsics.items():
            if camera_intrinsic_prefix not in sensor_name:
                continue
            cam_name = sensor_name.split(camera_intrinsic_prefix)[-1]
            camera_intrinsics_dict[cam_name] = intr
            if cam_name not in used_cameras:
                used_cameras.append(cam_name)


        # ==========================================================
        # ✅ camera_selection（作用在 cam_name 层）
        # ==========================================================
        camera_selection = None
        if cfg is not None and hasattr(cfg, 'camera_selection'):
            camera_selection = cfg.camera_selection

        if camera_selection and camera_selection.get('enable', False):
            selected = set(camera_selection.get('use_cameras', []))

            # 只保留 selection 中的 camera
            used_cameras = [c for c in used_cameras if c in selected]

            camera_extrinsics_dict = {
                c: v for c, v in camera_extrinsics_dict.items()
                if c in used_cameras
            }
            camera_intrinsics_dict = {
                c: v for c, v in camera_intrinsics_dict.items()
                if c in used_cameras
            }

            if len(used_cameras) == 0:
                continue
        # ==========================================================


        # ---------- lidar 文件索引 ----------
        lidar_file_index = {}
        lidar_sorted_ts = {}

        for lidar_name in used_lidars:
            lidar_path = scene_path / 'lidar' / lidar_name
            if not lidar_path.exists():
                print(f'[WARNING] lidar dir not found: {lidar_path}')
                continue

            files = list(lidar_path.glob('*.pcd'))
            if len(files) == 0:
                continue

            ts_list = np.array([float(f.stem) for f in files])
            idx_sort = np.argsort(ts_list)

            lidar_file_index[lidar_name] = dict(
                zip(ts_list[idx_sort], [files[i] for i in idx_sort])
            )
            lidar_sorted_ts[lidar_name] = ts_list[idx_sort]
            
        # ---------- camera 文件索引 ----------
        camera_file_index = {}
        camera_sorted_ts = {}

        camera_root = scene_path / 'camera'

        for cam_name in used_cameras:
            cam_dir = camera_root / f'{cam_name}_image'
            if not cam_dir.exists():
                continue

            img_files = list(cam_dir.glob('*.jpg')) + list(cam_dir.glob('*.png'))
            if len(img_files) == 0:
                continue

            ts_list = np.array([float(p.stem) for p in img_files])
            idx_sort = np.argsort(ts_list)

            camera_file_index[cam_name] = dict(
                zip(ts_list[idx_sort], [img_files[i] for i in idx_sort])
            )
            camera_sorted_ts[cam_name] = ts_list[idx_sort]

        # ---------- label 文件 ----------
        label_files = sorted(
            [p for p in label_path_scene.iterdir() if p.is_file()],
            key=lambda p: float(p.stem)
        )

        total_label_count += len(label_files)
        

        label_ts_list = np.array([float(p.stem) for p in label_files])

        save_path = out_sample_path
        save_path.mkdir(parents=True, exist_ok=True)

        # ---------- frame info ----------
        for label_file in label_files:
            frame_info_list.append({
                'frame_stem': label_file.stem,
                'used_lidars': used_lidars,
                'lidar_file_index': lidar_file_index,
                'lidar_sorted_ts': lidar_sorted_ts,
                'extrinsics_dict': extrinsics_dict,
                'used_cameras': used_cameras,
                'camera_file_index': camera_file_index,
                'camera_sorted_ts': camera_sorted_ts,
                'camera_extrinsics_dict': camera_extrinsics_dict,
                'camera_intrinsics_dict': camera_intrinsics_dict,
                'save_path': save_path,
                'label_file_list': label_files,
                'label_ts_list': label_ts_list,
                'max_diff': max_diff,
                'coord_transform': coord_transform,
                'gt_annotation_filter': gt_annotation_filter
            })


    
    print(f'\n[SUMMARY] total_label_count={total_label_count}, '
          f'lidar_dirs={len(lidar_dirs)}, '
          f'skip_no_label={skip_no_label}, skip_no_extrinsics={skip_no_extrinsics}')

    num_workers = min(32, os.cpu_count())
    # all_infos = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # chunksize 设置为 10-20 比较合适
        results = list(tqdm(executor.map(process_frame, frame_info_list, chunksize=20), 
                            total=len(frame_info_list), 
                            desc="Processing frames"))
    
    base_infos = [res for res in results if res]
    print(f"[Stage 1] cam+lidar process, valid frames: {len(base_infos)}")
    
    # =================== GT ===================
    all_infos = process_gt_for_infos(base_infos, device='cuda')
    print(f"[Stage 2] gt process, final frames: {len(all_infos)}")
    # gt_info = process_gt_annotations(frame_info, frame_id,lidar_path=info['lidar_path'])
    # if gt_info is None:
    #     return None

    # info.update(gt_info)
    # # 过滤掉空的结果
    # all_infos = [res for res in results if res]

    n = len(all_infos)
    
    # all_infos = []

    # for frame_info in tqdm(frame_info_list, desc="Processing frames"):
    #     res = process_frame(frame_info)
    #     if res:
    #         all_infos.append(res)

    # 查看label往图片上投影，验证内外参
    # for info in all_infos:
    #     validate_img_box(info)
    
    rng = np.random.default_rng()
    perm = rng.permutation(n)
    split_idx = int(n*0.9)
    train_results = [all_infos[i] for i in perm[:split_idx]]
    val_results = [all_infos[i] for i in perm[split_idx:]]
    print(f"Total frames: {n}, Train: {len(train_results)}, Val: {len(val_results)}")
    metadata = dict(version=version)
    data = dict(infos=train_results, metadata=metadata)
    info_path = osp.join(data_root,
                            '{}_infos_train.pkl'.format(info_prefix))
    mmengine.dump(data, info_path)
    data['infos'] = val_results
    info_val_path = osp.join(data_root,
                                '{}_infos_val.pkl'.format(info_prefix))
    mmengine.dump(data, info_val_path)
    return train_results, val_results


def create_kl_infos(data_root, info_prefix, version='v1.0-trainval', cfg=None):
    generate_frame_bin_parallel(data_root, info_prefix, version, cfg=cfg)


# ------------------- 脚本入口 -------------------
if __name__=="__main__":
    data_root = '/media/cx/bak/data/kl'
    version = 'v1.0-trainval'
    generate_frame_bin_parallel(data_root, 'kl', version)
