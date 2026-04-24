import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import pyvista as pv
import math
import os

class_names = {
    0:"Pedestrian",
    1:"Car",
    2:"IGV-Full",
    3:"Truck",
    4:"Trailer-Empty",
    5:"Trailer-Full",
    6:"IGV-Empty",
    7:"Crane",
    8:"OtherVehicle",
    9:"Cone",
    10:"ContainerForklift",
    11:"Forklift",
    12:"Lorry",
    13:"ConstructionVehicle",
    14:"WheelCrane"
}

def box_to_corners_3d(box, z_bottom=False):
    """
    Args:
        box: [x, y, z, l, w, h, yaw] or longer
        z_bottom: if True, z means bottom center

    Returns:
        corners: (8, 3) ndarray in world frame
    """
    box = np.asarray(box).astype(float)
    x, y, z, l, w, h, yaw = box[:7]

    z_center = z + h / 2 if z_bottom else z

    corners_local = np.array([
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])

    corners_world = corners_local @ R.T
    corners_world += np.array([x, y, z_center])

    return corners_world

BOX_EDGES = np.array([
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
], dtype=int)


def create_pyvista_box_lines_from_corners(corners):
    lines = np.hstack([[2, i, j] for i, j in BOX_EDGES])
    poly = pv.PolyData(corners)
    poly.lines = lines
    return poly

def create_pyvista_box_lines(box, z_bottom=False):
    corners = box_to_corners_3d(box, z_bottom)
    return create_pyvista_box_lines_from_corners(corners)


# =====================================================
# Grid stitching helpers
# =====================================================

def stitch_images_grid(imgs, num_cols=3):
    """Resize images to min-height and stitch into a grid.

    Pads incomplete rows with black images to keep alignment.
    Returns None if imgs is empty.
    """
    if len(imgs) == 0:
        return None
    h = min(im.shape[0] for im in imgs)
    imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
    rows = []
    for i in range(0, len(imgs), num_cols):
        row_imgs = imgs[i:i + num_cols]
        if len(row_imgs) < num_cols:
            pad = np.zeros_like(row_imgs[0])
            row_imgs += [pad] * (num_cols - len(row_imgs))
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def _stitch_max_height_grid(imgs, num_cols=3):
    """Resize images to max-height per row and stitch into a grid.

    Used by functions that draw boxes on individual camera images
    (images may have different sizes across cameras).
    Returns None if imgs is empty.
    """
    if len(imgs) == 0:
        return None
    rows = []
    for i in range(0, len(imgs), num_cols):
        row_imgs = imgs[i:i + num_cols]
        h_max = max(im.shape[0] for im in row_imgs)
        resized = []
        for im in row_imgs:
            if im.shape[0] != h_max:
                scale = h_max / im.shape[0]
                im = cv2.resize(im, (int(im.shape[1] * scale), h_max))
            resized.append(im)
        rows.append(cv2.hconcat(resized))
    return cv2.vconcat(rows)


def _read_and_label_images(img_paths):
    """Read images from paths and overlay camera name on each."""
    imgs = []
    for p in img_paths:
        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            print(f'[WARN] Failed to read image: {p}')
            continue
        cam_name = p.parent.name
        cv2.putText(img, cam_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        imgs.append(img)
    return imgs


# =====================================================
# 3D visualization (PyVista)
# =====================================================

def _add_axes(plotter, axis_len=5.0):
    """Add RGB axis lines at origin."""
    plotter.add_mesh(pv.Line((0, 0, 0), (axis_len, 0, 0)), color='red',   line_width=4)
    plotter.add_mesh(pv.Line((0, 0, 0), (0, axis_len, 0)), color='green', line_width=4)
    plotter.add_mesh(pv.Line((0, 0, 0), (0, 0, axis_len)), color='blue',  line_width=4)


def visualize_black_bg_vista(
    points,
    boxes=None,
    labels=None,
    scores=None,
    score_thresh=0.0,
    gt_boxes=None,
    gt_labels=None,
    focus_class_ids=None,
    vectors=None,
    vector_scale=1.0,
    vector_color='yellow',
):
    plotter = pv.Plotter()
    plotter.set_background('black')

    _add_axes(plotter)

    # Point Cloud
    pts = points[:, :3]
    pc = pv.PolyData(pts)
    plotter.add_points(pc, color='white', point_size=1, render_points_as_spheres=True)

    # Direction Vectors
    if vectors is not None:
        assert vectors.shape[0] == pts.shape[0], \
            "vectors 数量必须和 points 一致"
        for p, v in zip(pts, vectors):
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                continue
            v = v / v_norm * vector_scale
            line = pv.Line(p, p + v)
            plotter.add_mesh(line, color=vector_color, line_width=3)

    # GT boxes
    gt_label_pos = []
    gt_label_text = []

    if gt_boxes is not None:
        for i, box in enumerate(gt_boxes):
            gt_color = 'green'
            label_id = int(gt_labels[i]) if gt_labels is not None else None

            if label_id is not None and focus_class_ids and label_id in focus_class_ids:
                gt_color = 'yellow'

            box_lines = create_pyvista_box_lines(box)
            plotter.add_mesh(box_lines, color=gt_color, line_width=2)

            x, y, z, l, w, h, yaw = box[:7]
            gt_label_pos.append([x, y, z + h / 2 + 0.2])

            if label_id is not None:
                class_name = class_names.get(label_id, f'cls_{label_id}')
                gt_label_text.append(class_name)
            else:
                gt_label_text.append('GT')

    if len(gt_label_pos) > 0:
        plotter.add_point_labels(
            np.array(gt_label_pos),
            gt_label_text,
            text_color='green',
            font_size=14,
            point_size=0
        )

    # Pred boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            if scores is not None and scores[i] < score_thresh:
                continue
            box_lines = create_pyvista_box_lines(box)
            plotter.add_mesh(box_lines, color='red', line_width=2)

    plotter.show()


# =====================================================
# Multi-cam image display / save
# =====================================================

def show_multi_cam_images_from_path(img_paths, win_name='GT Cameras', num_cols=3):
    imgs = _read_and_label_images(img_paths)
    grid = stitch_images_grid(imgs, num_cols)
    if grid is not None:
        cv2.imshow(win_name, grid)
        cv2.waitKey(1)


def save_multi_cam_images_from_path(
    img_paths,
    save_dir='result_img',
    save_name=None,
    num_cols=3
):
    if len(img_paths) == 0:
        print('[WARN] img_paths is empty.')
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_name is None:
        first_path = Path(img_paths[0])
        save_name = first_path.stem + '.jpg'

    imgs = _read_and_label_images(img_paths)
    grid = stitch_images_grid(imgs, num_cols)
    if grid is None:
        print('[WARN] No valid images, skip saving.')
        return

    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), grid)
    print(f'[INFO] Saved multi-cam image to {save_path}')


# =====================================================
# Box projection onto images
# =====================================================

def _project_boxes_to_corners_cam(gt_boxes, R, t, z_bottom=False):
    """Project 3D boxes (lidar frame) to camera-frame corners.

    Args:
        gt_boxes: (N, >=7) boxes in lidar frame.
        R: (3, 3) rotation part of lidar2cam.
        t: (3,)   translation part of lidar2cam.
        z_bottom: if True, treat box[2] as bottom-center z (mmdet3d
            LiDARInstance3DBoxes convention). If False, treat it as
            geometric center (legacy KL info-file convention).

    Returns:
        list of (8, 3) arrays (only boxes in front of camera).
    """
    corners_list = []
    for box in gt_boxes:
        corners_lidar = box_to_corners_3d(box, z_bottom=z_bottom)
        corners_cam = corners_lidar @ R.T + t
        if not np.all(corners_cam[:, 2] <= 0):
            corners_list.append(corners_cam)
    return corners_list


def draw_boxes_on_image(img, corners_cam_list, K, color=(0, 255, 0), thickness=2):
    """Draw projected 3D box edges on an image.

    Args:
        img: BGR image (a copy is returned).
        corners_cam_list: list of (8, 3) corner arrays in camera frame.
        K: (3, 3) camera intrinsic.
        color: BGR color tuple.
        thickness: line thickness.
    """
    img_draw = img.copy()

    for corners_cam in corners_cam_list:
        if np.all(corners_cam[:, 2] <= 0):
            continue

        corners_clip = corners_cam.copy()
        corners_clip[corners_clip[:, 2] <= 0, 2] = 1e-6

        proj = K @ corners_clip.T           # (3, 8)
        proj = (proj[:2] / proj[2:3]).T.astype(int)  # (8, 2)

        for i, j in BOX_EDGES:
            if corners_cam[i, 2] <= 0 or corners_cam[j, 2] <= 0:
                continue
            cv2.line(img_draw, tuple(proj[i]), tuple(proj[j]), color, thickness)

    return img_draw


def save_multi_cam_images_with_boxes(
    img_paths,
    gt_boxes,
    data_sample,
    save_dir='result_img',
    save_name=None,
    num_cols=3,
):
    os.makedirs(save_dir, exist_ok=True)

    lidar2cam = data_sample.lidar2cam   # (N_cam, 4, 4)
    cam2img = data_sample.cam2img       # (N_cam, 4, 4)

    images_with_boxes = []

    for cam_id, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] Failed to read image: {img_path}')
            continue

        T_lidar2cam = lidar2cam[cam_id]
        K = cam2img[cam_id][:3, :3]
        R = T_lidar2cam[:3, :3]
        t = T_lidar2cam[:3, 3]

        corners_list = _project_boxes_to_corners_cam(gt_boxes, R, t)
        img_with_boxes = draw_boxes_on_image(img, corners_list, K)

        cam_name = f'CAM_{cam_id}'
        cv2.putText(
            img_with_boxes, cam_name, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (0, 255, 255), 2, cv2.LINE_AA
        )

        images_with_boxes.append(img_with_boxes)

    if len(images_with_boxes) == 0:
        print('[WARN] No valid images.')
        return

    final_img = _stitch_max_height_grid(images_with_boxes, num_cols)

    if save_name is None:
        save_name = 'multi_cam_gt.jpg'

    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, final_img)
    print(f'[INFO] Saved multi-cam image to {save_path}')


# =====================================================
# Coordinate transforms
# =====================================================

def lidar_to_camera(points_lidar, sensor2lidar_R, sensor2lidar_T):
    """Transform points from lidar frame to camera frame.

    Args:
        points_lidar: (N, >=3)
        sensor2lidar_R: (3, 3) cam -> lidar rotation.
        sensor2lidar_T: (3,)   cam -> lidar translation.
    """
    R = np.asarray(sensor2lidar_R).reshape(3, 3)
    T = np.asarray(sensor2lidar_T).reshape(3)

    R_lidar2cam = R.T
    t_lidar2cam = -R_lidar2cam @ T

    pts = points_lidar[:, :3]
    return pts @ R_lidar2cam.T + t_lidar2cam


def project_lidar_to_img(points_lidar, sensor2lidar_R, sensor2lidar_T, K):
    """Project lidar points to image pixel coordinates.

    Args:
        points_lidar: (N, 3)
        sensor2lidar_R: (3, 3) cam -> lidar rotation.
        sensor2lidar_T: (3,)   cam -> lidar translation.
        K: (3, 3) camera intrinsic.

    Returns:
        points_img: (M, 2) pixel coords of visible points.
        mask: (N,) bool mask of points in front of camera.
    """
    points_cam = lidar_to_camera(points_lidar, sensor2lidar_R, sensor2lidar_T)

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    points_img = points_cam @ np.asarray(K).T
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]

    return points_img[:, :2], mask


# =====================================================
# File I/O
# =====================================================

def load_lidar_points(lidar_path, num_features=5):
    """Load lidar point cloud from .bin or .pcd file.

    Args:
        lidar_path: path to point cloud file.
        num_features: number of features per point for .bin files (default 5).

    Returns:
        (N, 3) xyz coordinates.
    """
    if lidar_path.endswith('.bin'):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_features)
        return points[:, :3]
    elif lidar_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(lidar_path)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f'Unsupported lidar format: {lidar_path}')


# =====================================================
# Legacy / debug helpers
# =====================================================

def visualize_cam_points_and_boxes(
    points_cam,
    gt_boxes,
    sensor2lidar_R,
    sensor2lidar_T,
    axis_len=5.0,
    box_color='green'
):
    R = np.asarray(sensor2lidar_R)
    T = np.asarray(sensor2lidar_T)

    gt_corners_cam_list = []
    for box in gt_boxes:
        corners_lidar = box_to_corners_3d(box, z_bottom=False)
        corners_cam = lidar_to_camera(corners_lidar, R, T)
        gt_corners_cam_list.append(corners_cam)

    plotter = pv.Plotter()
    plotter.set_background('black')

    _add_axes(plotter, axis_len)

    plotter.add_points(
        points_cam,
        color='white',
        point_size=1,
        render_points_as_spheres=True
    )

    for corners_cam in gt_corners_cam_list:
        box_lines = create_pyvista_box_lines_from_corners(corners_cam)
        plotter.add_mesh(box_lines, color=box_color, line_width=2)

    plotter.show()


def draw_boxes_on_all_images(
    info,
    cam_names=None,
    mode='show',
    save_dir='gt_img',
    save_name=None
):
    if cam_names is None:
        cam_names = list(info['cams'].keys())

    lidar_path = info['lidar_path']
    gt_boxes = np.asarray(info['gt_boxes'])[:, :7]

    images_with_boxes = []

    for cam_name in cam_names:
        cam = info['cams'][cam_name]
        img_path = cam['data_path']
        img = cv2.imread(img_path)

        if img is None:
            print(f'[WARN] Failed to read image: {img_path}')
            continue

        R = np.array(cam['sensor2lidar_rotation'])
        T = np.array(cam['sensor2lidar_translation'])
        K = np.array(cam['cam_intrinsic'])

        corners_list = []
        for box in gt_boxes:
            corners_lidar = box_to_corners_3d(box, z_bottom=False)
            corners_cam = lidar_to_camera(corners_lidar, R, T)
            corners_list.append(corners_cam)

        img_with_boxes = draw_boxes_on_image(img, corners_list, K)

        cv2.putText(
            img_with_boxes, cam_name, org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
            color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )

        timestamp_str = str(cam.get('timestamp', 'N/A'))
        cv2.putText(
            img_with_boxes, timestamp_str, org=(20, 70),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA
        )

        images_with_boxes.append(img_with_boxes)

    if len(images_with_boxes) == 0:
        print('[ERROR] No valid images.')
        return

    final_img = _stitch_max_height_grid(images_with_boxes)

    if mode == 'save':
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            base = os.path.splitext(os.path.basename(lidar_path))[0]
            save_name = f'{base}_gt.jpg'

        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, final_img)
        print(f'[INFO] Saved GT image to: {save_path}')
    else:
        cv2.imshow('All GT boxes', final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def validate_img_box(info):
    print(info['lidar_path'])
    print(info['cams']['CAM_FRONT']['data_path'])
    draw_boxes_on_all_images(info, mode='save')
