import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import pyvista as pv
import math
import os


def box_to_corners_3d(box, z_bottom=False):
    """
    Args:
        box: [x, y, z, l, w, h, yaw] or longer
        z_bottom: if True, z means bottom center

    Returns:
        corners: (8, 3) ndarray in world frame
                 order is fixed and consistent
    """
    box = np.asarray(box).astype(float)
    x, y, z, l, w, h, yaw = box[:7]

    # z definition
    z_center = z + h / 2 if z_bottom else z

    # local corners (centered at origin)
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

    # rotation around z
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])

    # world transform
    corners_world = corners_local @ R.T
    corners_world += np.array([x, y, z_center])

    return corners_world

BOX_EDGES = np.array([
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
], dtype=int)


# def corners_to_lines(corners):
#     edges = [
#         (0,1),(1,2),(2,3),(3,0),
#         (4,5),(5,6),(6,7),(7,4),
#         (0,4),(1,5),(2,6),(3,7)
#     ]
#     lines = []
#     for i, j in edges:
#         lines.append(pv.Line(corners[i], corners[j]))
#     return lines
# def create_pyvista_box_lines(box, z_bottom=False):
#     corners = box_to_corners_3d(box, z_bottom)

#     edges = np.array([
#         [0,1],[1,2],[2,3],[3,0],
#         [4,5],[5,6],[6,7],[7,4],
#         [0,4],[1,5],[2,6],[3,7]
#     ])

#     lines = np.hstack([[2, e[0], e[1]] for e in edges])
    

#     poly = pv.PolyData(corners)
#     poly.lines = lines
#     return poly

def create_pyvista_box_lines_from_corners(corners):
    lines = np.hstack([[2, i, j] for i, j in BOX_EDGES])
    poly = pv.PolyData(corners)
    poly.lines = lines
    return poly
def create_pyvista_box_lines(box, z_bottom=False):
    corners = box_to_corners_3d(box, z_bottom)
    return create_pyvista_box_lines_from_corners(corners)


def visualize_black_bg_vista(
    points,
    boxes=None,
    labels=None,
    scores=None,
    score_thresh=0.0,
    gt_boxes=None,
    gt_labels=None,
    focus_class_ids=None,
    vectors=None,              # ✅ 新增
    vector_scale=1.0,          # ✅ 新增
    vector_color='yellow',     # ✅ 新增
):
    plotter = pv.Plotter()
    plotter.set_background('black')

    # ==========================
    # ⭐ 0️⃣ 原点坐标轴（世界坐标）
    # ==========================
    axis_len = 5.0  # 坐标轴长度（按场景可调）

    # X axis (red)
    x_axis = pv.Line((0, 0, 0), (axis_len, 0, 0))
    plotter.add_mesh(x_axis, color='red', line_width=4)

    # Y axis (green)
    y_axis = pv.Line((0, 0, 0), (0, axis_len, 0))
    plotter.add_mesh(y_axis, color='green', line_width=4)

    # Z axis (blue)
    z_axis = pv.Line((0, 0, 0), (0, 0, axis_len))
    plotter.add_mesh(z_axis, color='blue', line_width=4)

    # （可选）轴端文字
    # plotter.add_point_labels(
    #     np.array([
    #         [axis_len, 0, 0],
    #         [0, axis_len, 0],
    #         [0, 0, axis_len]
    #     ]),
    #     ['X', 'Y', 'Z'],
    #     text_color='white',
    #     font_size=14,
    #     point_size=0
    # )

    # ==========================
    # 1️⃣ Point Cloud
    # ==========================
    pts = points[:, :3]
    pc = pv.PolyData(pts)

    plotter.add_points(
        pc,
        color='white',
        point_size=1,
        render_points_as_spheres=True
    )

    # ==========================
    # 1️⃣➕ Direction Vectors (yaw debug)
    # ==========================
    if vectors is not None:
        assert vectors.shape[0] == pts.shape[0], \
            "vectors 数量必须和 points 一致"

        for p, v in zip(pts, vectors):
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                continue

            v = v / v_norm * vector_scale
            line = pv.Line(p, p + v)
            plotter.add_mesh(
                line,
                color=vector_color,
                line_width=3
            )
    # ==========================
    # 2️⃣ GT boxes
    # ==========================
    gt_label_pos = []
    gt_label_text = []

    if gt_boxes is not None:
        for i, box in enumerate(gt_boxes):
            gt_color = 'green'

            if gt_labels is not None and focus_class_ids:
                label_id = int(gt_labels[i])
                if label_id in focus_class_ids:
                    gt_color = 'yellow'

            box_lines = create_pyvista_box_lines(box)
            plotter.add_mesh(
                box_lines,
                color=gt_color,
                line_width=2
            )

            x, y, z, l, w, h, yaw = box[:7]
            gt_label_pos.append([x, y, z + h / 2 + 0.2])

            if gt_labels is not None:
                label_id = int(gt_labels[i])
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

    # ==========================
    # 3️⃣ Pred boxes
    # ==========================
    if boxes is not None:
        for i, box in enumerate(boxes):
            if scores is not None and scores[i] < score_thresh:
                continue

            box_lines = create_pyvista_box_lines(box)
            plotter.add_mesh(
                box_lines,
                color='red',
                line_width=2
            )

    plotter.show()



# def box_3d_to_corners(box):
#     """
#     box: [x, y, z, l, w, h, ry]
#     return: 8x3 corners in LiDAR frame
#     """
#     x, y, z, l, w, h, ry = box
#     # 先生成原始 box corners centered at origin
#     x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
#     y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
#     z_corners = [0,0,0,0,h,h,h,h]  # z from bottom=0 to top=h

#     corners = np.vstack([x_corners, y_corners, z_corners])  # 3x8

#     # 旋转
#     R = np.array([
#         [ np.cos(ry), -np.sin(ry), 0],
#         [ np.sin(ry),  np.cos(ry), 0],
#         [0,0,1]
#     ])
#     corners = R @ corners
#     # 平移到 box 中心
#     corners += np.array([[x],[y],[z]])
#     return corners.T  # 8x3



def show_multi_cam_images_from_path(img_paths, win_name='GT Cameras', num_cols=3):
    """
    img_paths: list[str | Path]
    num_cols: 每行多少张（6 张图时设为 3 → 两排）
    """

    imgs = []
    for p in img_paths:
        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            print(f'[WARN] Failed to read image: {p}')
            continue

        cam_name = p.parent.name
        cv2.putText(
            img,
            cam_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        imgs.append(img)

    if len(imgs) == 0:
        return

    # ========= 统一高度 =========
    h = min(img.shape[0] for img in imgs)
    imgs = [
        cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        for img in imgs
    ]

    # ========= 分行拼接 =========
    rows = []
    for i in range(0, len(imgs), num_cols):
        row = np.concatenate(imgs[i:i + num_cols], axis=1)
        rows.append(row)

    show = np.concatenate(rows, axis=0)

    cv2.imshow(win_name, show)
    cv2.waitKey(1)

    
def save_multi_cam_images_from_path(
    img_paths,
    save_dir='result_img',
    save_name=None,
    num_cols=3
):
    """
    img_paths: list[str | Path]
    save_dir: 保存目录
    save_name: 输出文件名（可选）
    num_cols: 每行多少张
    """

    if len(img_paths) == 0:
        print('[WARN] img_paths is empty.')
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= 自动生成文件名 =========
    if save_name is None:
        first_path = Path(img_paths[0])
        save_name = first_path.stem + '.jpg'

    imgs = []
    for p in img_paths:
        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            print(f'[WARN] Failed to read image: {p}')
            continue

        cam_name = p.parent.name
        cv2.putText(
            img,
            cam_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        imgs.append(img)

    if len(imgs) == 0:
        print('[WARN] No valid images, skip saving.')
        return

    # ========= 统一高度 =========
    h = min(img.shape[0] for img in imgs)
    imgs = [
        cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        for img in imgs
    ]

    # ========= 分行拼接 =========
    rows = []
    for i in range(0, len(imgs), num_cols):
        row = np.concatenate(imgs[i:i + num_cols], axis=1)
        rows.append(row)

    final_img = np.concatenate(rows, axis=0)

    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), final_img)
    print(f'[INFO] Saved multi-cam image to {save_path}')
    
def save_multi_cam_images_with_boxes(
    img_paths,
    save_dir='result_img',
    save_name=None,
    num_cols=3,
    gt_boxes=None,
    lidar2img=None
):
    """
    img_paths: list of image paths
    gt_boxes: N x 7, LiDAR frame
    lidar2img: array [num_cams, 4, 4]
    """

    if len(img_paths) == 0:
        print('[WARN] img_paths is empty.')
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= 自动生成文件名 =========
    if save_name is None:
        first_path = Path(img_paths[0])
        save_name = first_path.stem + '.jpg'

    imgs = []
    for cam_id, p in enumerate(img_paths):
        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            print(f'[WARN] Failed to read image: {p}')
            continue

        cam_name = p.parent.name
        cv2.putText(
            img,
            cam_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # ========= 投影 GT boxes =========
        if gt_boxes is not None and lidar2img is not None:
            # 🔑 lidar2img 是 4x4，这里只取前 3 行
            P = lidar2img[cam_id][:3, :]   # (3, 4)

            for box in gt_boxes:
                corners = box_3d_to_corners(box[0:7])  # 8x3
                corners_h = np.hstack(
                    [corners, np.ones((8, 1))]
                )  # 8x4

                uvz = P @ corners_h.T          # 3x8
                uv = (uvz[:2] / uvz[2:3]).T   # 8x2

                # 底面
                for k in range(4):
                    pt1 = tuple(uv[k].astype(int))
                    pt2 = tuple(uv[(k + 1) % 4].astype(int))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                # 顶面
                for k in range(4):
                    pt1 = tuple(uv[k + 4].astype(int))
                    pt2 = tuple(uv[(k + 1) % 4 + 4].astype(int))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                # 竖线
                for k in range(4):
                    pt1 = tuple(uv[k].astype(int))
                    pt2 = tuple(uv[k + 4].astype(int))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

        imgs.append(img)

    if len(imgs) == 0:
        print('[WARN] No valid images, skip saving.')
        return

    # ========= 统一高度 =========
    h = min(img.shape[0] for img in imgs)
    imgs = [
        cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        for img in imgs
    ]

    # ========= 分行拼接 =========
    rows = []
    for i in range(0, len(imgs), num_cols):
        rows.append(np.concatenate(imgs[i:i + num_cols], axis=1))

    final_img = np.concatenate(rows, axis=0)

    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), final_img)
    print(f'[INFO] Saved multi-cam image to {save_path}')

# def boxes3d_to_corners_lidar(boxes3d):
#     """
#     boxes3d: (N, 7) [x, y, z, dx, dy, dz, yaw]
#     return: (N, 8, 3)
#     """
#     corners_all = []

#     for box in boxes3d:
#         x, y, z, dx, dy, dz, yaw = box

#         # box local corners
#         x_c = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1])
#         y_c = dy / 2 * np.array([ 1, -1, -1,  1,  1, -1, -1,  1])
#         z_c = dz / 2 * np.array([ 1,  1,  1,  1, -1, -1, -1, -1])

#         corners = np.vstack([x_c, y_c, z_c])

#         # rotation around z
#         rot = np.array([
#             [ np.cos(yaw), -np.sin(yaw), 0],
#             [ np.sin(yaw),  np.cos(yaw), 0],
#             [ 0,            0,           1]
#         ])

#         corners = rot @ corners
#         corners = corners + np.array([[x], [y], [z]])

#         corners_all.append(corners.T)

#     return np.array(corners_all)

def create_3d_box_lines(corners, color=(1, 0, 0)):
    """
    corners: (8, 3)
    """
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        [color for _ in lines]
    )

    return line_set

def project_lidar_to_img(points_lidar, R, T, K):
    """
    points_lidar: (N, 3)
    R: (3, 3) sensor2lidar_rotation
    T: (3,)
    K: (3, 3) cam_intrinsic

    return: (N, 2), mask
    """
    # LiDAR → Camera (inverse)
    points_cam = (points_lidar - T) @ R.T

    # 只保留在相机前方的点
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    # Camera → Image
    points_img = points_cam @ K.T
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]

    return points_img[:, :2], mask


def load_lidar_points(lidar_path):
    if lidar_path.endswith('.bin'):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points[:, :3]
    elif lidar_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(lidar_path)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f'Unsupported lidar format: {lidar_path}')
    
def lidar_to_camera(points_lidar, sensor2lidar_R, sensor2lidar_T):
    """
    Args:
        points_lidar: (N, 3) or (N, >=3)
        sensor2lidar_R: (3, 3) cam -> lidar
        sensor2lidar_T: (3,)
    Returns:
        points_cam: (N, 3)
    """
    R = np.asarray(sensor2lidar_R).reshape(3, 3)
    T = np.asarray(sensor2lidar_T).reshape(3)

    R_lidar2cam = R.T
    t_lidar2cam = - R_lidar2cam @ T

    pts = points_lidar[:, :3]
    points_cam = pts @ R_lidar2cam.T + t_lidar2cam

    return points_cam

def draw_boxes_on_image(img, gt_corners_cam_list, K):
    img_draw = img.copy()

    for corners_cam in gt_corners_cam_list:
        # -----------------------
        # 只保留相机前方的点
        # -----------------------
        if np.all(corners_cam[:,2] <= 0):
            # box 完全在相机后面，跳过
            continue

        corners_cam_clipped = corners_cam.copy()
        corners_cam_clipped[corners_cam_clipped[:,2] <= 0, 2] = 1e-6  # 防止除0

        # 投影到图像
        corners_h = corners_cam_clipped.T  # (3,8)
        proj = K @ corners_h               # (3,8)
        proj = proj[:2] / proj[2:3]       # 除以 Z
        proj = proj.T.astype(int)          # (8,2)

        # 画线
        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        for i,j in edges:
            # 如果任何一个点在相机后方就跳过画这条线
            if corners_cam[i,2] <= 0 or corners_cam[j,2] <= 0:
                continue
            cv2.line(img_draw, tuple(proj[i]), tuple(proj[j]), color=(0,255,0), thickness=2)

    return img_draw


def visualize_cam_points_and_boxes(
    points_cam,
    gt_boxes,
    sensor2lidar_R,
    sensor2lidar_T,
    axis_len=5.0,
    box_color='green'
):
    """
    在 Camera 坐标系下可视化点云 + GT boxes

    Args:
        points_cam: (N,3) 相机坐标系点云
        gt_boxes: (M,7) LiDAR 坐标系 box
        sensor2lidar_R: (3,3) cam -> lidar
        sensor2lidar_T: (3,)
    """

    R = np.asarray(sensor2lidar_R)
    T = np.asarray(sensor2lidar_T)

    # --------------------------
    # 1️⃣ box -> camera corners
    # --------------------------
    gt_corners_cam_list = []
    for box in gt_boxes:
        corners_lidar = box_to_corners_3d(box, z_bottom=False)
        corners_cam = lidar_to_camera(corners_lidar, R, T)
        gt_corners_cam_list.append(corners_cam)

    # --------------------------
    # 2️⃣ PyVista 绘制
    # --------------------------
    plotter = pv.Plotter()
    plotter.set_background('black')

    # --- Camera 坐标轴 ---
    plotter.add_mesh(pv.Line((0,0,0), (axis_len,0,0)), color='red',   line_width=4)
    plotter.add_mesh(pv.Line((0,0,0), (0,axis_len,0)), color='green', line_width=4)
    plotter.add_mesh(pv.Line((0,0,0), (0,0,axis_len)), color='blue',  line_width=4)

    # --- 点云 ---
    plotter.add_points(
        points_cam,
        color='white',
        point_size=1,
        render_points_as_spheres=True
    )

    # --- GT boxes ---
    for corners_cam in gt_corners_cam_list:
        box_lines = create_pyvista_box_lines_from_corners(corners_cam)
        plotter.add_mesh(box_lines, color=box_color, line_width=2)

    plotter.show()
def draw_boxes_on_all_images(
    info,
    cam_names=None,
    mode='show',          # 只有 mode == 'save' 才保存
    save_dir='gt_img',
    save_name=None
):
    if cam_names is None:
        cam_names = list(info['cams'].keys())

    lidar_path = info['lidar_path']
    points = load_lidar_points(lidar_path)
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

        gt_corners_cam_list = []
        for box in gt_boxes:
            corners_lidar = box_to_corners_3d(box, z_bottom=False)
            corners_cam = lidar_to_camera(corners_lidar, R, T)
            gt_corners_cam_list.append(corners_cam)

        img_with_boxes = draw_boxes_on_image(img, gt_corners_cam_list, K)

        # ==========================
        # ⭐ 在图片左上角写相机名字
        # ==========================
        cv2.putText(
            img_with_boxes,
            cam_name,
            org=(20, 40),                 # 左上角位置
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 255),          # 黄色，黑底和实景都很清楚
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
        # ==========================
        # ⭐ 写 timestamp，字体稍小一点
        # ==========================
        timestamp_str = str(cam.get('timestamp', 'N/A'))  # 防止没有 timestamp
        cv2.putText(
            img_with_boxes,
            timestamp_str,
            org=(20, 70),                 # 相机名字下方
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,                # 比相机名字小
            color=(0, 255, 0),            # 绿色
            thickness=2,
            lineType=cv2.LINE_AA
        )

        images_with_boxes.append(img_with_boxes)


    if len(images_with_boxes) == 0:
        print('[ERROR] No valid images.')
        return

    # -------------------------
    # 拼接（两行，最多3列）
    # -------------------------
    rows = []
    num_per_row = 3

    for i in range(0, len(images_with_boxes), num_per_row):
        row_imgs = images_with_boxes[i:i + num_per_row]
        h_max = max(im.shape[0] for im in row_imgs)

        resized = []
        for im in row_imgs:
            if im.shape[0] != h_max:
                scale = h_max / im.shape[0]
                w_new = int(im.shape[1] * scale)
                im = cv2.resize(im, (w_new, h_max))
            resized.append(im)

        rows.append(cv2.hconcat(resized))

    final_img = cv2.vconcat(rows)

    # -------------------------
    # 保存 or 显示
    # -------------------------
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
    lidar_path=info['lidar_path']
    cams=info['cams']
    front_cam=cams['CAM_FRONT']
    front_cam_data_path=front_cam['data_path']
    sensor2lidar_rotation=front_cam['sensor2lidar_rotation']
    sensor2lidar_translation=front_cam['sensor2lidar_translation']
    cam_intrinsic=front_cam['cam_intrinsic']
    gt_boxes = np.asarray(info['gt_boxes'])[:, :7]
    gt_names=info['gt_names']
    
    
    points=load_lidar_points(lidar_path)
    print(lidar_path)
    print(front_cam_data_path)
    # visualize_black_bg_vista(points=points,gt_boxes=gt_boxes)
    
    img = cv2.imread(front_cam_data_path)

    R = np.array(front_cam['sensor2lidar_rotation'])
    T = np.array(front_cam['sensor2lidar_translation'])
    K = np.array(front_cam['cam_intrinsic'])

    points_cam = lidar_to_camera(points,R,T)

    # visualize_black_bg_vista(points=points_cam)
    # 3D 可视化（抽出来的函数）
    # visualize_cam_points_and_boxes(
    #     points_cam=points_cam,
    #     gt_boxes=gt_boxes,
    #     sensor2lidar_R=R,
    #     sensor2lidar_T=T,
    #     axis_len=5.0
    # )
    
    
    draw_boxes_on_all_images(
        info,
        # cam_names=[
        #     'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        #     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        # ],
        mode='save'
    )
    
    # img_with_boxes = draw_boxes_on_image(img, gt_corners_cam_list, K)
    # cv2.imshow('gt_boxes', img_with_boxes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
