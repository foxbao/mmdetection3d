"""Tests for KL box-derived occupancy targets."""

from __future__ import annotations

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.datasets.transforms import (ObjectNameFilter, ObjectRangeFilter,
                                         Pack3DDetInputs)
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from projects.KL8.transforms import (GenerateKLMultiFrameOccFromQueue,
                                     GenerateKLOccFromBoxes)


def _make_boxes(rows):
    return LiDARInstance3DBoxes(
        torch.tensor(rows, dtype=torch.float32).reshape(-1, 7), box_dim=7)


def _make_data_sample(rows=None, labels=None, track_ids=None):
    rows = [] if rows is None else rows
    labels = [] if labels is None else labels
    data = dict(
        bboxes_3d=_make_boxes(rows),
        labels_3d=torch.tensor(labels, dtype=torch.long))
    if track_ids is not None:
        data['track_ids_3d'] = torch.tensor(track_ids, dtype=torch.long)
    sample = Det3DDataSample()
    sample.gt_instances_3d = InstanceData(**data)
    return sample


def test_generate_kl_occ_from_axis_aligned_box():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-4, -4, -1, 4, 4, 3],
        occ_size=[8, 8, 4])
    results = transform(dict(
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert occ.shape == (8, 8, 4)
    assert occ.dtype == np.uint8
    assert np.unique(occ).tolist() == [0, 1]
    # 2m x 2m x 2m box over 1m voxels, using voxel-center inclusion.
    assert int((occ == 1).sum()) == 8


def test_generate_kl_occ_uses_label_offset_and_overlap_priority():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-4, -4, -1, 4, 4, 3],
        occ_size=[8, 8, 4])
    results = transform(dict(
        gt_bboxes_3d=_make_boxes([
            [0, 0, 0, 4, 4, 2, 0],
            [0, 0, 0, 2, 2, 2, 0],
        ]),
        gt_labels_3d=np.array([0, 2], dtype=np.int64)))

    occ = results['gt_occ']
    assert 1 in occ
    assert 3 in occ
    # Larger boxes are filled first; the smaller overlapping box keeps label 3.
    assert int((occ == 3).sum()) == 8


def test_generate_kl_occ_points_in_boxes_marks_unobserved_ignore():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-2, -2, 0, 2, 2, 2],
        occ_size=[4, 4, 2],
        mode='points_in_boxes')
    results = transform(dict(
        points=np.array([
            [-0.25, -0.25, 0.25, 0.0],
            [0.25, 0.25, 1.25, 0.0],
            [1.75, 1.75, 0.25, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert int((occ == 1).sum()) == 2
    assert int((occ == 255).sum()) == 6
    assert int((occ == 0).sum()) == occ.size - 8
    assert results['gt_occ_meta']['mode'] == 'points_in_boxes'


def test_generate_kl_occ_points_in_boxes_min_count_threshold():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-2, -2, 0, 2, 2, 2],
        occ_size=[4, 4, 2],
        mode='points_in_boxes',
        min_points_per_voxel=2)
    results = transform(dict(
        points=np.array([
            [-0.25, -0.25, 0.25, 0.0],
            [-0.20, -0.20, 0.20, 0.0],
            [0.25, 0.25, 1.25, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert int((occ == 1).sum()) == 1
    assert int((occ == 255).sum()) == 7


def test_generate_kl_occ_points_in_boxes_dilation_stays_inside_box():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-2, -2, 0, 2, 2, 2],
        occ_size=[4, 4, 2],
        mode='points_in_boxes',
        dilation_xy=1)
    results = transform(dict(
        points=np.array([[-0.25, -0.25, 0.25, 0.0]], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert int((occ == 1).sum()) == 4
    assert int((occ == 255).sum()) == 4
    assert int((occ == 0).sum()) == occ.size - 8


def test_generate_kl_occ_raycast_keeps_unknown_behind_hit():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 4, 1, 1],
        occ_size=[4, 2, 2],
        mode='raycast',
        fill_ground=False)
    results = transform(dict(
        points=np.array([[2.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[0, 1, 1] == 0
    assert occ[1, 1, 1] == 0
    assert occ[2, 1, 1] == 16
    assert occ[3, 1, 1] == 255
    assert results['gt_occ_meta']['mode'] == 'raycast'
    assert results['gt_occ_meta']['num_free_voxels'] > 0


def test_generate_kl_occ_raycast_semantic_overrides_scene_hit():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 4, 1, 1],
        occ_size=[4, 2, 2],
        mode='raycast',
        fill_ground=False)
    results = transform(dict(
        points=np.array([[2.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([[2.5, 0.0, -0.5, 1.0, 1.0, 1.0, 0.0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[2, 1, 1] == 1
    assert occ[3, 1, 1] == 255
    assert int((occ == 16).sum()) == 0
    assert int((occ == 17).sum()) == 0


def test_generate_kl_occ_raycast_removes_ground_under_obstacle_column():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 3, 1, 2],
        occ_size=[3, 2, 3],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55)
    results = transform(dict(
        points=np.array([
            [2.5, 0.0, -0.5, 0.0],
            [2.5, 0.0, 1.5, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[2, 1, 2] == 17
    assert occ[2, 1, 0] != 16


def test_generate_kl_occ_raycast_filters_sparse_obstacle_voxel():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 3, 1, 2],
        occ_size=[3, 2, 3],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_min_points_per_voxel=2)
    results = transform(dict(
        points=np.array([
            [2.5, 0.0, -0.5, 0.0],
            [2.5, 0.0, 1.5, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[2, 1, 2] == 255
    assert results['gt_occ_meta']['num_raw_obstacle_voxels'] == 1
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 0
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 1


def test_generate_kl_occ_raycast_filters_small_obstacle_component():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_min_component_voxels=2)
    results = transform(dict(
        points=np.array([
            [1.5, 0.0, -0.5, 0.0],
            [1.5, 0.0, 1.5, 0.0],
            [3.5, 0.0, -0.5, 0.0],
            [3.5, 0.0, 1.5, 0.0],
            [4.5, 0.0, -0.5, 0.0],
            [4.5, 0.0, 1.5, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[1, 1, 2] == 255
    assert occ[3, 1, 2] == 17
    assert occ[4, 1, 2] == 17
    assert results['gt_occ_meta']['num_raw_obstacle_voxels'] == 3
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 2
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 1


def test_generate_kl_occ_raycast_keeps_dense_small_obstacle_component():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_min_component_voxels=8,
        obstacle_small_component_keep_min_points=6)
    points = []
    for x in (1.5, 2.5, 3.5):
        points.append([x, 0.0, -0.5, 0.0])
        points.append([x, 0.0, 1.5, 0.0])
        points.append([x, 0.0, 1.5, 0.0])

    results = transform(dict(
        points=np.array(points, dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[1, 1, 2] == 17
    assert occ[2, 1, 2] == 17
    assert occ[3, 1, 2] == 17
    assert results['gt_occ_meta'][
        'obstacle_small_component_keep_min_points'] == 6
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 3
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 0


def test_generate_kl_occ_raycast_filters_thin_obstacle_component():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -2, -1, 8, 2, 3],
        occ_size=[8, 4, 4],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_min_component_voxels=2,
        obstacle_thin_component_min_major_span=3.0,
        obstacle_thin_component_max_minor_span=0.5,
        obstacle_thin_component_max_z_span=0.5)
    points = []
    for x in (1.5, 2.5, 3.5, 4.5):
        points.append([x, 0.5, -0.5, 0.0])
        points.append([x, 0.5, 1.5, 0.0])
    for y in (-0.5, 0.5):
        points.append([6.5, y, -0.5, 0.0])
        points.append([6.5, y, 1.5, 0.0])

    results = transform(dict(
        points=np.array(points, dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[1, 2, 2] == 255
    assert occ[2, 2, 2] == 255
    assert occ[3, 2, 2] == 255
    assert occ[4, 2, 2] == 255
    assert occ[6, 1, 2] == 17
    assert occ[6, 2, 2] == 17
    assert results['gt_occ_meta'][
        'obstacle_thin_component_min_major_span'] == 3.0
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 2
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 4


def test_generate_kl_occ_raycast_keeps_dense_thin_obstacle_component():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -2, -1, 8, 2, 3],
        occ_size=[8, 4, 4],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_min_component_voxels=2,
        obstacle_thin_component_min_major_span=3.0,
        obstacle_thin_component_max_minor_span=0.5,
        obstacle_thin_component_max_z_span=0.5,
        obstacle_thin_component_keep_min_points=8)
    points = []
    for x in (1.5, 2.5, 3.5, 4.5):
        points.append([x, 0.5, -0.5, 0.0])
        points.append([x, 0.5, 1.5, 0.0])
        points.append([x, 0.5, 1.5, 0.0])

    results = transform(dict(
        points=np.array(points, dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[1, 2, 2] == 17
    assert occ[2, 2, 2] == 17
    assert occ[3, 2, 2] == 17
    assert occ[4, 2, 2] == 17
    assert results['gt_occ_meta'][
        'obstacle_thin_component_keep_min_points'] == 8
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 4
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 0


def test_generate_kl_occ_raycast_ignores_obstacle_near_box_margin():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        mode='raycast',
        fill_ground=False,
        ground_height_threshold=0.55,
        obstacle_box_ignore_margin=1.0)
    results = transform(dict(
        points=np.array([
            [2.5, 0.0, 0.5, 0.0],
            [3.5, 0.0, -0.5, 0.0],
            [3.5, 0.0, 1.5, 0.0],
            [5.5, 0.0, -0.5, 0.0],
            [5.5, 0.0, 1.5, 0.0],
        ], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([[2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[2, 1, 1] == 1
    assert occ[3, 1, 2] == 255
    assert occ[5, 1, 2] == 17
    assert results['gt_occ_meta']['num_raw_obstacle_voxels'] == 2
    assert results['gt_occ_meta']['num_box_margin_obstacle_voxels'] == 1
    assert results['gt_occ_meta']['num_obstacle_voxels'] == 1
    assert results['gt_occ_meta']['num_filtered_obstacle_voxels'] == 1


def test_generate_kl_occ_ego_ignore_overwrites_labels():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-4, -4, -1, 4, 4, 3],
        occ_size=[8, 8, 4],
        ego_ignore_range=[-1, -1, -1, 1, 1, 3])
    results = transform(dict(
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    occ = results['gt_occ']
    assert int((occ == 1).sum()) == 0
    assert np.all(occ[3:5, 3:5, :] == 255)
    assert results['gt_occ_meta']['num_ego_ignore_voxels'] == 16


def test_generate_kl_occ_raycast_ego_ignore_overwrites_free_space():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[0, -1, -1, 4, 1, 1],
        occ_size=[4, 2, 2],
        mode='raycast',
        fill_ground=False,
        ego_ignore_range=[0, -1, -1, 2, 1, 1])
    results = transform(dict(
        points=np.array([[2.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        gt_bboxes_3d=_make_boxes([]),
        gt_labels_3d=np.empty((0, ), dtype=np.int64)))

    occ = results['gt_occ']
    assert occ[0, 1, 1] == 255
    assert occ[1, 1, 1] == 255
    assert occ[2, 1, 1] == 16
    assert results['gt_occ_meta']['num_ego_ignore_voxels'] == 8


def test_generate_kl_occ_current_frame_only_skips_history():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-4, -4, -1, 4, 4, 3],
        occ_size=[8, 8, 4],
        current_frame_only=True)
    results = transform(dict(
        _kl_is_current_frame=False,
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))

    assert 'gt_occ' not in results


def test_generate_kl_multiframe_occ_adds_history_scene_ground():
    transform = GenerateKLMultiFrameOccFromQueue(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        fill_ground=False)
    data_sample = _make_data_sample()
    data_sample.set_metainfo(
        dict(queue_metas={
            0: dict(ego2global=np.eye(4, dtype=np.float64)),
            1: dict(ego2global=np.eye(4, dtype=np.float64)),
        }))

    results = transform(dict(
        inputs=dict(
            points=np.array([[2.5, 0.0, -0.5, 0.0]], dtype=np.float32),
            history_points=[
                np.array([[4.5, 0.0, -0.5, 0.0]], dtype=np.float32)
            ]),
        data_samples=data_sample,
        _kl_history_data_samples=[_make_data_sample()]))

    occ = results['data_samples'].gt_pts_seg.occ.numpy()
    assert occ[2, 1, 0] == 16
    assert occ[4, 1, 0] == 16
    meta = results['data_samples'].metainfo['gt_occ_meta']
    assert meta['source'] == 'queue_static_multiframe_points_raycast'
    assert meta['num_history_frames'] == 1
    assert meta['num_history_scene_points'] == 1


def test_generate_kl_multiframe_occ_removes_history_box_points():
    transform = GenerateKLMultiFrameOccFromQueue(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        fill_ground=False)
    data_sample = _make_data_sample()
    data_sample.set_metainfo(
        dict(queue_metas={
            0: dict(ego2global=np.eye(4, dtype=np.float64)),
            1: dict(ego2global=np.eye(4, dtype=np.float64)),
        }))
    history_sample = _make_data_sample(
        rows=[[4.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]],
        labels=[0])

    results = transform(dict(
        inputs=dict(
            points=np.array([[2.5, 0.0, -0.5, 0.0]], dtype=np.float32),
            history_points=[
                np.array([[4.5, 0.0, 1.5, 0.0]], dtype=np.float32)
            ]),
        data_samples=data_sample,
        _kl_history_data_samples=[history_sample]))

    occ = results['data_samples'].gt_pts_seg.occ.numpy()
    assert occ[4, 1, 2] == 255
    meta = results['data_samples'].metainfo['gt_occ_meta']
    assert meta['num_history_points'] == 1
    assert meta['num_history_scene_points'] == 0


def test_generate_kl_multiframe_occ_aligns_dynamic_track_points():
    transform = GenerateKLMultiFrameOccFromQueue(
        point_cloud_range=[0, -1, -1, 6, 1, 2],
        occ_size=[6, 2, 3],
        fill_ground=False,
        aggregate_dynamic_instances=True)
    data_sample = _make_data_sample(
        rows=[[4.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
        labels=[0],
        track_ids=[42])
    data_sample.set_metainfo(
        dict(queue_metas={
            0: dict(ego2global=np.eye(4, dtype=np.float64)),
            1: dict(ego2global=np.eye(4, dtype=np.float64)),
        }))
    history_sample = _make_data_sample(
        rows=[[2.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
        labels=[0],
        track_ids=[42])

    results = transform(dict(
        inputs=dict(
            points=np.array([[1.5, 0.0, -0.5, 0.0]], dtype=np.float32),
            history_points=[
                np.array([[2.5, 0.0, 0.5, 0.0]], dtype=np.float32)
            ]),
        data_samples=data_sample,
        _kl_history_data_samples=[history_sample]))

    occ = results['data_samples'].gt_pts_seg.occ.numpy()
    assert occ[4, 1, 1] == 1
    meta = results['data_samples'].metainfo['gt_occ_meta']
    assert meta['aggregate_dynamic_instances']
    assert meta['num_dynamic_instances'] == 1
    assert meta['num_dynamic_history_points'] == 1
    assert meta['num_dynamic_points'] == 1


def test_object_filters_keep_gt_track_ids_aligned():
    range_filter = ObjectRangeFilter(
        point_cloud_range=[-2, -2, -1, 2, 2, 2])
    range_results = range_filter(dict(
        gt_bboxes_3d=_make_boxes([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ]),
        gt_labels_3d=np.array([0, 1], dtype=np.int64),
        gt_track_ids_3d=np.array([10, 11], dtype=np.int64)))
    assert range_results['gt_track_ids_3d'].tolist() == [10]

    name_filter = ObjectNameFilter(classes=['Pedestrian'])
    name_results = name_filter(dict(
        gt_bboxes_3d=_make_boxes([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ]),
        gt_labels_3d=np.array([0, 1], dtype=np.int64),
        gt_track_ids_3d=np.array([20, 21], dtype=np.int64)))
    assert name_results['gt_track_ids_3d'].tolist() == [20]


def test_pack_3d_det_inputs_carries_gt_occ():
    transform = GenerateKLOccFromBoxes(
        point_cloud_range=[-4, -4, -1, 4, 4, 3],
        occ_size=[8, 8, 4])
    results = transform(dict(
        gt_bboxes_3d=_make_boxes([[0, 0, 0, 2, 2, 2, 0]]),
        gt_labels_3d=np.array([0], dtype=np.int64)))
    results['gt_occ_meta'] = dict(marker='kept')

    packed = Pack3DDetInputs(keys=('gt_occ',))(results)
    gt_pts_seg = packed['data_samples'].gt_pts_seg
    assert hasattr(gt_pts_seg, 'occ')
    assert tuple(gt_pts_seg.occ.shape) == (8, 8, 4)
    assert gt_pts_seg.occ.dtype == torch.uint8
    assert packed['data_samples'].metainfo['gt_occ_meta']['marker'] == 'kept'


def test_pack_3d_det_inputs_carries_gt_track_ids():
    packed = Pack3DDetInputs(keys=('gt_track_ids_3d', ))(
        dict(gt_track_ids_3d=np.array([42, 43], dtype=np.int64)))

    track_ids = packed['data_samples'].gt_instances_3d.track_ids_3d
    assert track_ids.tolist() == [42, 43]
