# CLAUDE.md - mmdetection3d BEVFusion Project

## Quick Links
- [Documentation](https://mmdetection3d.readthedocs.io/en/latest/)
- [OpenMMLab](https://openmmlab.com/)

## Project Overview

This is a modified **mmdetection3d** repository with **BEVFusion** integration for multi-modal 3D object detection.

### Key Components

| Component | Location |
|-----------|----------|
| Main Detector | `projects/BEVFusion/bevfusion/bevfusion.py` |
| Configs | `projects/BEVFusion/configs/` |
| Inference Tools | `tools/infer.py`, `tools/utils/visualize_tools.py` |

### Architecture

**BEVFusion** is an multi-modal 3D detector that:
1. Extracts features from **LiDAR** (VoxelNet/SECOND) and **Camera** (Swin/ResNet) branches
2. Transforms image features to Bird's Eye View (BEV) using **DepthLSSTransform**
3. Fuses features via Convolutional fuser
4. Detects 3D objects with a shared head

### Current Modifications

| File | Purpose |
|------ |---------|
| `tools/infer.py` | Multi-modality inference with `--vis-mode` (lidar/multi/gt) |
| `tools/utils/visualize_tools.py` | Visualization utilities for 3D detections |
| `tools/count_boxes.py` | Box counting utility |
| `projects/BEVFusion/configs/bevfusion_lidar_camera_resnet_yx_kl.py` | Custom KL dataset config |

### Dataset: KL Dataset
Custom dataset with classes: `WheelCrane`, `ScreenContainer`, `Hopper`, `CraneTruck`, `DumpTruck`, `Excavator`, `WheelLoader`

### Common Commands

```bash
# Training
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_camera_resnet_yx_kl.py

# Inference with visualization
python tools/infer.py configs/bevfusion_lidar_camera_resnet_yx_kl.py <checkpoint> --vis-mode multi

# Count boxes
python tools/count_boxes.py <data_path>
```

### Debugging Tips

- **CUDA out of memory**: Reduce `samples_per_gpu` in config
- **Visualization**: Use `--vis-mode lidar` for point cloud only, `--vis-mode multi` for fused view
- **QAT**: Check `projects/BEVFusion/qat/` for quantization-aware training files

### File Structure
```
projects/BEVFusion/
├── bevfusion/          # BEVFusion implementation
│   ├── bevfusion.py   # Main detector class
│   └── ops/           # Voxelization ops
├── configs/           # Configuration files
└── qat/               # Quantization-aware training (input_data, ns_input_data, onnx models)
```
