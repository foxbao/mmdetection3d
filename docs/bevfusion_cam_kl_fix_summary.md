# BEVFusion Camera-Only KL 配置修复总结

**日期**: 2026-03-30（更新于 2026-03-31）
**配置文件**: `projects/BEVFusion/configs/bevfusion_cam_resnet_kl.py`

## 问题现象

训练过程中 mAP 始终上不去，训练日志显示：
- `loss_bbox` 始终接近零（0.0000~0.0014），模型完全没有学习 bbox 回归
- 训练初期 loss 爆炸（iter 50 时 loss=13803, grad_norm=nan）
- `loss_heatmap` 下降极慢，从 ~30 到 ~26（2 个 epoch）

## 修改内容

### 1. `loss_bbox` reduction: `'none'` → `'mean'`（核心修复）

- **根本原因**: `reduction='none'` 导致 L1Loss 返回完整的 `[B, 500, 10]` 张量（大部分元素被 mask 为 0）
- mmengine 的 `parse_losses` 对它调用 `.mean()`，等于除以了 15000 而不是正样本数
- loss_bbox 的梯度被稀释了约 **1500 倍**，模型几乎无法学习 bbox 回归
- 官方 CenterPoint 配置（`configs/_base_/models/centerpoint_pillar02_second_secfpn_nus.py`）用的就是 `reduction='mean'`

### 2. ~~`code_size`: `9` → `10`~~（已撤回，保持 `9`）

- 最初认为 common_heads 有 vel(2) 所以 code_size 应为 10
- **实际验证发现 `code_size=9` 是正确的**：`CenterPointBBoxCoder.decode()` 输出 9 维 `[x, y, z, w, l, h, rot, vx, vy]`，与训练时 10 维的 `anno_box`（含 sub-pixel reg offset）不是同一回事
- 改成 10 后 eval 时报错：`AssertionError: ... must be 10, but got boxes with shape (N, 9)`
- **此项不影响训练**，`code_size` 仅在推理 decode 时使用

### 3. `clip_grad.max_norm`: `35` → `5`

- 防止训练初期 loss 爆炸（之前 iter 50 时 loss=13803, grad_norm=nan）
- MIT BEVFusion camera-only 参考配置使用 max_norm=5

### 4. `val_interval`: `5` → `2`

- 更早看到评估结果

### 5. `auto_scale_lr`: 启用

- 当 GPU 数量变化时自动调整学习率

### 6. `checkpoint`: 增加 `max_keep_ckpts=3, save_last=True`

- 防止磁盘被填满

## 备注

- KL 数据集没有速度标注（`gt_velocity` 全部为零），但 common_heads 保留了 vel 头，code_weights 中 vel 权重为 0.2，影响不大
- DepthLSSTransform 没有显式的深度监督 loss，深度预测质量完全依赖检测 loss 的反向传播
- 空间尺寸经过验证是匹配的：DepthLSSTransform 输出 120x120（downsample=2），LSSFPN 输出 120x120，CenterHead 期望 120x120（grid_size 960 / out_size_factor 8）
