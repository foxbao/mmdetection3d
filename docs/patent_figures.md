# 专利技术交底书附图说明

当前 Word 技术交底书中使用的是图片附图，不再使用 Mermaid 自动排版图。其中补充图A为港口方向对称目标结构示例图；图1至图5为黑白线框流程图或原理图，由 `tools/render_patent_flowcharts.py` 按固定坐标生成；图6为点云鸟瞰对比图，由 `tools/find_orientation_contrast_frames.py` 生成后整理为黑白线型标注版本。渲染结果位于 `docs/patent_figures_generated/`，结构示例图位于 `docs/scene_image/`。

## 补充图A：方向对称目标结构示例图

文件：

- `docs/scene_image/IGV_1024_cropped.jpg`
- `docs/scene_image/WheelCrane_1024.jpg`

展示智能引导车（IGV）和轮胎吊（WheelCrane）的结构对称性示例，用于说明该类港口作业目标在单帧外观或 BEV 感知结果中前后差异较弱。

## 图1：整体流程示意图

文件：`docs/patent_figures_generated/fig1_overall_flow.png`

展示目标类别划分、训练阶段 `π` 周期旋转损失、推理阶段等价方向对齐、时序朝向稳定化以及最终几何一致化输出之间的关系。

## 图2：`π` 周期旋转损失原理示意图

文件：`docs/patent_figures_generated/fig2_pi_loss.png`

展示方向确定类目标采用常规唯一朝向监督，方向对称类目标在原始真值编码和 `π` 翻转等价编码之间选择更接近预测的编码作为监督目标。

## 图3：训练阶段目标编码翻转流程图

文件：`docs/patent_figures_generated/fig3_training_flip.png`

展示根据目标类别判断是否启用 `π` 周期处理，并根据 `d_orig` 与 `d_flip` 的比较结果选择旋转监督目标。

## 图4：时序朝向异常检测流程图

文件：`docs/patent_figures_generated/fig4_temporal_anomaly.png`

展示推理阶段对方向对称目标维护历史状态队列、进行等价方向对齐、聚类确定稳定基准、判定异常跳变、修正或平滑朝向，并同步更新方向向量和三维框的流程。

## 图5：朝向角度聚类示意图

文件：`docs/patent_figures_generated/fig5_orientation_cluster.png`

展示历史朝向中可能出现的两个相差约 `180°` 的角度簇，以及选取主簇作为稳定朝向基准的过程。

## 图6：单帧朝向可视化对比图组

文件：

- `docs/patent_figures_generated/fig6_1_igv_empty.png`
- `docs/patent_figures_generated/fig6_2_igv_empty.png`
- `docs/patent_figures_generated/fig6_3_wheelcrane.png`
- `docs/patent_figures_generated/fig6_4_wheelcrane.png`

展示 IGV-Empty（空载智能引导车）和 WheelCrane（轮胎吊）单帧点云鸟瞰图中的对照方案预测框、本发明方案预测框和真值框对比。图中 `GT` 表示真值框，`A` 表示对照方案预测框，`B` 表示本发明方案预测框，三者采用不同线型和标号区分。
