# 专利附图（Mermaid 草稿）

> 用 Typora 或支持 Mermaid 的编辑器打开即可预览

---

## 图1：整体流程示意图

```mermaid
flowchart TB
    subgraph S1["S1: 目标类别分类"]
        CLS["港口检测目标"] --> DEF["方向确定类<br/>集卡/拖车/叉车/小汽车/行人..."]
        CLS --> SYM["方向对称类<br/>IGV / 轮胎吊"]
    end

    subgraph S2["S2: 训练阶段 — π 周期旋转损失"]
        INPUT_T["标注数据<br/>(含 180° 歧义)"] --> ENCODE["sin/cos 旋转编码"]
        ENCODE --> JUDGE{"属于方向对称类?"}
        JUDGE -- 否 --> NORMAL["常规 L1 损失"]
        JUDGE -- 是 --> COMPARE["比较 d_orig 与 d_flip"]
        COMPARE --> FLIP{"d_flip < d_orig ?"}
        FLIP -- 是 --> DO_FLIP["翻转真值:<br/>(sin_gt, cos_gt) → (-sin_gt, -cos_gt)"]
        FLIP -- 否 --> KEEP["保持原始真值"]
        DO_FLIP --> LOSS["计算 L1 损失"]
        KEEP --> LOSS
        NORMAL --> LOSS
    end

    subgraph S3["S3: 推理后处理 — 时序朝向异常检测"]
        DET["模型推理输出"] --> HIST["维护历史状态队列<br/>(最近 N 帧)"]
        HIST --> CLUSTER["朝向角度聚类<br/>确定稳定基准 θ_stable"]
        CLUSTER --> ANOMALY{"朝向异常?<br/>Δθ > τ 或 Δθ > k·max变化"}
        ANOMALY -- 异常 --> REPLACE["使用历史稳定朝向替代"]
        ANOMALY -- 正常 --> ACCEPT["接受当前朝向"]
        REPLACE --> OUTPUT["输出最终朝向"]
        ACCEPT --> OUTPUT
    end

    S1 --> S2
    S2 --> S3
```

---

## 图2：π 周期旋转损失原理示意图

```mermaid
flowchart LR
    subgraph CASE_A["方向确定类 (如集卡)"]
        direction TB
        A_PRED["预测: θ_pred"] --> A_LOSS["L1 损失<br/>|sin_pred - sin_gt| + |cos_pred - cos_gt|"]
        A_GT["真值: θ_gt"] --> A_LOSS
    end

    subgraph CASE_B["方向对称类 (如 IGV)"]
        direction TB
        B_PRED["预测: θ_pred"] --> B_D1["d_orig = |sin_pred - sin_gt|<br/>+ |cos_pred - cos_gt|"]
        B_GT["真值: θ_gt"] --> B_D1
        B_PRED --> B_D2["d_flip = |sin_pred + sin_gt|<br/>+ |cos_pred + cos_gt|"]
        B_GT --> B_D2
        B_D1 --> B_MIN["取 min(d_orig, d_flip)<br/>作为损失"]
        B_D2 --> B_MIN
    end
```

```mermaid
flowchart TB
    subgraph UNIT_CIRCLE["单位圆上的等价性示意"]
        direction LR
        THETA["θ_gt = 0°<br/>(sin=0, cos=1)"]
        THETA_PI["θ_gt + π = 180°<br/>(sin=0, cos=-1)"]
        EQ["对方向对称类:<br/>两者描述同一物理状态"]
        THETA --- EQ
        THETA_PI --- EQ
    end

    subgraph GRADIENT["梯度方向对比"]
        direction TB
        BEFORE["改进前: 梯度冲突"] --> G1["GT=0° 时: 梯度 → cos=+1"]
        BEFORE --> G2["GT=180° 时: 梯度 → cos=-1"]
        G1 --> CONFLICT["方向相反, 无法收敛 ✗"]
        G2 --> CONFLICT

        AFTER["改进后: 梯度一致"] --> G3["GT=0° → 保持, 梯度 → cos=+1"]
        AFTER --> G4["GT=180° → 翻转为0°, 梯度 → cos=+1"]
        G3 --> CONVERGE["方向一致, 稳定收敛 ✓"]
        G4 --> CONVERGE
    end
```

---

## 图3：时序朝向异常检测工作流程图

```mermaid
flowchart TB
    START["接收新检测帧"] --> VALID{"检测有效?"}
    VALID -- 否 --> REJECT["丢弃"]
    VALID -- 是 --> EXTRACT["提取历史 N 帧朝向角"]
    EXTRACT --> STATIC{"目标主要静止?<br/>(80%帧速度 < 0.1m/s)"}
    STATIC -- 否 --> ACCEPT_DIRECT["直接接受新检测朝向"]
    STATIC -- 是 --> CLUSTER["对历史朝向进行聚类"]
    CLUSTER --> FIND_MAX["选取最大簇<br/>中心 = θ_cluster"]
    FIND_MAX --> CHECK_SHIFT{"聚类中心偏移?<br/>|θ_cluster - θ_stable| > ε"}
    CHECK_SHIFT -- 是 --> COUNT["统计最近 M 帧<br/>偏离稳定基准的帧数"]
    CHECK_SHIFT -- 否 --> USE_STABLE["使用当前 θ_stable"]
    COUNT --> MAJORITY{"> 50% 偏离?"}
    MAJORITY -- 是 --> UPDATE["更新: θ_stable ← θ_cluster"]
    MAJORITY -- 否 --> USE_STABLE
    UPDATE --> CALC_DIFF["计算 Δθ = |θ_new - θ_stable|"]
    USE_STABLE --> CALC_DIFF
    CALC_DIFF --> ANOMALY{"Δθ > τ 或<br/>Δθ > k · max历史变化?"}
    ANOMALY -- 异常 --> REPLACE["输出 θ_stable 替代"]
    ANOMALY -- 正常 --> SMOOTH["移动平均平滑后输出"]
    REPLACE --> END_NODE["输出最终朝向"]
    SMOOTH --> END_NODE
    ACCEPT_DIRECT --> END_NODE
```

---

## 图4：朝向角度聚类示意图

```mermaid
flowchart TB
    subgraph INPUT["输入: 历史 N 帧朝向角"]
        H["θ₁=30° θ₂=31° θ₃=29° θ₄=210° θ₅=30° θ₆=32° θ₇=31° θ₈=209° θ₉=30° θ₁₀=31°"]
    end

    INPUT --> STEP1["遍历每个角度"]
    STEP1 --> CHECK{"与已有簇中心<br/>角度差 < ε (3°) ?"}
    CHECK -- 是 --> MERGE["归入该簇, 更新簇中心:<br/>θ = atan2(Σsin, Σcos)"]
    CHECK -- 否 --> NEW["创建新簇"]

    MERGE --> RESULT
    NEW --> RESULT

    subgraph RESULT["聚类结果"]
        C1["簇1: {30°,31°,29°,30°,32°,31°,30°,31°}<br/>中心=30.5°, 样本数=8 ← 主簇 ✓"]
        C2["簇2: {210°,209°}<br/>中心=209.5°, 样本数=2"]
    end

    RESULT --> SELECT["选取样本数最多的簇"]
    SELECT --> STABLE["稳定朝向基准 θ_stable = 30.5°"]
```

---

## 图5：实验对比（表格形式，需转为柱状图）

| 类别 | 改进前 AOE (rad) | 改进后 AOE (rad) | 降幅 |
|------|:---:|:---:|:---:|
| IGV-Full | 0.650 | < 0.20 | > 69% |
| IGV-Empty | 0.673 | < 0.20 | > 70% |
| WheelCrane | 1.105 | < 0.30 | > 73% |
| Car (参考) | 0.045 | 0.045 | - |
| Truck (参考) | 0.026 | 0.026 | - |

> 注: 改进后数据待训练完成后填入真实值
