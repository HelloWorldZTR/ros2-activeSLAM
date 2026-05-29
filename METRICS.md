# SLAM Evaluator Metrics

本文档总结 `src/activeslam/activeslam/slam_evaluator.py` 当前记录的评价指标，以及它们在代码中的计算公式。辅助计算函数主要位于 `src/activeslam/activeslam/slam_evaluator_utils.py`。

## 采样与输出

`slam_evaluator` 按 `sample_interval` 定时采样，默认 1.0 秒。每次采样会尽量记录：

- 估计轨迹：`trajectory_est.csv`
- Gazebo ground truth 轨迹：`trajectory_gt.csv`
- 覆盖率随时间变化：`coverage_time.csv`
- 覆盖率随路径长度变化：`coverage_path.csv`
- 当前汇总指标：`metrics.json`

节点退出时还会保存最终地图 `final_map.pgm`、`final_map.yaml`，并在有最终地图时计算最终 IoU 指标。

## 评价区域

覆盖率和 IoU 都只在评价区域内统计。评价区域由参数决定：

- 如果 `eval_min_x`、`eval_max_x`、`eval_min_y`、`eval_max_y` 全部是有限数，则使用手动边界。
- 否则从 world 文件中的 box collision obstacles 推导边界，并向外扩展 `eval_margin`。
- 如果无法从 world 提取障碍物，则使用整张 OccupancyGrid 地图。

设评价区域 mask 为：

```text
M(i, j) = 1, if cell center (x_j, y_i) is inside evaluation bounds
M(i, j) = 0, otherwise
```

其中：

```text
x_j = origin_x + (j + 0.5) * resolution
y_i = origin_y + (i + 0.5) * resolution
```

## Coverage

Coverage 表示评价区域内，SLAM 地图已经变为已知状态的栅格比例。OccupancyGrid 中 `-1` 表示未知，其他值都视为已知。

设地图栅格值为 `G(i, j)`，评价区域为 `M(i, j)`：

```text
known_cells = count((G(i, j) != -1) and M(i, j))
total_cells = count(M(i, j))
coverage = known_cells / total_cells
```

LaTeX 表达式：

```latex
\mathcal{K} = \{(i,j) \mid G_{ij} \neq -1,\ M_{ij}=1\}
```

```latex
\mathcal{E} = \{(i,j) \mid M_{ij}=1\}
```

```latex
\mathrm{Coverage}
= \frac{|\mathcal{K}|}{|\mathcal{E}|}
= \frac{\sum_{i,j}\mathbf{1}[G_{ij}\neq -1]\mathbf{1}[M_{ij}=1]}
       {\sum_{i,j}\mathbf{1}[M_{ij}=1]}
```

如果 `total_cells == 0`，代码返回：

```text
coverage = 0.0
known_cells = 0
total_cells = 0
```

相关输出：

- `coverage_time.csv`: `time_sec, coverage, known_cells, total_cells`
- `coverage_path.csv`: `path_length, coverage`
- `metrics.json`: `final_coverage`

`final_coverage` 是最近一次成功计算出的 coverage，不是整段实验的平均值。

## Total Path Length

`total_path_length` 使用估计轨迹计算，而不是 ground truth。每次 TF 成功查询到估计位姿后，累加当前估计位置与上一次估计位置之间的欧氏距离。

设第 `k` 个估计位置为：

```text
p_k = (x_k, y_k)
```

则总路径长度为：

```text
L = sum_{k=2..N} ||p_k - p_{k-1}||_2
  = sum_{k=2..N} sqrt((x_k - x_{k-1})^2 + (y_k - y_{k-1})^2)
```

第一帧估计位姿只用于初始化，不增加路径长度。

相关输出：

- `coverage_path.csv`: `path_length`
- `metrics.json`: `total_path_length`

## ATE RMSE

ATE 使用估计轨迹和 ground truth 轨迹计算，仅比较平面位置 `(x, y)`，不比较 yaw。

### 时间匹配

对每个估计轨迹样本 `e_k`，代码在 ground truth 样本中寻找时间戳最近的样本 `g_m`。如果二者时间差不超过 `max_dt = 0.2s`，则形成一组匹配：

```text
|t(e_k) - t(g_m)| <= 0.2
```

LaTeX 表达式：

```latex
m(k)=\arg\min_m |t(e_k)-t(g_m)|
```

```latex
\mathcal{P}=\{(e_k,g_{m(k)})\mid |t(e_k)-t(g_{m(k)})|\leq 0.2\}
```

没有匹配样本时，`ate_rmse` 不写入 `metrics.json`。

### 初始平移对齐

代码只做初始平移对齐，不做旋转或尺度对齐。设第一组匹配为 `(e_1, g_1)`，平移 offset 为：

```text
delta_x = x(g_1) - x(e_1)
delta_y = y(g_1) - y(e_1)
```

LaTeX 表达式：

```latex
\boldsymbol{\delta}
= \begin{bmatrix}\delta_x\\\delta_y\end{bmatrix}
= \begin{bmatrix}x(g_1)-x(e_1)\\y(g_1)-y(e_1)\end{bmatrix}
```

### 单点误差

每组匹配的 ATE 单点误差为：

```text
error_k = sqrt((x(e_k) + delta_x - x(g_k))^2
             + (y(e_k) + delta_y - y(g_k))^2)
```

LaTeX 表达式：

```latex
\epsilon_k
= \left\|
\begin{bmatrix}x(e_k)\\y(e_k)\end{bmatrix}
+ \boldsymbol{\delta}
- \begin{bmatrix}x(g_k)\\y(g_k)\end{bmatrix}
\right\|_2
```

### RMSE

设共有 `N` 组匹配：

```text
ATE_RMSE = sqrt((1 / N) * sum_{k=1..N} error_k^2)
```

LaTeX 表达式：

```latex
\mathrm{ATE}_{\mathrm{RMSE}}
= \sqrt{\frac{1}{N}\sum_{k=1}^{N}\epsilon_k^2}
```

相关输出：

- 图像中的 ATE 曲线使用每个 `error_k`
- `metrics.json`: `ate_rmse`, `ate_samples`

`ate_samples` 是成功时间匹配后的样本对数量。

## Occupied IoU

Occupied IoU 在节点退出保存最终输出时计算。ground truth occupied map 来自 world 文件中的 box collision obstacles 栅格化结果；预测 occupied map 来自最终 OccupancyGrid。

预测 occupied 条件：

```text
pred_occupied(i, j) = (G(i, j) >= 50) and known(i, j)
known(i, j) = (G(i, j) != -1) and M(i, j)
```

ground truth occupied 条件：

```text
gt_occupied(i, j) = rasterized_world_obstacle(i, j) and M(i, j)
```

公式：

```text
occupied_iou = count(pred_occupied and gt_occupied and known)
             / count((pred_occupied or gt_occupied) and known)
```

LaTeX 表达式：

```latex
\mathcal{P}_{occ}=\{(i,j)\mid G_{ij}\geq 50,\ G_{ij}\neq -1,\ M_{ij}=1\}
```

```latex
\mathcal{T}_{occ}=\{(i,j)\mid W_{ij}=1,\ M_{ij}=1\}
```

```latex
\mathrm{IoU}_{occ}
= \frac{|\mathcal{P}_{occ}\cap\mathcal{T}_{occ}\cap\mathcal{K}|}
       {|(\mathcal{P}_{occ}\cup\mathcal{T}_{occ})\cap\mathcal{K}|}
```

如果分母为 0，返回 `None`。

相关输出：

- `metrics.json`: `occupied_iou`

注意：IoU 只在 SLAM 已知区域 `known` 内计算，因此未知区域不会进入交集或并集。

## Free IoU

Free IoU 也在节点退出保存最终输出时计算。预测 free map 来自最终 OccupancyGrid，ground truth free map 是评价区域内非 obstacle 的区域。

预测 free 条件：

```text
pred_free(i, j) = (0 <= G(i, j) < 50) and known(i, j)
known(i, j) = (G(i, j) != -1) and M(i, j)
```

ground truth free 条件：

```text
gt_free(i, j) = not rasterized_world_obstacle(i, j) and M(i, j)
```

公式：

```text
free_iou = count(pred_free and gt_free and known)
         / count((pred_free or gt_free) and known)
```

LaTeX 表达式：

```latex
\mathcal{P}_{free}=\{(i,j)\mid 0\leq G_{ij}<50,\ G_{ij}\neq -1,\ M_{ij}=1\}
```

```latex
\mathcal{T}_{free}=\{(i,j)\mid W_{ij}=0,\ M_{ij}=1\}
```

```latex
\mathrm{IoU}_{free}
= \frac{|\mathcal{P}_{free}\cap\mathcal{T}_{free}\cap\mathcal{K}|}
       {|(\mathcal{P}_{free}\cup\mathcal{T}_{free})\cap\mathcal{K}|}
```

如果分母为 0，返回 `None`。

相关输出：

- `metrics.json`: `free_iou`

## Total Time

`total_time` 是 evaluator 节点运行到写出 `metrics.json` 时的 elapsed time：

```text
total_time = now_sec - start_time
```

相关输出：

- `metrics.json`: `total_time`

## Sample Counts

`metrics.json` 还记录样本数量，用于判断本次实验的数据完整性：

```text
estimated_samples = len(est_samples)
ground_truth_samples = len(gt_samples)
ate_samples = number of time-matched estimate/ground-truth pairs
```

其中 `ate_samples` 只在至少有一组 ATE 匹配时写入。

## metrics.json 字段

当前 `metrics.json` 可能包含：

- `final_coverage`
- `total_path_length`
- `total_time`
- `world_name`
- `evaluation_bounds`
- `gt_model_name`
- `gt_topic`
- `est_parent_frame`
- `est_child_frame`
- `estimated_samples`
- `ground_truth_samples`
- `ate_rmse`
- `ate_samples`
- `occupied_iou`
- `free_iou`

其中 `occupied_iou` 和 `free_iou` 只在最终保存阶段由最终地图计算；运行中间写出的 `metrics.json` 通常还不包含这两个字段。
