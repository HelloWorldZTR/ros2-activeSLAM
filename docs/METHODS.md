# Active SLAM 方法实现说明

本文档面向第一次阅读代码的人，总结当前仓库中已经实现的 Active SLAM
探索方法。这里的 Active SLAM 主要指“下一步去哪里”的探索目标选择策略：
`slam_toolbox` 仍负责在线建图，Nav2 仍负责全局规划、局部控制、恢复行为和
`/cmd_vel` 发布，`exploration_coordinator` 只负责任务编排和目标选择。

## 1. 方法总览

主入口是 `src/activeslam/launch/slam.launch.py`。launch 启动 Gazebo、
TurtleBot3、`slam_toolbox`、Nav2、`activeslam.exploration_coordinator`，
并默认启动 evaluator 和 RViz。探索策略通过 `slam_mode` 选择：

| `slam_mode` | 方法 | 核心思想 |
| --- | --- | --- |
| `frontier` | Frontier baseline | 在共享 safe-frontier 池中按局部潜在未知面积和距离排序，选择第一个 Nav2 可达目标。 |
| `approx_graph` | 近似图优化探索 | 对多个 Nav2 可达 frontier 路径做 hallucination，用 TF 轨迹近似 pose graph 的 D-opt 风格目标排序。 |
| `gbsae` | GBSAE 先验图适配 | 加载 per-world topo-metric 先验图，按确定性路线推进，必要时用分配给 active vertex 的 frontier 打开道路。 |
| `gvd_gbsae` | 在线 GVD bootstrap + GBSAE | 先在 obstacle-only GVD 骨架上快速扫掠，扫掠比例达阈值后把 live GVD component 转成 GBSAE 图继续执行。 |
| `gvd_hierarchical` | 分层在线 GVD | 宏观层用 live GVD 的开放 TSP 路线遍历；每个最终经过的宏观节点触发局部 Region frontier 清扫；最后全局 frontier 扫尾。 |

旧 launch 参数 `exploration_strategy:=frontier|graph|graph_based` 仍兼容，其中
`graph` 和 `graph_based` 会映射为 `slam_mode:=approx_graph`，但 launch 会打印弃用警告。

## 2. 共同运行框架

五种方法共享同一个 ROS 节点、同一套状态机、同一套 frontier 检测和 Nav2 action
适配层。不同方法主要只改变“生成哪些候选”和“如何给可达候选排序”。

### 2.1 输入、状态和输出

`exploration_coordinator` 的输入是：

- `/map`：`slam_toolbox` 发布的 `OccupancyGrid`。
- TF `map -> base_footprint`：机器人当前位姿。
- Nav2 action server：路径预检、导航、旋转和短距离前探。

节点维护一个显式 `state` 字段：

```text
waiting_for_nav2 -> initial_spin -> idle -> selecting -> navigating
```

导航成功后可能进入 `aligning_frontier_probe` 和 `probing_frontier`。GVD 宏观阶段
若长时间没有有效平移，可能进入 `random_recovery_spin` 和
`random_recovery_drive`。标准 frontier 阶段在地图稳定且没有 frontier 后进入
`complete`。详细状态图见 `docs/exploration_coordinator_state_machine.md`。

除 `state` 外，还有三个关键子状态：

- `selection_kind`：区分当前正在检查 frontier、GBSAE 顶点、GVD 目标、层级局部
  frontier 或 GVD obstruction fallback。
- `current_navigation_kind`：区分当前 Nav2 目标来自 frontier、GBSAE 顶点、GVD
  bootstrap 还是层级宏观顶点。
- `gvd_phase`：在 GVD 方法中区分 `bootstrap`、`gbsae`、`macro`、`local_clear`
  和 `tail_cleanup`。

RViz 输出统一通过 `/goal_point`、`/frontier_markers` 和 `/pose_graph_markers`
发布。`/pose_graph_markers` 会根据模式显示近似 pose graph、GBSAE 先验图或
GVD 拓扑。

### 2.2 Nav2 action 边界

`nav2_backend.py` 是一个很薄的异步 action adapter：

- `/compute_path_to_pose`：检查候选目标可达性，并返回 Nav2 实际路径点和路径长度。
- `/navigate_to_pose`：执行最终目标。
- `/spin`：执行初始扫描、frontier probe 对齐和 GVD 随机恢复转向。
- `/drive_on_heading`：执行 frontier probe 和 GVD 随机恢复短距离直行。

每类 action 都有 generation guard。新一轮请求开始后，旧回调即使返回也会被丢弃
或取消，避免异步 action 结果污染当前状态。默认 YAML 中，单次路径请求超时为
`5s`，导航目标超时为 `30s`。失败目标附近 `0.6m` 会进入 `20s` 冷却。

探索节点不直接发布 `/cmd_vel`。所有运动都通过 Nav2 行为树、controller 或
behavior server 完成。

### 2.3 Frontier 检测与小 unknown 填补

`frontier_detector.py` 检测两类 frontier：

1. `unknown` frontier：free 栅格 `data == 0`，四邻域存在 unknown 栅格
   `data == -1`。
2. `open_edge` frontier：free 栅格位于当前 `/map` 数组边界，且没有被归入
   `unknown` frontier。它用于处理 `slam_toolbox` 发布紧凑地图时，地图矩形外部
   不属于 `/map.data` 的情况。

两类 mask 分别做八邻域聚类。默认普通 frontier 至少 `10px`。在检测前，detector
会尝试抑制由小 frontier 暴露的有界 unknown 碎片：如果 unknown 连通块很小、
不接触地图边界、也没有被大 frontier 暴露，就根据相邻 free/occupied 几何投票
填成低置信度 free 或 occupied。低置信度 free 不会作为安全落点，较大的 unknown
组件和接触 `/map` 边界的组件仍保持 unknown。

### 2.4 Safe Goal 与共享候选池

所有 frontier 驱动的方法都不直接把 frontier 质心交给 Nav2。每个 frontier
cluster 会先生成一个安全落点：

1. 从 cluster cells 中采样最多 `40` 个 seed。
2. 以 seed 到机器人方向为参考，在 seed 后方搜索 standoff 目标。
3. 目标必须位于已知 free 栅格，满足 obstacle clearance、地图边缘限制和
   Nav2 到达半径限制。
4. seed 到目标的短线段不能穿过 occupied 栅格。

默认 YAML 参数为：

- `frontier_goal_search_radius: 1.2`
- `frontier_goal_clearance: 0.20`
- `frontier_goal_standoff: 0.45`
- `nav2_goal_reach_radius: 0.25`

对每个安全落点，以 frontier seed 为中心在 `frontier_information_gain_radius=1.0m`
内估计潜在未知面积。`open_edge` frontier 会把圆形邻域中落在 `/map` 数组外的
区域也视作潜在未知。共享候选池的基础 utility 是：

```text
utility = information_gain / (distance(robot, safe_goal) + 0.1)
```

`frontier`、`approx_graph`、`gbsae` 和 `gvd_hierarchical` 的局部清扫都会复用
这套 safe goal 生成逻辑；差别在于后续如何过滤、分配和排序。

### 2.5 Frontier Probe

当 Nav2 到达 frontier 的地图内安全落点后，部分模式会继续调用 Nav2 behavior
向 frontier 外法线方向短距离前探：

- `frontier`、`approx_graph`、`gbsae` 默认启用 probe。
- `gvd_gbsae` 和 `gvd_hierarchical` 的宏观 GVD 导航默认关闭 probe。
- `gvd_hierarchical` 的局部 Region 清扫默认启用 probe。

普通 `unknown` frontier 默认前探 `0.45m`，速度 `0.20m/s`；`open_edge` frontier
默认前探 `2.0m`，速度 `0.30m/s`。如朝向未对准，先用 Nav2 `Spin` 对齐，再用
`DriveOnHeading` 前探。前探仍由 Nav2 做碰撞检查。

### 2.6 完成与重试

标准 frontier 阶段找不到 frontier 时，节点检查已知栅格数量是否稳定。默认要求
约 `10s` 窗口内增长率小于 `2%`，满足后进入 `complete`。否则等待
`frontier_retry_interval=3s` 后重试。

所有模式都会把不可达或失败目标加入冷却；GBSAE 的可选 loop revisit 不可达时会
直接跳过；GVD 宏观阶段若长时间没有有效平移，会用有限次数的随机 `Spin` +
`DriveOnHeading` 脱困，然后重新选点。

## 3. `frontier`：直接 Safe-Frontier Baseline

`frontier` 是默认模式，也是其他 frontier 驱动策略的基础。

关键流程：

1. 用共享框架生成 safe-frontier 候选池。
2. 按 `information_gain / distance` utility 排序。
3. 取前 `frontier_planning_attempts=12` 个候选。
4. 依次调用 Nav2 `/compute_path_to_pose`。
5. 第一个可达目标立即进入 `/navigate_to_pose`。
6. 成功到达后按配置执行 frontier probe；失败或超时则标记冷却并重试。

这个 baseline 不在所有可达候选之间比较 Nav2 路径长度。Nav2 路径只用于可达性
验证，目标排序来自局部信息增益和安全落点距离。

## 4. `approx_graph`：TF 轨迹近似图评分

`approx_graph` 使用同一批 safe-frontier 候选，但不会选择第一个可达目标，而是
对多个可达候选路径做图优化风格评分。

关键流程：

1. 协调器用 TF 轨迹维护一个本地 `WeightedPoseGraph`。
2. 每轮选择从共享候选池取前 `graph_max_frontier_candidates=8` 个候选。
3. 对每个候选调用 Nav2 `/compute_path_to_pose`，保留所有可达路径。
4. 对每条路径复制当前近似图，并沿路径插入 hallucinated 节点。
5. 计算 hallucinated graph 的 D-opt 风格分数，扣除路径长度惩罚。
6. 选择分数最高的可达 frontier 交给 Nav2 执行。

近似 pose graph 的节点插入条件是机器人相对上一节点移动至少 `0.5m`，或 yaw
变化至少 `0.35rad`。顺序 odometry edge 使用：

$$
\boldsymbol{\Omega}_{\mathrm{odom}}
= \frac{1}{2}
\operatorname{diag}
\left(
\sigma_x^2,\,
\sigma_y^2,\,
\sigma_\theta^2
\right)^{-1}
$$

默认协方差为 `(0.04, 0.04, 0.008)`。当新节点与历史节点距离不超过 `2.0m`，
且节点序号间隔至少 `20`，会添加最多 `3` 条几何近似 loop closure，权重倍率为
`1.5`。

候选路径 hallucination 时，路径上每隔约 `0.75m` 插入节点。节点附近 `1.5m`
内的 unknown 比例会提高顺序边 information；occupied 比例超过 `0.03` 时允许
hallucinate loop closure。最终目标为：

$$
S(p)
=
D_{\mathrm{approx}}(G_p)
-
\alpha_{\mathrm{path}}\,\ell(p)
$$

其中 $G_p$ 是插入候选路径后的图，$\ell(p)$ 是 Nav2 路径长度，
`graph_path_cost_weight` 默认 `0.05`。

这个图不是 `slam_toolbox` 的内部 pose graph。当前实现没有读取 `slam_toolbox`
实时节点、约束边或 Fisher information matrix；它只是用于目标排序的启发式近似。

## 5. `gbsae`：先验拓扑度量图路线

`gbsae` 加载 `activeslam_resource/maps/<world>.gbsae.json`。JSON 必须定义匹配
world、二维顶点和连通边；缺失或不合法会在启动早期失败。当前仓库包含
`slam_rooms.gbsae.json` 和 `slam_office.gbsae.json`。

关键流程：

1. 获得初始 TF 后，选择距离机器人最近的先验顶点作为起点。
2. 在先验图上构造确定性 nearest-unvisited greedy route，并展开中间最短路。
3. route 构造时评估可选谱 loop revisit，只有收益大于额外路径代价时才插入。
4. 执行 active route step：若目标顶点在当前地图中已知 free 且不在冷却中，先用
   Nav2 预检，再直接导航到该顶点。
5. 若 active 顶点当前不可直接导航，则把共享 safe-frontier 候选分配给尚未完成的
   先验顶点，只尝试分配给 active vertex 的 frontier。
6. active vertex 的 frontier 全不可达时推进 route，避免卡死。
7. 先验 route 完成后回到标准 frontier 流程补覆盖。

谱 loop revisit 使用 observed graph 的 weighted spanning tree D-opt 指标。先验边
的 information weight 为 $1/d$，其中 $d$ 是边两端顶点的欧氏距离。候选 loop 的
目标函数是：

$$
J_{\mathrm{loop}}
=
\Delta D
-
\beta_{\mathrm{loop}}\,2d_e
$$

`gbsae_loop_path_cost_weight` 默认 `0.01`。loop revisit 是路线级可选往返，不代表
`slam_toolbox` 后端已经建立了同样的回环约束。可选 revisit 不可达时会跳过。

## 6. `gvd_gbsae`：在线 GVD Bootstrap 后切换 GBSAE

`gvd_gbsae` 不依赖手工先验图。它先用 `config/gvd_worlds.yaml` 中的粗矩形边界
构建在线 obstacle-only GVD，再在扫掠到足够区域后把 live GVD component 转换成
GBSAE 图。

关键流程：

1. 在粗矩形边界内把已观测 occupied 栅格膨胀为障碍；free 和 unknown 都按可通行
   处理，得到 obstacle-only traversability。
2. 对 traversability 做 thinning，得到 GVD skeleton，并压缩成拓扑图。
3. 对候选 skeleton 目标执行局部 A*，A* 成本偏好 skeleton/centerline，惩罚偏离
   中轴线的路径。
4. 目标 utility 综合边界未扫掠方向、目标距离、历史轨迹重叠惩罚和当前朝向直行性。
5. 取前 `gvd_nav2_planning_attempts=8` 个 GVD 候选，仍用 Nav2 预检，首个可达目标
   交给 Nav2 执行。
6. 导航期间如果新观测障碍切断 active GVD path，取消当前 Nav2 目标，先尝试重新
   规划到原目标，再沿原路径从远到近选择可达 checkpoint 作为 fallback。
7. 轨迹扫掠面积达到 `gvd_sweep_switch_ratio=0.8` 后，用机器人所在 live component
   创建 GBSAE planner，进入与 `gbsae` 相同的 route/frontier 分配流程。

GVD 拓扑连通性不是盲信 skeleton component。压缩后若存在多个 component，会优先
查询 GVD 层连接；失败时用更宽松 clearance 的双向 A* 尝试桥接。缓存命中仍会按
当前地图复核。普通 GVD 构建、局部 GVD A* 和 active-path invalidation 使用
`gvd_obstacle_clearance=0.18`；仅 switching connection fallback A* 使用
`gvd_reconnection_clearance=0.04`，桥接结果仍需通过 Nav2 预检。

GVD 节点周围 unknown 比例高于阈值时会标为 unconfident。每个纯 unconfident 子图
会被压成 MST，减少 unknown 区域里过早想象出的闭环，但仍允许向 unknown 方向探索。

GVD bootstrap 若长时间没有有效平移，会启动有限随机恢复：随机 `Spin`，再短距离
`DriveOnHeading`。恢复动作仍由 Nav2 执行，不直接发布速度命令。

## 7. `gvd_hierarchical`：宏观 GVD + 局部 Region 清扫

`gvd_hierarchical` 也使用在线 obstacle-only GVD，但不切换到 GBSAE。它把 live GVD
作为宏观骨架，按开放 TSP 遍历骨架节点，并在宏观路线的“最后一次经过某节点”时
切入局部 frontier 清扫。

关键流程：

1. 按当前 `/map` 和粗世界边界重建 GVD topology。
2. 用空间半径迁移已探索节点、已清扫节点、active vertex 等状态，避免 live graph
   重建后状态丢失。
3. 从机器人所在 component 中取尚未局部清扫的节点作为 required targets。
4. 使用 NetworkX `traveling_salesman_problem(..., cycle=False)` 构造开放 TSP 路线；
   已清扫节点不再作为必访目标，但仍可作为最短路 transit。
5. 每次宏观目标仍先经 Nav2 预检，再导航到 selected GVD vertex。
6. 当 active vertex 已经到达它在当前 route 中的最后一次出现，并且机器人足够接近，
   `gvd_phase` 切换为 `local_clear`。
7. 局部清扫结束后标记该 vertex cleared，回到宏观 TSP；宏观节点全部清扫后进入
   `tail_cleanup`，使用标准 frontier 流程做全局扫尾。

地图更新会把宏观路线标为 dirty。宏观导航期间最多每 `0.5s` 重建一次路线；如果
更新暴露了应立即局部清扫的 final vertex，才会抢占当前 Nav2 宏观目标。普通 TSP
首步变化不会打断正在执行的宏观导航。局部清扫和 frontier probe 期间也会后台重建
GVD/TSP，但不会取消当前局部动作。

局部清扫使用 `local_free_flood_mask()` 在 active vertex 附近生成矩形 Region：

- Region 以 active vertex 附近的 free/unknown seed 为中心，最大半边长默认
  `2.5m`。
- Region 不能跨 occupied 栅格、粗世界边界或其他 live GVD vertex。
- 选择目标兼顾归一化面积和方形度，权重分别为
  `gvd_hierarchical_region_area_weight` 和
  `gvd_hierarchical_region_squareness_weight`。
- frontier cluster 必须接触 Region；safe goal 可以位于 Region 外。

局部候选先按 `cluster.size / distance` 的简单规则排序。默认还会为每个 Region
创建独立 approximate pose graph：对 Region 内可达路径用与 `approx_graph` 相同的
D-opt 风格评分，派发最高分目标。局部 graph 在离开 Region 后丢弃。

该模式依赖 NetworkX `>=3.4,<4`，因为 Ubuntu Jammy 仓库里的 `python3-networkx`
版本过旧。

## 8. 方法差异总结

| 特性 | `frontier` | `approx_graph` | `gbsae` | `gvd_gbsae` | `gvd_hierarchical` |
| --- | --- | --- | --- | --- | --- |
| 基础地图输入 | `/map` | `/map` + TF 近似图 | `/map` + 先验图 | `/map` + live GVD | `/map` + live GVD |
| 候选来源 | safe-frontier | safe-frontier | 先验顶点或分配 frontier | GVD skeleton，后续 GBSAE/frontier | GVD vertex，局部 Region frontier |
| Nav2 预检 | 顺序取首个可达 | 多路径评分后取最高 | 顶点和 frontier 都预检 | GVD 目标和 GBSAE 阶段都预检 | 宏观顶点和局部 frontier 都预检 |
| 图评分 | 无 | 路径 hallucination D-opt | 先验路线谱 loop revisit | GVD 目标 utility + GBSAE | 宏观 open-TSP + 局部 D-opt |
| 先验依赖 | 无 | 无 | `<world>.gbsae.json` | 粗世界边界 | 粗世界边界 |
| probe 默认 | 开 | 开 | 开 | 宏观关 | 宏观关，局部开 |
| 收尾 | 地图稳定结束 | 地图稳定结束 | route 后 frontier 补覆盖 | live GBSAE 后 frontier 补覆盖 | 宏观/局部后全局 frontier 扫尾 |

## 9. 关键文件

- `src/activeslam/launch/slam.launch.py`：主 launch 和 `slam_mode` 参数入口。
- `src/activeslam/config/exploration.yaml`：探索参数默认值。
- `src/activeslam/config/nav2_params.yaml`：Nav2 planner、controller、costmap 和 behavior 参数。
- `src/activeslam/activeslam/exploration_coordinator.py`：状态机、模式编排和 Nav2 action 调度。
- `src/activeslam/activeslam/nav2_backend.py`：异步 Nav2 action adapter。
- `src/activeslam/activeslam/frontier_detector.py`：frontier 检测、聚类和小 unknown 填补。
- `src/activeslam/activeslam/frontier_goal_utils.py`：safe goal、未知面积、外法线和失败冷却。
- `src/activeslam/activeslam/frontier_selection.py`：共享 frontier candidate 排序和 probe 开关规则。
- `src/activeslam/activeslam/graph_exploration.py`：近似 pose graph、D-opt 风格评分和可视化。
- `src/activeslam/activeslam/gbsae_exploration.py`：GBSAE 先验图加载、路线、loop revisit 和 frontier 分配。
- `src/activeslam/activeslam/gvd_exploration.py`：GVD topology、A*、switching connection、层级 TSP 和局部 Region。
- `src/activeslam/config/gvd_worlds.yaml`：GVD 方法使用的 per-world 粗矩形边界。
- `src/activeslam_resource/maps/*.gbsae.json`：GBSAE topo-metric 图资产。

## 10. 当前实现限制

- `slam_toolbox` 未做修改，探索层也没有读取其内部实时 pose graph。
- `approx_graph` 和层级局部清扫中的 D-opt 分数都是目标排序启发式，不代表 SLAM
  后端已加入相同约束。
- `gbsae` 依赖 per-world JSON 先验图；缺失资产会启动失败。
- `gvd_gbsae` 和 `gvd_hierarchical` 使用的是 obstacle-only GVD。unknown 被视为
  可通行探索空间，最终执行仍交给 Nav2 验证和控制。
- GVD switching connection 的 fallback A* 是连通性假设，RViz 中会以橙色桥接边
  与原始青色 GVD skeleton 区分；执行前仍需 Nav2 预检。
- `gvd_hierarchical` 的宏观 open-TSP 是近似路线，地图更新导致的首目标变化不会
  抢占当前宏观 Nav2 目标，除非需要立即切入局部清扫。
