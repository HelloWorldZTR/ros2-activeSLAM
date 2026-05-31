# ROS2 Active SLAM

## 运行 SLAM

在容器内执行：

```bash
cd /home/ubuntu/ros2_ws
source setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gvd_hierarchical
```

探索节点会先调用 Nav2 `Spin` 完成一圈初始扫描，再持续选择 frontier
并通过 Nav2 导航。Nav2 负责全局规划、局部控制和恢复行为。主 launch 默认
同时启动 evaluator、RViz 和 evaluator 实时曲线，无需额外终端。

在bc01执行

```bash
cd /home/psirobot/projects/ros2_ws/src
source setup.sh
cb
s
ros2 launch activeslam slam.launch.py map:=slam_rooms
```

## 查看 SLAM 地图

主 launch 默认使用项目自带 RViz 配置，直接显示 `/map`、`/scan`、机器人、
`/plan`、`/goal_point`、`/frontier_markers`、`/pose_graph_markers` 和半透明
`/global_costmap/costmap`。Fixed Frame 已设为 `map`。

`/map` 只包含 SLAM 当前发布的有限矩形栅格。矩形内部的灰色区域是
`data == -1` 的 unknown 栅格；矩形外部的深色区域只是 RViz 背景，不属于
`/map.data`。探索节点默认将接触该外部边界的 free 格作为开放边缘 frontier：
普通 unknown frontier marker 为绿色，开放边缘 marker 为青色。两类 frontier
按局部潜在 unknown 面积和安全落点距离统一排序，再交给 Nav2 检查可达性。

## 可选参数

远端或无图形界面环境中，显式关闭 Gazebo GUI、RViz 和 evaluator 实时曲线：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms \
  gui:=false run_rviz:=false plot_live:=false save_plots:=false
```

如需按需关闭 evaluator 或 RViz：

```bash
ros2 launch activeslam slam.launch.py run_evaluator:=false run_rviz:=false
```

如果需要指定 TurtleBot3 型号，例如 `waffle`：

```bash
ros2 launch activeslam slam.launch.py turtlebot3_model:=waffle
```

如果需要选择 Gazebo 地图，例如 TurtleBot3 house：
无需加入world后缀

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms x_pose:=-2.0 y_pose:=-0.5
```

Evaluator 默认随 launch 启动，但只对 `slam_*` 地图生成可信指标。这些地图将
墙体作为内嵌 box collision 保存。多数 `turtlebot3_*` 地图依赖
`model://...` include，当前 evaluator 不会递归解析模型目录；选择此类地图
时会自动跳过 evaluator，并在终端和 `logs/evaluator_skipped.log` 中记录原因。

项目还包含基于
[`Dataset-of-Gazebo-Worlds-Models-and-Maps`](https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps)
Office 地图生成的复杂测试场景。该版本将二维占据区域转换为内嵌 box collision，
因此仍可计算 coverage 和 IoU。launch 默认使用 `PublicBathroomB` 内部出生点
`(12.1, 1.5)`：

```bash
ros2 launch activeslam slam.launch.py map:=slam_office
```

Office 也包含可继续手工校对的初版 GBSAE 先验图：

```bash
ros2 launch activeslam slam.launch.py map:=slam_office slam_mode:=gbsae
```

探索策略通过 `slam_mode` 选择。YAML 默认值是 `frontier`；不传 launch
覆盖参数时读取 `activeslam/config/exploration.yaml`。现有模式：

- `frontier`：按 frontier 信息增益和安全落点距离排序。
- `approx_graph`：使用 TF 轨迹近似 pose graph，并用 D-opt 风格评分选择 frontier。
- `gbsae`：使用先验拓扑度量图、frontier 分配、路线推进和谱分析 loop revisit。
- `gvd_gbsae`：先基于雷达已观测障碍构建在线 GVD 骨架并快速扫掠，再切换到
  由实时骨架生成的 GBSAE 路线。第一阶段将 unknown 和 free 都视为可通行区域，
  但仍由 Nav2 做最终路径规划、DWB 控制和恢复。
- `gvd_hierarchical`：单次分层探索。宏观层贪心遍历在线 GVD 节点，在叶节点或
  最多剩余一个未探索分支的节点处切入矩形 flood 限定的局部 frontier 清空，最后执行全局
  frontier 扫尾。

启用近似图评分策略：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=approx_graph
```

GBSAE 当前为 `slam_rooms` 提供手工校对的先验图，并为 `slam_office` 提供自动
生成的初版先验图：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gbsae
MAP=slam_rooms SLAM_MODE=gbsae RUN_SECONDS=120 ./run.zsh
```

无需手工先验图的在线 GVD + GBSAE 模式：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gvd_gbsae
```

该模式使用 `config/gvd_worlds.yaml` 中不含墙体结构的粗矩形边界。GVD 快速扫掠
目标综合考虑尚未扫过的边框方向、距离、与历史轨迹管道的重叠面积以及当前朝向；
局部 A* 只把已观测障碍视为墙，并对偏离 GVD 中轴线的路径增加连续代价，避免
为了缩短几步而贴墙行驶。若新扫描出的障碍切断当前路径，探索节点会立即取消
Nav2 目标并重新选点。轨迹扫掠面积达到配置阈值后切换到实时 GBSAE。

实时拓扑图不会盲信细化后的 GVD 连通性。骨架压缩后若仍存在多个 component，
节点会优先查询缓存中的 GVD 连接；GVD 无法补边时，再使用双向 A* 在障碍栅格上
寻找可通行桥接边。缓存命中仍会重新校验路径，避免新观测障碍使旧连接失效。
连通判断、局部 GVD A* 和路径失效检测统一使用 `gvd_obstacle_clearance: 0.18`
膨胀已知障碍，与 TurtleBot 圆形 footprint 半径一致；单像素窄缝不会再被当作
可执行通路。可用 `gvd_switching_connections_enabled` 关闭切换连接机制进行
ablation study。

如果 GVD bootstrap 阶段 TF 位姿长时间没有产生足够的有效平移，节点会启动有界的轻量
随机脱困：随机调用一次 Nav2 `Spin`，再调用短距离 `DriveOnHeading`。原地旋转
和微小位置抖动不会刷新进展计时器。恢复动作仍使用 Nav2 local costmap 做碰撞
检查，不会让探索节点直接发布 `/cmd_vel`。对应参数均以 `gvd_stuck_*` 和
`gvd_random_recovery_*` 开头，可用于 ablation study。空闲、选点、预检和导航
执行期间统一使用同一套进展 watchdog，避免机器人静止在原地重复空选点。

无需切换到 GBSAE 第二阶段、直接完成宏观到微观一次探索的实验模式：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gvd_hierarchical
```

该模式按空间半径迁移 live GVD 重建前后的节点状态。宏观节点使用局部 unknown
面积除以拓扑距离的朴素 utility；局部清空从叶节点或近耗尽分支出发，生成多个受墙体、
粗先验边界和其他 live GVD 顶点约束的贪心矩形 Region，并选择最接近正方形的
候选。Region 可以覆盖 unknown，但不能跨过占据栅格；它表示局部探索范围，不表示
已验证可通行空间。Region 内仍按 `frontier cluster size / distance` 依次派发
Nav2 目标，并强制 safe goal 位于 Region 内部。
宏观图遍历结束后，剩余 frontier 会由全局贪心扫尾收集。RViz 中绿色节点表示
宏观已探索，黄绿色节点表示局部已清空，红色节点表示当前宏观位置；局部清空
期间的半透明青色格子表示当前选中的矩形 Region，已经清扫完成的 Region 会保留
青色矩形轮廓。Region 使用生成时的 `/map` geometry 快照，因此地图扩张后不会
因栅格 origin 或尺寸变化发生错位。

选择其他地图并启用 `gbsae` 会在启动早期报告缺少对应
`<world>.gbsae.json`。旧命令中的 `exploration_strategy:=frontier|graph`
仍兼容，其中 `graph` 映射到 `approx_graph`，但会打印弃用警告。

Nav2 参数默认读取 `activeslam/config/nav2_params.yaml`。如需使用其他配置：

```bash
ros2 launch activeslam slam.launch.py nav2_params_file:=/path/to/nav2_params.yaml
```

开放地图边缘 frontier 默认启用。如需对照旧行为，可在
`activeslam/config/exploration.yaml` 中将
`frontier_include_open_map_edges` 设为 `false`。

`frontier`、`approx_graph` 和 `gbsae` 默认启用全部 frontier probe。当 Nav2
到达地图内部安全落点后，探索节点会调用 Nav2 `Spin` 对准局部外法线，再调用
`DriveOnHeading` 低速前探。普通 unknown frontier 使用保守的 `0.45m` 前探打开
门口等狭窄入口；开放边缘 frontier 使用 `2.0m` 前探扩展 `/map` 边界。

`gvd_gbsae` 和 `gvd_hierarchical` 默认关闭全部 frontier probe，避免 GVD 轨迹
执行期间额外前探改变宏观遍历行为。可使用 `frontier_mode_probes_enabled` 和
`gvd_mode_probes_enabled` 覆盖模式默认值，再使用
`frontier_unknown_probe_enabled` 与 `frontier_open_edge_probe_enabled` 单独关闭
某一类 probe，便于消融实验。前探速度由 Behavior Server 直接发布到 `/cmd_vel`；
它不经过 velocity smoother，但仍使用 Nav2 local costmap 的碰撞检查。探索节点
本身不会直接发布速度命令。
