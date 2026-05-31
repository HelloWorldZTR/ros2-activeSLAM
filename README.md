# ROS2 Active SLAM

## 运行 SLAM

在容器内执行：

```bash
cd /home/ubuntu/ros2_ws
source setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py map:=slam_office slam_mode:=online_gbsae
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
- `online_gbsae`：先用偏向外扩的 frontier 启发式粗探，再从 SLAM map 在线提取
  骨架拓扑图并运行 GBSAE。该模式不读取房间结构先验。

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

在线自举拓扑模式仅使用每张地图的粗矩形边界。边界保存在
`activeslam/config/online_gbsae_worlds.yaml`，用于估算粗探比例和剩余 unknown
方向，不包含墙、门或房间结构。已知面积达到 `50%` 后，节点使用纯 NumPy
thinning 从已知 free 区域提取；高质量未选 frontier 会保留为有限距离的虚拟
branch 叶节点。bootstrap 评分不奖励远距离目标，也不使用 pose graph novelty。
Nav2 返回路径穿越已知区域的长度占比越高，扣分越多。bootstrap 到达 frontier
后默认使用 Nav2 `Spin + DriveOnHeading` 前探 `2.0m`，执行期碰撞检查仍由
Behavior Server 负责：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms_corridor slam_mode:=online_gbsae
MAP=slam_rooms_corridor SLAM_MODE=online_gbsae RUN_SECONDS=180 ./run.zsh
```

在线模式默认提供三个 ablation 开关：

```yaml
online_gbsae_directional_prior_enabled: true
online_gbsae_branch_hypotheses_enabled: true
online_gbsae_explored_migration_enabled: true
online_gbsae_bootstrap_probe_enabled: true
```

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

当 Nav2 到达 frontier 的地图内部安全落点后，探索节点会调用 Nav2 `Spin`
对准局部外法线，再调用 `DriveOnHeading` 低速前探。普通 unknown frontier
使用保守的 `0.45m` 前探打开门口等狭窄入口；开放边缘 frontier 使用 `2.0m`
前探扩展 `/map` 边界。前探速度由 Behavior Server 直接发布到 `/cmd_vel`；
它不经过 velocity smoother，但仍使用 Nav2 local costmap 的碰撞检查。
探索节点本身不会直接发布速度命令。
