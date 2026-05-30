# ROS2 Active SLAM

## 运行 SLAM

在容器内执行：

```bash
cd /home/ubuntu/ros2_ws
source setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py map:=slam_rooms
ros2 run activeslam slam_evaluator --ros-args -p use_sim_time:=true -p world_name:=slam_rooms
```

探索节点会先调用 Nav2 `Spin` 完成一圈初始扫描，再持续选择 frontier
并通过 Nav2 导航。Nav2 负责全局规划、局部控制和恢复行为。

在bc01执行

```bash
cd /home/psirobot/projects/ros2_ws/src
source setup.sh
cb
s
# in terminal 2
ros2 launch activeslam slam.launch.py map:=slam_rooms
# in terminal 1
ros2 run activeslam slam_evaluator --ros-args -p use_sim_time:=true -p world_name:=slam_rooms
```

## 查看 SLAM 地图

直接启动 RViz（推荐）

在容器里新开终端执行：

```bash
rviz2
```

RViz 打开后：

1. 左侧 Displays 点击 Add
2. 选择 Map
3. Topic 选 /map
4. 把 Fixed Frame 设为 map（或 odom，一般 map 更合适）

`/map` 只包含 SLAM 当前发布的有限矩形栅格。矩形内部的灰色区域是
`data == -1` 的 unknown 栅格；矩形外部的深色区域只是 RViz 背景，不属于
`/map.data`。探索节点默认将接触该外部边界的 free 格作为开放边缘 frontier：
普通 unknown frontier marker 为绿色，开放边缘 marker 为青色。两类 frontier
按局部潜在 unknown 面积和安全落点距离统一排序，再交给 Nav2 检查可达性。

## 可选参数

如果需要指定 TurtleBot3 型号，例如 `waffle`：

```bash
ros2 launch activeslam slam.launch.py turtlebot3_model:=waffle
```

如果需要选择 Gazebo 地图，例如 TurtleBot3 house：
无需加入world后缀

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms x_pose:=-2.0 y_pose:=-0.5
```

如果需要启用图评分策略：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms exploration_strategy:=graph
```

Nav2 参数默认读取 `activeslam/config/nav2_params.yaml`。如需使用其他配置：

```bash
ros2 launch activeslam slam.launch.py nav2_params_file:=/path/to/nav2_params.yaml
```

开放地图边缘 frontier 默认启用。如需对照旧行为，可在
`activeslam/config/exploration.yaml` 中将
`frontier_include_open_map_edges` 设为 `false`。

当 Nav2 到达开放边缘 frontier 的地图内部安全落点后，探索节点会调用
Nav2 `Spin` 对准局部外法线，再调用 `DriveOnHeading` 低速向边界外前探
`2.0m`。前探速度由 Behavior Server 直接发布到 `/cmd_vel`；它不经过
velocity smoother，但仍使用 Nav2 local costmap 的碰撞检查。探索节点本身
不会直接发布速度命令。
