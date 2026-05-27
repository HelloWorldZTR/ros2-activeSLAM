# ROS2 Active SLAM

## 运行 SLAM

在容器内执行：

```bash
cd /home/ubuntu/ros2_ws
source setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py map:=slam_rooms
```

如果需要同时记录评估指标：

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms enable_evaluator:=true
```

在bc01执行

```bash
cd /home/psirobot/projects/ros2_ws/src
source setup.sh
cb
s
# in terminal 2
ros2 launch activeslam slam.launch.py map:=slam_rooms
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

## 可选参数

如果需要指定 TurtleBot3 型号，例如 `waffle`：

```bash
ros2 launch activeslam slam.launch.py turtlebot3_model:=waffle
```

如果需要选择 Gazebo 地图，例如 TurtleBot3 house：
不许加入world后缀

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms x_pose:=-2.0 y_pose:=-0.5
```
