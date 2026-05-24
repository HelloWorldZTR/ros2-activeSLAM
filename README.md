# ROS2 Active SLAM

这个项目使用 Docker 启动 ROS 2 Humble 仿真环境，在 Gazebo 中加载 TurtleBot3，并运行 `activeslam` 包里的 SLAM 启动文件。

## 启动 Docker

首次启动（或 Dockerfile/仿真源码有变更时）在项目根目录执行：

```bash
./run_docker_setup.sh
```

这个脚本会准备仿真源码依赖，并使用当前 `Dockerfile` 强制重建 `ros2:latest`。

后续启动直接执行：

```bash
./run_docker.sh
```

这个脚本会删除旧的 `ros2-active-slam` 容器，然后启动新容器。

**注意**：当前脚本使用 `linux/arm64` 平台，如在amd64平台使用，需手动将PLATFORM改为amd64

由于 ROS 2 Humble 在 Jammy 的 `arm64` 源里没有现成的 `gazebo_ros_pkgs` 和 `turtlebot3_gazebo` 二进制包，这个仓库会额外准备两份源码到工作区：

1. `src/gazebo_ros_pkgs`
2. `src/turtlebot3_simulations`

其中 Gazebo 11 本体通过 Open Robotics 的 `gazebo11-non-amd64` PPA 安装。

启动后可在浏览器打开：`http://127.0.0.1:6080`

## 运行 SLAM

在容器内执行：

```bash
cd /home/ubuntu/ros2_ws
source setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py
```

`setup.sh` 会把 `MAKEFLAGS`、`CMAKE_BUILD_PARALLEL_LEVEL` 和 `NINJAFLAGS` 都限制到 `1`，并让 `colcon` 顺序构建，用来降低 Gazebo 相关源码在 ARM 容器中的 OOM 概率。

这个 launch 会：

1. 打开 Gazebo
2. 加载 TurtleBot3
3. 启动 `slam_toolbox`
4. 启动 `exploration_coordinator` 控制节点，使用 frontier-based planning 自动探索

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
