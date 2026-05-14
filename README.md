# ROS2 Active SLAM

这个项目使用 Docker 启动 ROS 2 Humble 仿真环境，在 Gazebo 中加载 TurtleBot3，并运行 `activeslam` 包里的 SLAM 启动文件。

## 启动 Docker

在项目根目录执行：

```bash
./run_docker.sh
```

这个脚本会先准备仿真源码依赖，再删除旧的 `ros2-active-slam` 容器，使用当前 `Dockerfile` 强制重建 `ros2:latest`，最后启动新容器。

当前脚本固定使用 `linux/arm64` 平台，也就是原生 ARM 镜像。

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
4. 启动 `random_walker` 控制节点，让机器人随机游走

## 可选参数

如果需要指定 TurtleBot3 型号，例如 `waffle`：

```bash
ros2 launch activeslam slam.launch.py turtlebot3_model:=waffle
```
