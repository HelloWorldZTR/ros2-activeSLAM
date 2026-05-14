# ROS2 Active SLAM

这个项目使用 Docker 启动 ROS 2 Humble 仿真环境，在 Gazebo 中加载 TurtleBot3，并运行 `activeslam` 包里的 SLAM 启动文件。

## 启动 Docker

在项目根目录执行：

```bash
./run_docker.sh
```

脚本会完成两件事：

1. 构建 Docker 镜像 `ros2`
2. 启动容器，并将本地 `src/` 挂载到容器内 `/home/ubuntu/ros2_ws`

启动后可在浏览器打开：

```text
http://127.0.0.1:6080
```

## 容器内编译

进入容器终端后，执行：

```bash
cd /home/ubuntu/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果你使用仓库自带脚本，也可以执行：

```bash
source /home/ubuntu/ros2_ws/setup.sh
cb
source /home/ubuntu/ros2_ws/install/setup.bash
```

## 运行 SLAM

在容器内执行：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch activeslam slam.launch.py
```

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
