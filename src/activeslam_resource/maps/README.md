# Active SLAM Maps

Put Gazebo `.world` files in this directory.

Files are installed to:

```text
share/activeslam_resource/maps
```

The `activeslam` launch file selects a map by world basename, for example:

```bash
ros2 launch activeslam slam.launch.py map:=turtlebot3_house
```

TurtleBot-scale SLAM test worlds:

```bash
ros2 launch activeslam slam.launch.py map:=slam_loop
ros2 launch activeslam slam.launch.py map:=slam_rooms
ros2 launch activeslam slam.launch.py map:=slam_rooms_corridor
ros2 launch activeslam slam.launch.py map:=slam_landmarks
```
