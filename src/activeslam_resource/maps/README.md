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

GBSAE prior graphs use the same world basename with a `.gbsae.json` suffix.
`slam_rooms.gbsae.json` is the initial hand-authored topo-metric prior:

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gbsae
```

Starting `gbsae` for a world without a matching JSON asset fails early.
