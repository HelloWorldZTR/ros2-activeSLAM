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
ros2 launch activeslam slam.launch.py map:=slam_office
```

`slam_office.world` is an IoU-friendly adaptation of the office occupancy map
from
[`Dataset-of-Gazebo-Worlds-Models-and-Maps`](https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps).
Its occupied pixels are merged into inline Gazebo box collisions, so the
simulator and `slam_evaluator` use the same obstacle geometry. Regenerate it
after downloading the upstream office archive with:

```bash
python3 tools/generate_slam_office_world.py \
  /path/to/office/map/map.png \
  src/activeslam_resource/maps/slam_office.world
```

GBSAE prior graphs use the same world basename with a `.gbsae.json` suffix.
`slam_rooms.gbsae.json` is the initial hand-authored topo-metric prior:

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gbsae
```

Starting `gbsae` for a world without a matching JSON asset fails early.
