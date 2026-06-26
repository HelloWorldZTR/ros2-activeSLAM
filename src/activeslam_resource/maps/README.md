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

The initial reviewable GBSAE prior for the office map is generated from the
same occupancy image. It samples clear interior waypoints, validates
line-of-sight edges, and keeps the component connected to the
`PublicBathroomB` seed:

```bash
python3 tools/generate_office_gbsae_prior.py \
  /path/to/office/map/map.png \
  src/activeslam_resource/maps/slam_office.gbsae.json
ros2 launch activeslam slam.launch.py map:=slam_office slam_mode:=gbsae
```

GBSAE prior graphs use the same world basename with a `.gbsae.json` suffix.
The small inline-box benchmark priors are generated directly from their SDF
box collisions, with conservative per-world free-space envelopes:

```bash
python3 tools/generate_inline_world_gbsae_priors.py \
  --world slam_landmarks \
  --world slam_loop \
  --world slam_rooms \
  --world slam_rooms_corridor
```

Run any generated prior with:

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms slam_mode:=gbsae
```

Starting `gbsae` for a world without a matching JSON asset fails early.
