#!/bin/bash

# Use compile flags to limit parallelism, so that the memory is not exhausted
export MAKEFLAGS=-j1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NINJAFLAGS=-j1

# Use gazebo_ros packages from the system ROS install instead of rebuilding the
# vendored src/gazebo_ros_pkgs tree.
GAZEBO_ROS_PKGS_SKIP='gazebo_dev gazebo_msgs gazebo_ros gazebo_plugins gazebo_ros_pkgs'
alias cb='rm -f /home/psirobot/projects/ros2_ws/src/install/activeslam/share/activeslam/launch/slam.launch.py && colcon build --symlink-install --executor sequential --parallel-workers 1 --packages-skip ${GAZEBO_ROS_PKGS_SKIP}'
alias s='source /home/psirobot/projects/ros2_ws/src/install/setup.zsh'
alias r='ros2 launch activeslam slam.launch.py'

export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:/home/psirobot/projects/ros2_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models:/home/psirobot/projects/ros2_ws/src/activeslam_resource/models:/home/psirobot/projects/ros2_ws/install/activeslam_resource/share/activeslam_resource/models:${GAZEBO_MODEL_PATH}
export GAZEBO_MODEL_DATABASE_URI=""
export TURTLEBOT3_MODEL=burger

source /opt/ros/humble/setup.bash
source /home/psirobot/projects/ros2_ws/src/install/setup.bash
