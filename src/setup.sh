#!/bin/bash

# Use compile flags to limit parallelism, so that the memory is not exhausted
export MAKEFLAGS=-j1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NINJAFLAGS=-j1

alias cb='rm -f /home/ubuntu/ros2_ws/install/activeslam/share/activeslam/slam.launch.py && colcon build --symlink-install --executor sequential --parallel-workers 1'
alias s='source /home/ubuntu/ros2_ws/install/setup.bash'
alias r='ros2 launch activeslam slam.launch.py'

export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:/home/ubuntu/ros2_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models:/home/ubuntu/ros2_ws/src/activeslam_resource/models:/home/ubuntu/ros2_ws/install/activeslam_resource/share/activeslam_resource/models:${GAZEBO_MODEL_PATH}
export GAZEBO_MODEL_DATABASE_URI=""
export TURTLEBOT3_MODEL=waffle

source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
