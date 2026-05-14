#!/bin/bash

# Use compile flags to limit parallelism, so that the memory is not exhausted
export MAKEFLAGS=-j1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NINJAFLAGS=-j1

alias cb='colcon build --symlink-install --executor sequential --parallel-workers 1'

export GAZEBO_MODEL_PATH=/home/ubuntu/ros2_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH}
export GAZEBO_MODEL_DATABASE_URI=""
export TURTLEBOT3_MODEL=waffle

source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
