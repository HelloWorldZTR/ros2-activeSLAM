#!/bin/bash

set -e

cd "$(dirname "$0")"

clone_if_missing() {
    local repo_url="$1"
    local branch="$2"
    local target_dir="$3"

    if [ -d "${target_dir}/.git" ]; then
        echo "Using existing source: ${target_dir}"
        return
    fi

    rm -rf "${target_dir}"
    git clone --depth 1 -b "${branch}" "${repo_url}" "${target_dir}"
}

clone_if_missing \
    "https://github.com/ros-simulation/gazebo_ros_pkgs.git" \
    "ros2" \
    "src/gazebo_ros_pkgs"

clone_if_missing \
    "https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git" \
    "humble" \
    "src/turtlebot3_simulations"
