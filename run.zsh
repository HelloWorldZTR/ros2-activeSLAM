#!/usr/bin/env zsh
set -euo pipefail

# Fast headless experiment runner for the remote ROS 2 Humble workspace.
# Usage examples:
#   ./run.zsh
#   MAP=slam_loop STRATEGY=graph RUN_SECONDS=300 ./run.zsh
#   REAL_TIME_UPDATE_RATE=0 MAX_STEP_SIZE=0.001 ./run.zsh

WORKSPACE=${WORKSPACE:-/home/psirobot/projects/ros2_ws}
SRC_DIR=${SRC_DIR:-${WORKSPACE}/src}
MAP=${MAP:-slam_rooms}
PLANNER=${PLANNER:-astar}
STRATEGY=${STRATEGY:-frontier}
RUN_SECONDS=${RUN_SECONDS:-0}
REAL_TIME_UPDATE_RATE=${REAL_TIME_UPDATE_RATE:-0}
MAX_STEP_SIZE=${MAX_STEP_SIZE:-0.001}
LOG_ROOT=${LOG_ROOT:-logs}

cd "${SRC_DIR}"

source_ros_setup() {
  # ROS/colcon setup scripts may read optional unset variables such as
  # COLCON_TRACE, so source them with nounset temporarily disabled.
  set +u
  source "$1"
  set -u
}

source_ros_setup /opt/ros/humble/setup.zsh

export MAKEFLAGS=-j1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NINJAFLAGS=-j1
GAZEBO_ROS_PKGS_SKIP='gazebo_dev gazebo_msgs gazebo_ros gazebo_plugins gazebo_ros_pkgs'
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:${SRC_DIR}/turtlebot3_simulations/turtlebot3_gazebo/models:${SRC_DIR}/activeslam_resource/models:${SRC_DIR}/install/activeslam_resource/share/activeslam_resource/models:${GAZEBO_MODEL_PATH:-}
export GAZEBO_MODEL_DATABASE_URI=""
export TURTLEBOT3_MODEL=burger

mkdir -p "${LOG_ROOT}"
stamp=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_ROOT}/fast_${MAP}_${STRATEGY}_${stamp}.log"

echo "Fast experiment"
echo "  map: ${MAP}"
echo "  planner: ${PLANNER}"
echo "  strategy: ${STRATEGY}"
echo "  gazebo real_time_update_rate: ${REAL_TIME_UPDATE_RATE} (0 means uncapped)"
echo "  gazebo max_step_size: ${MAX_STEP_SIZE}"
echo "  log: ${log_file}"

pkill -f "ros2 launch activeslam slam.launch.py" 2>/dev/null || true
pkill -f "gzserver" 2>/dev/null || true
pkill -f "gzclient" 2>/dev/null || true

rm -f "${SRC_DIR}/install/activeslam/share/activeslam/launch/slam.launch.py" 2>/dev/null || true
colcon build \
  --symlink-install \
  --executor sequential \
  --parallel-workers 1 \
  --packages-skip ${=GAZEBO_ROS_PKGS_SKIP}

source_ros_setup "${SRC_DIR}/install/setup.zsh"

ros2 launch activeslam slam.launch.py \
  map:="${MAP}" \
  planner_type:="${PLANNER}" \
  exploration_strategy:="${STRATEGY}" \
  gui:=false \
  run_evaluator:=true \
  plot_live:=false \
  save_plots:=false \
  log_root:="${LOG_ROOT}" \
  2>&1 | tee "${log_file}" &

launch_pid=$!

cleanup() {
  echo
  echo "Stopping experiment..."
  kill "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

sleep 8

if command -v gz >/dev/null 2>&1; then
  echo "Applying Gazebo fast physics..."
  gz physics -s "${MAX_STEP_SIZE}" -u "${REAL_TIME_UPDATE_RATE}" || \
    echo "Warning: gz physics update failed; continuing with world defaults."
else
  echo "Warning: gz command not found; continuing with world defaults."
fi

if [[ "${RUN_SECONDS}" -gt 0 ]]; then
  sleep "${RUN_SECONDS}"
else
  wait "${launch_pid}"
fi
