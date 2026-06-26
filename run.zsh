#!/usr/bin/env zsh
set -euo pipefail

# Fast headless experiment runner for the remote ROS 2 Humble workspace.
# Usage examples:
#   ./run.zsh
#   MAP=slam_loop SLAM_MODE=approx_graph ./run.zsh
#   MAP=slam_rooms SLAM_MODE=gbsae RUN_SECONDS=120 ./run.zsh
#   MAPS="slam_landmarks slam_loop" RUN_SECONDS=600 ./run.zsh
#   REAL_TIME_UPDATE_RATE=0 MAX_STEP_SIZE=0.001 ./run.zsh
# Default physics is capped near 5x realtime; set REAL_TIME_UPDATE_RATE=0
# only for quick smoke tests where controller stability is less important.

WORKSPACE=${WORKSPACE:-/home/psirobot/projects/ros2_ws}
SRC_DIR=${SRC_DIR:-${WORKSPACE}/src}
DEFAULT_MAPS=(slam_landmarks slam_loop slam_rooms slam_rooms_corridor)
if [[ -z "${SLAM_MODE:-}" && -n "${STRATEGY:-}" ]]; then
  echo "Warning: STRATEGY is deprecated; use SLAM_MODE."
fi
SLAM_MODE=${SLAM_MODE:-${STRATEGY:-frontier}}
RUN_SECONDS=${RUN_SECONDS:-300}
REAL_TIME_UPDATE_RATE=${REAL_TIME_UPDATE_RATE:-5000}
MAX_STEP_SIZE=${MAX_STEP_SIZE:-0.001}
LOG_ROOT=${LOG_ROOT:-logs}
RUN_MAPS=()
launch_pid=""

cd "${SRC_DIR}"

parse_map_list() {
  if [[ -n "${MAPS:-}" ]]; then
    local raw_maps="${MAPS//,/ }"
    RUN_MAPS=(${=raw_maps})
  elif [[ -n "${MAP:-}" ]]; then
    RUN_MAPS=("${MAP}")
  else
    RUN_MAPS=("${DEFAULT_MAPS[@]}")
  fi

  if (( ${#RUN_MAPS[@]} == 0 )); then
    echo "No maps specified. Set MAP=slam_rooms or MAPS=\"slam_rooms slam_loop\"."
    exit 1
  fi
}

deactivate_conda_if_needed() {
  if [[ -z "${CONDA_PREFIX:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    return
  fi

  echo "Conda environment detected; deactivating before ROS setup."

  while [[ "${CONDA_SHLVL:-0}" -gt 0 ]] && command -v conda >/dev/null 2>&1; do
    conda deactivate || break
  done

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    local conda_prefix="${CONDA_PREFIX}"
    local -a clean_path
    local entry
    for entry in "${path[@]}"; do
      if [[ "${entry}" != "${conda_prefix}/bin" && "${entry}" != "${conda_prefix}/condabin" ]]; then
        clean_path+=("${entry}")
      fi
    done
    path=("${clean_path[@]}")
  fi

  unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER
  unset CONDA_PYTHON_EXE CONDA_SHLVL
}

source_ros_setup() {
  # ROS/colcon setup scripts may read optional unset variables such as
  # COLCON_TRACE, so source them with nounset temporarily disabled.
  set +u
  source "$1"
  set -u
}

stop_existing_sim() {
  pkill -f "[r]os2 launch activeslam slam.launch.py" 2>/dev/null || true
  pkill -f "[g]zserver" 2>/dev/null || true
  pkill -f "[g]zclient" 2>/dev/null || true
}

cleanup() {
  if [[ -z "${launch_pid:-}" ]]; then
    return
  fi

  echo
  echo "Stopping experiment..."
  kill "${launch_pid}" 2>/dev/null || true
  pkill -f "[r]os2 launch activeslam slam.launch.py" 2>/dev/null || true
  sleep 2
  pkill -f "[e]xploration_coordinator" 2>/dev/null || true
  pkill -f "[s]lam_evaluator" 2>/dev/null || true
  pkill -f "[s]ync_slam_toolbox_node" 2>/dev/null || true
  pkill -f "[s]pawn_entity.py" 2>/dev/null || true
  pkill -f "[g]zserver" 2>/dev/null || true
  pkill -f "[g]zclient" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "[g]zserver" 2>/dev/null || true
  pkill -KILL -f "[g]zclient" 2>/dev/null || true

  wait "${launch_pid}" 2>/dev/null || true
  launch_pid=""
}
trap cleanup INT TERM EXIT

run_experiment() {
  local map_name="$1"
  local stamp=$(date +%Y%m%d_%H%M%S)
  local run_root="${LOG_ROOT}/run_${map_name}_${SLAM_MODE}_${stamp}"
  local log_file="${run_root}/launch.log"

  mkdir -p "${run_root}"

  echo "Fast experiment"
  echo "  map: ${map_name}"
  echo "  slam mode: ${SLAM_MODE}"
  echo "  run seconds: ${RUN_SECONDS}"
  echo "  gazebo real_time_update_rate: ${REAL_TIME_UPDATE_RATE} (0 means uncapped)"
  echo "  gazebo max_step_size: ${MAX_STEP_SIZE}"
  echo "  output: ${run_root}"
  echo "  log: ${log_file}"

  stop_existing_sim

  ros2 launch activeslam slam.launch.py \
    map:="${map_name}" \
    slam_mode:="${SLAM_MODE}" \
    gui:=false \
    run_rviz:=false \
    run_evaluator:=true \
    plot_live:=false \
    save_plots:=false \
    log_root:="${run_root}" \
    > >(tee "${log_file}") 2>&1 &

  launch_pid=$!

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

  cleanup
}

parse_map_list
deactivate_conda_if_needed
source_ros_setup /opt/ros/humble/setup.zsh

export MAKEFLAGS=-j1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NINJAFLAGS=-j1
GAZEBO_ROS_PKGS_SKIP='gazebo_dev gazebo_msgs gazebo_ros gazebo_plugins gazebo_ros_pkgs'
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:${SRC_DIR}/turtlebot3_simulations/turtlebot3_gazebo/models:${SRC_DIR}/activeslam_resource/models:${SRC_DIR}/install/activeslam_resource/share/activeslam_resource/models:${GAZEBO_MODEL_PATH:-}
export GAZEBO_MODEL_DATABASE_URI=""
export TURTLEBOT3_MODEL=burger

mkdir -p "${LOG_ROOT}"
echo "Maps: ${RUN_MAPS[*]}"

rm -f "${SRC_DIR}/install/activeslam/share/activeslam/launch/slam.launch.py" 2>/dev/null || true
colcon build \
  --symlink-install \
  --executor sequential \
  --parallel-workers 1 \
  --packages-skip ${=GAZEBO_ROS_PKGS_SKIP}

source_ros_setup "${SRC_DIR}/install/setup.zsh"

for map_name in "${RUN_MAPS[@]}"; do
  run_experiment "${map_name}"
done
