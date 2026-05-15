#!/bin/bash

set -e

# Change the current directory to the location of this script
cd "$(dirname "$0")"

IMAGE_NAME=ros2:latest
BASE_IMAGE=tiryoh/ros2-desktop-vnc:humble
PLATFORM=linux/arm64

bash ./prepare_sim_sources.sh

# Always rebuild the latest image from the current Dockerfile.
BUILD_PULL_ARGS=()
if docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Base image found locally, skipping pull: ${BASE_IMAGE}"
else
    echo "Base image not found locally, will pull using Docker default config: ${BASE_IMAGE}"
    BUILD_PULL_ARGS+=(--pull)
fi

docker build --platform "${PLATFORM}" --no-cache "${BUILD_PULL_ARGS[@]}" -t "${IMAGE_NAME}" .

echo "Setup complete. You can now run ./run_docker.sh"
echo "Docker platform: ${PLATFORM}"
echo "Rerun this setup if Dockerfile or simulation sources change."
