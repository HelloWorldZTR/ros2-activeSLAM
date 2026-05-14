#!/bin/bash

set -e

# Change the current directory to the location of this script
cd "$(dirname "$0")"

IMAGE_NAME=ros2:latest
CONTAINER_NAME=ros2-active-slam
PLATFORM=linux/arm64

bash ./prepare_sim_sources.sh

# Remove any old container so the next run always uses the latest image.
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Always rebuild the latest arm64 image from the current Dockerfile.
docker build --platform "${PLATFORM}" --pull --no-cache -t "${IMAGE_NAME}" .

echo "Your desktop is running on http://127.0.0.1:6080"
echo "Docker platform: ${PLATFORM}"
docker run \
    --memory=16g \
    --memory-swap=32g \
    --name "${CONTAINER_NAME}" \
    --platform "${PLATFORM}" \
    -p 6080:80 \
    --security-opt seccomp=unconfined \
    --shm-size=512m \
    -v ${PWD}/src:/home/ubuntu/ros2_ws \
    "${IMAGE_NAME}"
