#!/bin/bash

set -e

# Change the current directory to the location of this script
cd "$(dirname "$0")"

IMAGE_NAME=ros2:latest
CONTAINER_NAME=ros2-active-slam
PLATFORM=linux/amd64

# Remove any old container so the next run always uses the latest image.
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image not found: ${IMAGE_NAME}"
    echo "Run ./run_docker_setup.sh to build it first."
    exit 1
fi

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
