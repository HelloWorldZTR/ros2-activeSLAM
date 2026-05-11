#!/bin/bash

# Change the current directory to the location of this script
cd "$(dirname "$0")"

# Build the image
docker build -t ros2 .

echo "Your desktop is running on http://127.0.0.1:6080"
# docker run \
#     -p 6080:80 \
#     --security-opt seccomp=unconfined \
#     --shm-size=512m \
#     -v /Users/andy/Develop/ros2-activeSLAM/src:/home/ubuntu/ros2_ws \
#     ros2
docker run \
    -p 6080:80 \
    --security-opt seccomp=unconfined \
    --shm-size=512m \
    -v ${PWD}/src:/home/ubuntu/ros2_ws \
    ros2