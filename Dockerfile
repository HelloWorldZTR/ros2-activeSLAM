# Start from the Tiryoh Humble image
# This image is multiarch and supports both AMD64 and ARM64 architectures
FROM tiryoh/ros2-desktop-vnc:humble

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO humble

# Switch to Tsinghua University mirror for faster package installation (Optional)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Install additional ROS 2 packages or tools
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-bringup \
    ros-$ROS_DISTRO-slam-toolbox \
    ros-$ROS_DISTRO-turtlebot3 \
    ros-$ROS_DISTRO-turtlebot3-msgs \
    ros-$ROS_DISTRO-turtlebot3-bringup \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /home/ubuntu
