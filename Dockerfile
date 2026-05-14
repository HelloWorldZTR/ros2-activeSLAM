# Start from the Tiryoh Humble image
# This image is multiarch and supports both AMD64 and ARM64 architectures
FROM tiryoh/ros2-desktop-vnc:humble

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO humble
ENV DEBIAN_FRONTEND noninteractive

# Switch to Tsinghua University mirror for faster package installation (Optional)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Gazebo Classic is not published as an official ROS 2 arm64 binary on Jammy,
# so we use the Open Robotics PPA for the Gazebo 11 binaries.
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:openrobotics/gazebo11-non-amd64 \
    && apt-get update

# Install additional ROS 2 packages or tools.
RUN apt-get update && apt-get install -y \
    gazebo \
    libgazebo-dev \
    ros-$ROS_DISTRO-camera-info-manager \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-bringup \
    ros-$ROS_DISTRO-slam-toolbox \
    ros-$ROS_DISTRO-gazebo-dev \
    ros-$ROS_DISTRO-gazebo-msgs \
    ros-$ROS_DISTRO-turtlebot3 \
    ros-$ROS_DISTRO-turtlebot3-msgs \
    ros-$ROS_DISTRO-turtlebot3-bringup \
    ros-$ROS_DISTRO-robot-state-publisher \
    ros-$ROS_DISTRO-joint-state-publisher \
    ros-$ROS_DISTRO-urdf \
    ros-$ROS_DISTRO-xacro \
    python3-colcon-common-extensions \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /home/ubuntu
