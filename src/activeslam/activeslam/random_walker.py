import random

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class RandomWalker(Node):
    def __init__(self):
        super().__init__('random_walker')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.front_obstacle_distance = float('inf')
        self.current_twist = Twist()
        self.mode_deadline = self.get_clock().now()

        self.obstacle_distance = self.declare_parameter('obstacle_distance', 0.45).value
        self.forward_speed_range = (
            self.declare_parameter('min_linear_speed', 0.05).value,
            self.declare_parameter('max_linear_speed', 0.18).value,
        )
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 1.2).value

        self.choose_random_motion()
        self.motion_timer = self.create_timer(0.2, self.control_loop)

        self.get_logger().info('Random walker started.')

    def scan_callback(self, msg: LaserScan):
        front_ranges = list(msg.ranges[:20]) + list(msg.ranges[-20:])
        valid_ranges = [
            distance
            for distance in front_ranges
            if msg.range_min < distance < msg.range_max
        ]
        self.front_obstacle_distance = min(valid_ranges) if valid_ranges else float('inf')

    def choose_random_motion(self):
        self.current_twist = Twist()
        self.current_twist.linear.x = random.uniform(*self.forward_speed_range)
        self.current_twist.angular.z = random.uniform(
            -self.max_angular_speed / 2.0, self.max_angular_speed / 2.0
        )
        duration = random.uniform(1.5, 4.0)
        self.mode_deadline = self.get_clock().now() + Duration(seconds=duration)

    def choose_turn_motion(self):
        self.current_twist = Twist()
        self.current_twist.angular.z = random.choice([-1.0, 1.0]) * random.uniform(
            0.6, self.max_angular_speed
        )
        duration = random.uniform(1.0, 2.5)
        self.mode_deadline = self.get_clock().now() + Duration(seconds=duration)

    def control_loop(self):
        if (
            self.front_obstacle_distance < self.obstacle_distance
            and self.current_twist.linear.x > 0.0
        ):
            self.choose_turn_motion()
        elif self.get_clock().now() >= self.mode_deadline:
            self.choose_random_motion()

        self.cmd_pub.publish(self.current_twist)

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RandomWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
