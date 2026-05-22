import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


class HybridFollower(Node):

    def __init__(self):

        super().__init__('hybrid_follower')

        # =====================================================
        # QoS FOR LIDAR
        # =====================================================

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # =====================================================
        # SUBSCRIBERS
        # =====================================================

        self.create_subscription(
            Float32,
            '/leader_direction',
            self.direction_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )

        # =====================================================
        # PUBLISHER
        # =====================================================

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # =====================================================
        # STATE
        # =====================================================

        self.scan = None

        self.leader_visible = False

        self.last_seen_time = time.time()

        self.target_distance = 0.60

        # =====================================================
        # LOOP
        # =====================================================

        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.get_logger().info(
            "Hybrid follower started"
        )

    # =========================================================
    # CALLBACKS
    # =========================================================

    def direction_callback(self, msg):

        if msg.data != 999.0:

            self.leader_visible = True
            self.last_seen_time = time.time()

        else:

            self.leader_visible = False

    def scan_callback(self, msg):

        self.scan = msg

    # =========================================================
    # CONTROL LOOP
    # =========================================================

    def control_loop(self):

        if self.scan is None:
            return

        cmd = TwistStamped()

        cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )

        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0

        current_time = time.time()

        # =====================================================
        # LEADER VISIBLE
        # =====================================================

        if self.leader_visible:

            ranges = self.scan.ranges

            valid_ranges = []
            valid_indices = []

            search_width = 25

            # =================================================
            # FRONT LEFT SIDE
            # =================================================

            for i in range(0, search_width):

                r = ranges[i]

                if math.isinf(r):
                    continue

                if math.isnan(r):
                    continue

                if r < 0.08:
                    continue

                if r > 2.5:
                    continue

                valid_ranges.append(r)
                valid_indices.append(i)

            # =================================================
            # FRONT RIGHT SIDE
            # =================================================

            for i in range(
                len(ranges) - search_width,
                len(ranges)
            ):

                r = ranges[i]

                if math.isinf(r):
                    continue

                if math.isnan(r):
                    continue

                if r < 0.08:
                    continue

                if r > 2.5:
                    continue

                valid_ranges.append(r)
                valid_indices.append(i)

            # =================================================
            # NO FRONT TARGET
            # =================================================

            if len(valid_ranges) == 0:

                self.get_logger().info(
                    "NO FRONT TARGET"
                )

            # =================================================
            # TARGET FOUND
            # =================================================

            else:

                best_distance = min(valid_ranges)

                best_index = valid_indices[
                    valid_ranges.index(best_distance)
                ]

                # =============================================
                # CONVERT INDEX TO ANGLE
                # =============================================

                actual_angle = (
                    self.scan.angle_min +
                    best_index *
                    self.scan.angle_increment
                )

                # Convert to [-pi, pi]
                if actual_angle > math.pi:

                    actual_angle -= (
                        2 * math.pi
                    )

                angle_error = math.degrees(
                    actual_angle
                )

                # =============================================
                # ANGULAR CONTROL
                # =============================================

                cmd.twist.angular.z = (
                    0.012 * angle_error
                )

                # Deadzone
                if abs(angle_error) < 5:

                    cmd.twist.angular.z = 0.0

                # Clamp
                cmd.twist.angular.z = max(
                    -0.4,
                    min(
                        0.4,
                        cmd.twist.angular.z
                    )
                )

                # =============================================
                # DISTANCE CONTROL
                # =============================================

                distance_error = (
                    best_distance -
                    self.target_distance
                )

                cmd.twist.linear.x = (
                    0.55 * distance_error
                )

                cmd.twist.linear.x = max(
                    -0.05,
                    min(
                        0.20,
                        cmd.twist.linear.x
                    )
                )

                # =============================================
                # SLOW DOWN WHILE TURNING
                # =============================================

                turn_strength = abs(
                    cmd.twist.angular.z
                )

                cmd.twist.linear.x *= (
                    max(
                        0.3,
                        1.0 - 1.5 * turn_strength
                    )
                )

                # =============================================
                # DEBUG
                # =============================================

                self.get_logger().info(
                    f"TRACKING | "
                    f"dist={best_distance:.2f} "
                    f"angle={angle_error:.1f}"
                )

        # =====================================================
        # CAMERA LOST
        # =====================================================

        else:

            lost_duration = (
                current_time -
                self.last_seen_time
            )

            # =================================================
            # SHORT MEMORY
            # =================================================

            if lost_duration < 1.0:

                cmd.twist.linear.x = 0.03

                self.get_logger().info(
                    "MEMORY TRACK"
                )

            # =================================================
            # SEARCH
            # =================================================

            else:

                cmd.twist.angular.z = 0.16

                self.get_logger().info(
                    "SEARCHING"
                )

        self.cmd_pub.publish(cmd)


def main(args=None):

    rclpy.init(args=args)

    node = HybridFollower()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
