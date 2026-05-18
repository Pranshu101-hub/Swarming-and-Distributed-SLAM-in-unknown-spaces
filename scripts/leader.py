import math
import random
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

import tf2_ros


class PotentialFieldNavigator(Node):

    def __init__(self):
        super().__init__('potential_field_nav')

        # GOAL
        self.goal_x = 1.0
        self.goal_y = -2.0

        # LIDAR
        self.scan = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/leader/scan',
            self.scan_callback,
            qos
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # CMD VEL
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/leader/cmd_vel',
            10
        )

        # STUCK DETECTION
        self.last_positions = []
        self.escape_until = 0.0
        self.escape_direction = 1.0

        # TIMER
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

    def scan_callback(self, msg):
        self.scan = msg

    def normalize_angle(self, angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def get_yaw(self, q):

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    def compute_potential_field(self):

        if self.scan is None:
            return (0.0, 0.0)

        ranges = self.scan.ranges

        angle = self.scan.angle_min

        fx = 0.0
        fy = 0.0

        for r in ranges:

            if not math.isfinite(r):
                angle += self.scan.angle_increment
                continue

            # Ignore far obstacles
            if r > 1.2:
                angle += self.scan.angle_increment
                continue

            # Stronger repulsion nearby
            strength = 0.08 / (r * r)

            # Repulsive force
            fx += strength * math.cos(angle)
            fy +    = strength * math.sin(angle)

            angle += self.scan.angle_increment

        return (fx, fy)

    def is_stuck(self, x, y):

        self.last_positions.append((x, y))

        if len(self.last_positions) > 25:
            self.last_positions.pop(0)

        if len(self.last_positions) < 25:
            return False

        x0, y0 = self.last_positions[0]

        movement = math.sqrt(
            (x - x0) ** 2 +
            (y - y0) ** 2
        )

        return movement < 0.08

    def control_loop(self):

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                'leader/base_link',
                rclpy.time.Time()
            )

        except Exception as e:
            self.get_logger().warn(str(e))
            return

        # CURRENT POSE
        x = tf.transform.translation.x
        y = tf.transform.translation.y

        q = tf.transform.rotation

        yaw = self.get_yaw(q)

        # GOAL VECTOR
        dx = self.goal_x - x
        dy = self.goal_y - y

        goal_dist = math.sqrt(
            dx * dx +
            dy * dy
        )

        # GOAL REACHED
        cmd = TwistStamped()

        if goal_dist < 0.15:

            self.get_logger().info(
                'GOAL REACHED'
            )

            self.cmd_pub.publish(cmd)
            return

        # STUCK ESCAPE
        current_time = time.time()

        if current_time < self.escape_until:

            cmd.twist.linear.x = 0.02

            cmd.twist.angular.z = (
                1.2 *
                self.escape_direction
            )

            self.cmd_pub.publish(cmd)
            return

        # ATTRACTIVE FORCE
        k_goal = 1.0

        goal_fx = k_goal * dx
        goal_fy = k_goal * dy

        # REPULSIVE FORCE
        obs_fx, obs_fy = self.compute_potential_field()

        # TOTAL FORCE
        total_fx = goal_fx + obs_fx
        total_fy = goal_fy + obs_fy

        target_angle = math.atan2(
            total_fy,
            total_fx
        )

        angle_error = self.normalize_angle(
            target_angle - yaw
        )

        # STUCK DETECTION
        if self.is_stuck(x, y):

            self.escape_until = current_time + 2.0

            self.escape_direction = random.choice([-1.0, 1.0])

            self.get_logger().warn(
                'STUCK -> ESCAPE MODE'
            )

        # MOTION CONTROL
        cmd.twist.angular.z = (
            1.4 *
            angle_error
        )

        # Move faster when aligned
        alignment = max(
            0.0,
            math.cos(angle_error)
        )

        cmd.twist.linear.x = (
            0.12 *
            alignment
        )

        # SPEED LIMITS
        cmd.twist.linear.x = max(
            min(cmd.twist.linear.x, 0.15),
            -0.15
        )

        cmd.twist.angular.z = max(
            min(cmd.twist.angular.z, 1.5),
            -1.5
        )

        self.cmd_pub.publish(cmd)


def main(args=None):

    rclpy.init(args=args)

    node = PotentialFieldNavigator()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
