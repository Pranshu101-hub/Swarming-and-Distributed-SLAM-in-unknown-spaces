import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped

import tf2_ros


class GlobalFollowerTF(Node):

    def __init__(self):
        super().__init__('global_follower_tf')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/follower1/cmd_vel',
            10
        )

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.follow_distance = 0.5

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):

        try:
            leader_tf = self.tf_buffer.lookup_transform(
                'map',
                'leader/base_link',
                rclpy.time.Time()
            )

            follower_tf = self.tf_buffer.lookup_transform(
                'map',
                'follower1/base_link',
                rclpy.time.Time()
            )

        except Exception as e:
            self.get_logger().warn(str(e))
            return

        lx = leader_tf.transform.translation.x
        ly = leader_tf.transform.translation.y

        fx = follower_tf.transform.translation.x
        fy = follower_tf.transform.translation.y

        fq = follower_tf.transform.rotation
        follower_yaw = self.get_yaw(fq)

        dx = lx - fx
        dy = ly - fy

        distance = math.sqrt(dx * dx + dy * dy)

        target_angle = math.atan2(dy, dx)

        angle_error = self.normalize_angle(
            target_angle - follower_yaw
        )

        distance_error = distance - self.follow_distance

        cmd = TwistStamped()

        cmd.twist.angular.z = 2.0 * angle_error

        if abs(angle_error) < 0.4:
            cmd.twist.linear.x = 0.6 * distance_error
        else:
            cmd.twist.linear.x = 0.0

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

    node = GlobalFollowerTF()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
