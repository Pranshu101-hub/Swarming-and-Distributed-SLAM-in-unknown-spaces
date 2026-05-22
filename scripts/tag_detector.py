import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32

import cv2
from pupil_apriltags import Detector


class TagDirectionPublisher(Node):

    def __init__(self):

        super().__init__('tag_direction')

        self.publisher = self.create_publisher(
            Float32,
            '/leader_direction',
            10
        )

        pipeline = (
            "libcamerasrc ! "
            "video/x-raw,width=960,height=720,framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )

        if not self.cap.isOpened():

            self.get_logger().error(
                "Failed to open camera"
            )

            raise RuntimeError("Camera failed")

        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            refine_edges=1
        )

        self.center_x = 960 / 2

        self.filtered_direction = 0.0

        self.timer = self.create_timer(
            0.06,
            self.process_frame
        )

        self.get_logger().info(
            "Tag direction publisher started"
        )

    def process_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        tags = self.detector.detect(gray)

        msg = Float32()

        # =====================================================
        # NO TAG
        # =====================================================

        if len(tags) == 0:

            msg.data = 999.0

        # =====================================================
        # TAG DETECTED
        # =====================================================

        else:

            tag = tags[0]

            error = (
                tag.center[0] -
                self.center_x
            )

            normalized = (
                error / self.center_x
            )

            # =================================================
            # SMOOTHING
            # =================================================

            alpha = 0.15

            self.filtered_direction = (
                alpha * normalized
                +
                (1.0 - alpha) *
                self.filtered_direction
            )

            msg.data = float(
                self.filtered_direction
            )

            self.get_logger().info(
                f"Direction: "
                f"{self.filtered_direction:.2f}"
            )

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = TagDirectionPublisher()

    rclpy.spin(node)

    node.cap.release()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
