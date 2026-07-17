import math
import random
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TwistStamped, Point
from sensor_msgs.msg import LaserScan

import tf2_ros

FREE = 0
OCCUPIED = 100
UNKNOWN = -1
MIN_FRONTIER_SIZE = 3
GOAL_REACHED_DIST = 0.30
STUCK_WINDOW = 40           # more history before declaring stuck
STUCK_THRESHOLD = 0.05      # tighter threshold
ESCAPE_DURATION = 1.5       # shorter escape
LINEAR_MAX = 0.10           # slower, safer
ANGULAR_MAX = 1.2
VISITED_RADIUS = 0.5
OBSTACLE_RANGE = 0.8        # only react to obstacles within 0.8m (was 1.2)
REPULSION_STRENGTH = 0.05   # weaker repulsion (was 0.08)
FRONT_SAFETY_DIST = 0.40    # stop if obstacle within 40cm directly ahead
FRONT_CONE_FRAC = 6         # 1/6 of scan = ~60 degree front cone


class RobotState:

    def __init__(self, name, scan_topic, cmd_topic, base_frame):
        self.name = name
        self.scan_topic = scan_topic
        self.cmd_topic = cmd_topic
        self.base_frame = base_frame
        self.scan = None
        self.goal_x = None
        self.goal_y = None
        self.last_positions = []
        self.escape_until = 0.0
        self.escape_dir = 1.0


class SwarmCoordinator(Node):

    def __init__(self):
        super().__init__('swarm_coordinator')

        self.robots = [
            RobotState(
                name='robot1',
                scan_topic='/robot1/scan',
                cmd_topic='/robot1/cmd_vel',
                base_frame='robot1/base_link',
            ),
            RobotState(
                name='robot2',
                scan_topic='/robot2/scan',
                cmd_topic='/robot2/cmd_vel',
                base_frame='robot2/base_link',
            ),
        ]

        self.visited_frontiers = set()
        self.manual_goal_active = False

        self.map_data = None
        self.map_info = None

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos,
        )

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.scan_subs = []
        self.cmd_pubs = []

        for i, robot in enumerate(self.robots):
            sub = self.create_subscription(
                LaserScan,
                robot.scan_topic,
                lambda msg, idx=i: self.scan_callback(msg, idx),
                scan_qos,
            )
            self.scan_subs.append(sub)

            pub = self.create_publisher(
                TwistStamped,
                robot.cmd_topic,
                10,
            )
            self.cmd_pubs.append(pub)

        self.manual_goal_sub = self.create_subscription(
            Point,
            '/manual_goal',
            self.manual_goal_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.frontier_timer = self.create_timer(5.0, self.assign_frontiers)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('SwarmCoordinator started.')

    def manual_goal_callback(self, msg):
        # z=999 is sentinel from dashboard "resume autonomous" button
        if msg.z == 999.0:
            self.manual_goal_active = False
            self.robots[0].goal_x = None
            self.robots[0].goal_y = None
            self.get_logger().info('manual goal cleared, resuming autonomous exploration')
            return
        self.robots[0].goal_x = msg.x
        self.robots[0].goal_y = msg.y
        self.manual_goal_active = True
        self.get_logger().info(
            f'manual goal override: robot1 -> ({msg.x:.2f}, {msg.y:.2f})'
        )

    def map_callback(self, msg):
        self.map_data = list(msg.data)
        self.map_info = msg.info

    def scan_callback(self, msg, idx):
        self.robots[idx].scan = msg

    def detect_frontiers(self):
        if self.map_data is None or self.map_info is None:
            return []

        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        data = self.map_data

        def idx(r, c):
            return r * width + c

        def neighbors4(r, c):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    yield nr, nc

        frontier_cells = set()
        for r in range(height):
            for c in range(width):
                if data[idx(r, c)] != FREE:
                    continue
                for nr, nc in neighbors4(r, c):
                    if data[idx(nr, nc)] == UNKNOWN:
                        frontier_cells.add((r, c))
                        break

        if not frontier_cells:
            return []

        visited = set()
        clusters = []

        for start in frontier_cells:
            if start in visited:
                continue
            cluster = []
            queue = deque([start])
            visited.add(start)
            while queue:
                cell = queue.popleft()
                cluster.append(cell)
                r, c = cell
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in visited and (nr, nc) in frontier_cells:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            if len(cluster) >= MIN_FRONTIER_SIZE:
                clusters.append(cluster)

        centroids = []
        for cluster in clusters:
            avg_r = sum(r for r, c in cluster) / len(cluster)
            avg_c = sum(c for r, c in cluster) / len(cluster)
            wx = origin_x + (avg_c + 0.5) * resolution
            wy = origin_y + (avg_r + 0.5) * resolution
            centroids.append((wx, wy))

        return centroids

    def assign_frontiers(self):
        if self.manual_goal_active:
            self.get_logger().info('manual goal active, skipping frontier assignment for robot1')
            return

        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info('No frontiers detected.')
            return

        poses = []
        for robot in self.robots:
            pos = self.get_robot_pose(robot)
            poses.append(pos)

        assigned = []
        for i, robot in enumerate(self.robots):
            if poses[i] is None:
                continue
            rx, ry, _ = poses[i]
            best_dist = float('inf')
            best_frontier = None
            for fx, fy in frontiers:
                if (fx, fy) in assigned:
                    continue
                too_close = any(
                    math.sqrt((fx - vx) ** 2 + (fy - vy) ** 2) < VISITED_RADIUS
                    for vx, vy in self.visited_frontiers
                )
                if too_close:
                    continue
                dist = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_frontier = (fx, fy)
            if best_frontier is not None:
                robot.goal_x, robot.goal_y = best_frontier
                assigned.append(best_frontier)
                self.get_logger().info(
                    f'{robot.name} assigned frontier '
                    f'({robot.goal_x:.2f}, {robot.goal_y:.2f})'
                )
            else:
                self.get_logger().info(f'{robot.name} no new frontiers available.')

    def get_robot_pose(self, robot):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                robot.base_frame,
                rclpy.time.Time(),
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            yaw = self.get_yaw(tf.transform.rotation)
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f'{robot.name} TF: {e}')
            return None

    def get_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def is_front_blocked(self, robot):
        if robot.scan is None:
            return False
        ranges = robot.scan.ranges
        n = len(ranges)
        half = n // 2
        cone = n // FRONT_CONE_FRAC
        for fi in range(half - cone, half + cone):
            if 0 <= fi < n and math.isfinite(ranges[fi]) and ranges[fi] < FRONT_SAFETY_DIST:
                return True
        return False

    def compute_repulsion(self, robot):
        if robot.scan is None:
            return 0.0, 0.0
        fx, fy = 0.0, 0.0
        angle = robot.scan.angle_min
        for r in robot.scan.ranges:
            if math.isfinite(r) and r < OBSTACLE_RANGE:
                strength = REPULSION_STRENGTH / (r * r)
                fx += strength * math.cos(angle)
                fy += strength * math.sin(angle)
            angle += robot.scan.angle_increment
        return fx, fy

    def is_stuck(self, robot, x, y):
        robot.last_positions.append((x, y))
        if len(robot.last_positions) > STUCK_WINDOW:
            robot.last_positions.pop(0)
        if len(robot.last_positions) < STUCK_WINDOW:
            return False
        x0, y0 = robot.last_positions[0]
        return math.sqrt((x - x0) ** 2 + (y - y0) ** 2) < STUCK_THRESHOLD

    def control_loop(self):
        for i, robot in enumerate(self.robots):
            cmd = TwistStamped()

            if robot.goal_x is None:
                self.cmd_pubs[i].publish(cmd)
                continue

            pose = self.get_robot_pose(robot)
            if pose is None:
                continue

            x, y, yaw = pose
            now = time.time()

            dx = robot.goal_x - x
            dy = robot.goal_y - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < GOAL_REACHED_DIST:
                self.get_logger().info(f'{robot.name} reached goal.')
                self.visited_frontiers.add((robot.goal_x, robot.goal_y))
                robot.goal_x = None
                robot.goal_y = None
                if i == 0 and self.manual_goal_active:
                    self.manual_goal_active = False
                    self.get_logger().info('manual goal reached, resuming frontier exploration')
                self.cmd_pubs[i].publish(cmd)
                continue

            # escape mode
            if now < robot.escape_until:
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 1.0 * robot.escape_dir
                self.cmd_pubs[i].publish(cmd)
                continue

            # front safety — stop and turn if obstacle directly ahead
            if self.is_front_blocked(robot):
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 0.6 * (1.0 if robot.escape_dir >= 0 else -1.0)
                self.cmd_pubs[i].publish(cmd)
                continue

            # potential field
            goal_fx = 1.0 * dx
            goal_fy = 1.0 * dy
            obs_fx, obs_fy = self.compute_repulsion(robot)
            total_fx = goal_fx + obs_fx
            total_fy = goal_fy + obs_fy

            target_angle = math.atan2(total_fy, total_fx)
            angle_error = self.normalize_angle(target_angle - yaw)

            # stuck detection
            if self.is_stuck(robot, x, y):
                robot.escape_until = now + ESCAPE_DURATION
                robot.escape_dir = random.choice([-1.0, 1.0])
                self.get_logger().warn(f'{robot.name} STUCK -> escape')

            alignment = max(0.0, math.cos(angle_error))
            cmd.twist.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, 1.2 * angle_error))
            cmd.twist.linear.x = max(-LINEAR_MAX, min(LINEAR_MAX, 0.10 * alignment))

            self.cmd_pubs[i].publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SwarmCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
