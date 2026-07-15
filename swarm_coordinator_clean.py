import math # added this library for all geometric calcs
import random # randomly pick an escape direction,either 1 or -1 so that it doesnt always spin the same way when stuck (1 for forward -1 for backward/ 1 for left -1 for right)
import time #used to get current timestamp (time.time()) to compare against escape_until. tells us whether escape mode is still active.
from collections import deque # used in bfs frontier clustering. bettr and faster than a list for appending and popping from both directions. the queue here just uses FIFO algo

import rclpy # the ROS2 Python library. lets you create nodes, spin them, and use ROS2 time.
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
#quality of Service settings for ROS2 topics. controls how messages are delivered. RELIABLE means every message is guaranteed to arrive. BEST_EFFORT means drop messages if network is busy. TRANSIENT_LOCAL means new subscribers get the last published message even if they subscribed late  important for the map topic.

from nav_msgs.msg import OccupancyGrid #the map message type. contains a flat array of cell values (-1, 0, or 100) plus metadata (width, height, resolution, origin).
from geometry_msgs.msg import TwistStamped # cntains linear velocity,and angular velocity plus a timestamp
from sensor_msgs.msg import LaserScan #lidar message contains an array of distance readings at different angles

import tf2_ros #the ROS2 transform library. lets you ask "where is robot1/base_link relative to the map frame right now?

FREE = 0 # sort of an occupancy cell value,0 indicating the cell is empty and the bot can drive here
OCCUPIED = 100 # occupancy grid value meaning that theres an obstacle here and the robot cannot drive here . interpret these as flags.
UNKNOWN = -1 # occupancy grid value meaning the robot has never seen this cell. this is what we want to explore.
#FRONTIER_CLUSTER_RADIUS = 0.5
MIN_FRONTIER_SIZE = 3 # minimum number of cells a frontier cluster must have to be considered valid. filters out single-cell noise artifacts in the map.
GOAL_REACHED_DIST = 0.30 # if the robot is within 0.3 meters of its goal, it's considered reached. then the goal is cleared and a new frontier gets assigned.

STUCK_WINDOW = 25 #how many past positions to track for stuck detection. at 10Hz this is 2.5 seconds of history.
STUCK_THRESHOLD = 0.08 # if total displacement over the last 25 ticks is less than 8cm, robot is stuck. then initiate escape mode
ESCAPE_DURATION = 2.0 # robot spins in place for 2 seconds to escape when stuck
LINEAR_MAX = 0.15 # max forward speed in ms 
ANGULAR_MAX = 1.5  # mac turning speed in rad/s


class RobotState:

    def __init__(self, name, scan_topic, cmd_topic, base_frame): # this is just a data container one instanc eper robot. variable names are pretty self explanatory
        self.name = name #string, "robot1" or "robot2". used in log messages so you know which robot is doing what.
        self.scan_topic = scan_topic # the ROS topic this robot publishes LiDAR data to. /robot1/scan or /robot2/scan.
        self.cmd_topic = cmd_topic#he ROS topic this robot listens to for velocity commands. /robot1/cmd_vel or /robot2/cmd_vel
        self.base_frame = base_frame # the TF frame for this robot's base. robot1/base_link or robot2/base_link. used to look up the robot's position in the map.
        self.scan = None #stores the most recent LaserScan message received from this robot. starts as None. updated every time a new scan arrives.
        self.goal_x = None #the world coordinates (in meters) of the frontier this robot is currently heading to. None means no goal assigned. set by assign_frontiers, cleared when robot reaches the goal.
        self.goal_y = None #the world coordinates (in meters) of the frontier this robot is currently heading to. None means no goal assigned. set by assign_frontiers, cleared when robot reaches the goal.
        self.last_positions = []#list of (x, y) tuples. stores the robot's recent positions. used by is_stuck to check if the robot has barely moved.
        self.escape_until = 0.0 # a float timestamp. if time.time() < escape_until, the robot is in escape mode and spins instead of navigating. set to now + 2.0 when stuck is detected.
        self.escape_dir = 1.0 #either +1.0 or -1.0. determines which direction to spin during escape. randomized so the robot doesn't always get stuck spinning into the same wall.


class SwarmCoordinator(Node):
 
    def __init__(self):
        super().__init__('swarm_coordinator')
 
        # create one RobotState per robot, topics and frames must match what bringup published
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
 
        self.map_data = None    # flat list of cell values from the merged map, length = width x height
        self.map_info = None    # metadata: width, height, resolution, origin position
 
        # map needs RELIABLE + TRANSIENT_LOCAL so we get every update and get the latest map on startup
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
 
        # subscribe to the merged map, calls map_callback every time the map updates
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos,
        )
 
        # lidar is published fast (10Hz), ok to drop messages if busy
        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT
 
        self.scan_subs = []     # list of lidar subscribers, one per robot
        self.cmd_pubs = []      # list of velocity publishers, one per robot
 
        # create a lidar subscriber and velocity publisher for each robot
        for i, robot in enumerate(self.robots):
 
            # the lambda captures idx=i so each subscription knows which robot it belongs to
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
 
        # tf lets us ask "where is robot1 in the map right now"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
 
        # every 5 seconds, look at the map and assign each robot a new frontier to explore
        self.frontier_timer = self.create_timer(5.0, self.assign_frontiers)
 
        # every 0.1 seconds (10Hz), compute velocity commands and send them to each robot
        self.control_timer = self.create_timer(0.1, self.control_loop)
 
        self.get_logger().info('SwarmCoordinator started.')
 
    def map_callback(self, msg):
        # called every time the merged map updates
        # msg.data is a tuple so we convert to list for easier indexing
        self.map_data = list(msg.data)
        self.map_info = msg.info
 
    def scan_callback(self, msg, idx):
        # called every time a robot's lidar publishes
        # idx is 0 for robot1, 1 for robot2
        self.robots[idx].scan = msg
 
    def detect_frontiers(self):
        # finds all the edges between explored and unexplored space in the merged map
        # returns a list of (x, y) world coordinates, one per frontier cluster
 
        if self.map_data is None or self.map_info is None:
            return []
 
        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution      # meters per cell, e.g. 0.05 means 5cm per cell
        origin_x = self.map_info.origin.position.x   # world x of the bottom-left corner of the map
        origin_y = self.map_info.origin.position.y   # world y of the bottom-left corner of the map
        data = self.map_data
 
        # converts row, column to flat array index
        # the map is stored as one long list, not a 2D grid
        def idx(r, c):
            return r * width + c
 
        # returns the 4 cells directly above, below, left, right of a given cell
        # checks bounds so we dont go off the edge of the map
        def neighbors4(r, c):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    yield nr, nc
 
        # step 1: find every cell that is FREE and has at least one UNKNOWN neighbor
        # these are the frontier cells, the edge of what we know
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
 
        # step 2: group connected frontier cells into clusters using BFS
        # avoids assigning two robots to goals that are 5cm apart
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
            # only keep clusters big enough to be real frontiers
            if len(cluster) >= MIN_FRONTIER_SIZE:
                clusters.append(cluster)
 
        # step 3: compute the centroid of each cluster and convert to world coordinates
        centroids = []
        for cluster in clusters:
            avg_r = sum(r for r, c in cluster) / len(cluster)
            avg_c = sum(c for r, c in cluster) / len(cluster)
            # + 0.5 centers the point inside the cell rather than at its corner
            wx = origin_x + (avg_c + 0.5) * resolution
            wy = origin_y + (avg_r + 0.5) * resolution
            centroids.append((wx, wy))
 
        return centroids
 
    def assign_frontiers(self):
        # called every 5 seconds
        # finds all frontiers and gives each robot a different one to go to
 
        frontiers = self.detect_frontiers()
 
        if not frontiers:
            self.get_logger().info('No frontiers detected.')
            return
 
        # get current pose of each robot
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
 
            # find the closest frontier that hasnt already been given to another robot
            for fx, fy in frontiers:
                if (fx, fy) in assigned:
                    continue
                # Euclidean distance from robot to frontier
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
 
    def get_robot_pose(self, robot):
        # asks TF where this robot currently is in the map frame
        # returns (x, y, yaw) or None if the transform isnt available yet
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
        # converts quaternion rotation to a single yaw angle in radians
        # quaternions are how ROS stores 3D rotation but for a flat floor robot we only care about yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
 
    def normalize_angle(self, a):
        # keeps an angle in the range -pi to pi
        # needed because subtracting two angles can produce values outside that range
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a
 
    def compute_repulsion(self, robot):
        # reads the robots lidar scan and computes a force pushing it away from nearby obstacles
        # closer obstacles push harder: strength = 0.08 / distance^2
        # returns total repulsion as (fx, fy) force vector
        if robot.scan is None:
            return 0.0, 0.0
        fx, fy = 0.0, 0.0
        angle = robot.scan.angle_min   # start at the first lidar beam angle
        for r in robot.scan.ranges:
            if math.isfinite(r) and r < 1.2:   # only care about obstacles closer than 1.2 meters
                strength = 0.08 / (r * r)       # inverse square, much stronger up close
                fx += strength * math.cos(angle)
                fy += strength * math.sin(angle)
            angle += robot.scan.angle_increment  # move to next beam angle
        return fx, fy
 
    def is_stuck(self, robot, x, y):
        # checks if the robot has barely moved over the last STUCK_WINDOW ticks
        # keeps a rolling list of recent positions and compares oldest to newest
        robot.last_positions.append((x, y))
        if len(robot.last_positions) > STUCK_WINDOW:
            robot.last_positions.pop(0)     # drop the oldest entry
        if len(robot.last_positions) < STUCK_WINDOW:
            return False    # not enough history yet
        x0, y0 = robot.last_positions[0]
        return math.sqrt((x - x0) ** 2 + (y - y0) ** 2) < STUCK_THRESHOLD
 
    def control_loop(self):
        # runs at 10Hz, computes and publishes velocity commands for each robot
 
        for i, robot in enumerate(self.robots):
            cmd = TwistStamped()   # starts as zero velocity
 
            # no goal assigned yet, publish zero and wait
            if robot.goal_x is None:
                self.cmd_pubs[i].publish(cmd)
                continue
 
            pose = self.get_robot_pose(robot)
            if pose is None:
                continue   # TF not ready yet, skip this tick
 
            x, y, yaw = pose
            now = time.time()
 
            # vector from robot to goal
            dx = robot.goal_x - x
            dy = robot.goal_y - y
            dist = math.sqrt(dx * dx + dy * dy)
 
            # close enough, goal reached, clear it and wait for next frontier assignment
            if dist < GOAL_REACHED_DIST:
                self.get_logger().info(f'{robot.name} reached frontier.')
                robot.goal_x = None
                robot.goal_y = None
                self.cmd_pubs[i].publish(cmd)
                continue
 
            # in escape mode, just spin and inch forward until escape_until timestamp passes
            if now < robot.escape_until:
                cmd.twist.linear.x = 0.02
                cmd.twist.angular.z = 1.2 * robot.escape_dir
                self.cmd_pubs[i].publish(cmd)
                continue
 
            # attractive force pulling robot toward its goal
            goal_fx = 1.0 * dx
            goal_fy = 1.0 * dy
 
            # repulsive force pushing robot away from obstacles
            obs_fx, obs_fy = self.compute_repulsion(robot)
 
            # add them together to get the combined direction to move
            total_fx = goal_fx + obs_fx
            total_fy = goal_fy + obs_fy
 
            # convert combined force vector to a target heading angle
            target_angle = math.atan2(total_fy, total_fx)
 
            # how far off is the robot from where it needs to point
            angle_error = self.normalize_angle(target_angle - yaw)
 
            # if robot hasnt moved enough recently, trigger escape mode
            if self.is_stuck(robot, x, y):
                robot.escape_until = now + ESCAPE_DURATION
                robot.escape_dir = random.choice([-1.0, 1.0])   # random spin direction
                self.get_logger().warn(f'{robot.name} STUCK -> escape')
 
            # alignment is 1.0 when pointing at goal, 0 when facing sideways, negative when facing away
            # this slows the robot down until its roughly pointed in the right direction
            alignment = max(0.0, math.cos(angle_error))
 
            # turn toward target angle, clamped to max angular speed
            cmd.twist.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, 1.4 * angle_error))
 
            # move forward proportional to how well aligned we are, clamped to max linear speed
            cmd.twist.linear.x = max(-LINEAR_MAX, min(LINEAR_MAX, 0.12 * alignment))
 
            self.cmd_pubs[i].publish(cmd)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = SwarmCoordinator()
    rclpy.spin(node)       # keeps the node running and processing callbacks
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
