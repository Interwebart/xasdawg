import gymnasium as gym
import numpy as np
import rclpy
import sys
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    VehicleLocalPosition,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleStatus,
    VehicleCommand,
    VehicleAttitude
)
from math import asin, atan2

MAX_POSITION_VALUE = 10.0
MAX_VELOCITY_VALUE = 2.0
MAX_ANGLE_VALUE = np.pi
DESIRED_HEIGHT = 2.0
EPISODE_MAX_STEPS = 200
HEIGHT_THRESHOLD = 0.1
CONNECTION_TIMEOUT = 25

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=sys.argv)

        self.node = Node('drone_gym_env')

        self.init_subscribers()
        self.init_publishers()
        self.define_spaces()

        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        self.step_count = 0
        self.armed = False
        self.nav_state = 0
        self.vehicle_status = None

        self.wait_for_connection()

    def init_subscribers(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.local_position_sub = self.node.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos
        )

        self.attitude_sub = self.node.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos
        )

        self.status_sub = self.node.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.status_callback,
            qos
        )

    def init_publishers(self):
        self.offboard_pub = self.node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub = self.node.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.cmd_pub = self.node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

    def define_spaces(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-MAX_POSITION_VALUE]*3 + [-MAX_VELOCITY_VALUE]*3 + [-MAX_ANGLE_VALUE]*3, dtype=np.float32),
            high=np.array([MAX_POSITION_VALUE]*3 + [MAX_VELOCITY_VALUE]*3 + [MAX_ANGLE_VALUE]*3, dtype=np.float32),
            dtype=np.float32
        )

    def wait_for_connection(self):
        self.node.get_logger().info("⌛ Подключение к PX4...")
        start_time = time.time()
        while time.time() - start_time < CONNECTION_TIMEOUT:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.vehicle_status is not None:
                self.node.get_logger().info("✅ PX4 подключен!")
                return
        raise ConnectionError("❌ Не удалось подключиться к PX4")

    def position_callback(self, msg):
        self.current_position = np.array([msg.x, msg.y, -msg.z])
        self.current_velocity = np.array([msg.vx, msg.vy, -msg.vz])

    def attitude_callback(self, msg):
        q = msg.q
        self.current_attitude = np.array([
            atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
            asin(2*(q[0]*q[2] - q[3]*q[1])),
            atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
        ])

    def status_callback(self, msg):
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2
        self.nav_state = msg.nav_state

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.timestamp = int(time.time() * 1e6)
        self.cmd_pub.publish(msg)

    def publish_offboard(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(time.time() * 1e6)
        self.offboard_pub.publish(msg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.send_command(400, 0.0)
        time.sleep(1)

        self.send_command(400, 1.0)
        time.sleep(1)

        for _ in range(15):
            self.publish_offboard()
            time.sleep(0.2)

        self.send_command(176, 1.0, 6.0)

        start_time = time.time()
        while time.time() - start_time < 5:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.nav_state == 14:
                break

        return np.concatenate([self.current_position, self.current_velocity, self.current_attitude]), {}

    def step(self, action):
        self.step_count += 1
        self.publish_offboard()

        scaled_action = np.clip(action, -0.8, 0.8) * np.array([MAX_POSITION_VALUE, MAX_POSITION_VALUE, MAX_POSITION_VALUE, MAX_ANGLE_VALUE])
        self.trajectory_pub.publish(TrajectorySetpoint(position=scaled_action[:3].tolist(), yaw=float(scaled_action[3]), timestamp=int(time.time()*1e6)))

        time.sleep(0.1)
        rclpy.spin_once(self.node, timeout_sec=0.02)

        roll, pitch, yaw = self.current_attitude
        flipped = abs(roll) > np.pi / 3 or abs(pitch) > np.pi / 3

        observation = np.concatenate([self.current_position, self.current_velocity, self.current_attitude])
        reward = -abs(self.current_position[2] - DESIRED_HEIGHT)
        done = flipped or self.step_count >= EPISODE_MAX_STEPS
        height_error = abs(self.current_position[2] - DESIRED_HEIGHT)
        velocity_penalty = -0.1 * np.linalg.norm(self.current_velocity)
        action_penalty = -0.01 * np.linalg.norm(action)
        reward = -height_error + velocity_penalty + action_penalty
        truncated = False

        return observation, reward, done, truncated, {}
