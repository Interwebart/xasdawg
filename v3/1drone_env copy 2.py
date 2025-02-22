# Файл: drone_env.py
import gymnasium as gym
import numpy as np
import rclpy
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
import time
from math import asin, atan2

# Константы окружения
MAX_POSITION_VALUE = 10.0
MAX_VELOCITY_VALUE = 2.0
MAX_ANGLE_VALUE = np.pi
EPISODE_LENGTH = 1000
DESIRED_HEIGHT = 2.0
CONNECTION_TIMEOUT = 10

# Награды
REWARD_STEP = -0.1
REWARD_CRASH = -100
HEIGHT_THRESHOLD = 0.1

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('drone_gym_env')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-MAX_POSITION_VALUE]*3 + [-MAX_VELOCITY_VALUE]*3 + [-MAX_ANGLE_VALUE]*3, dtype=np.float32),
            high=np.array([MAX_POSITION_VALUE]*3 + [MAX_VELOCITY_VALUE]*3 + [MAX_ANGLE_VALUE]*3, dtype=np.float32),
            dtype=np.float32
        )

        # Subscribers
        self.local_position_sub = self.node.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile)
            
        self.attitude_sub = self.node.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile)
            
        self.status_sub = self.node.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.status_callback,
            qos_profile)

        # Publishers
        self.offboard_control_pub = self.node.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10)
            
        self.trajectory_pub = self.node.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10)
            
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10)

        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        self.vehicle_status = None
        self.step_count = 0
        self.armed = False
        self.in_air = False
        
        self._check_connection()

    def _check_connection(self):
        start_time = time.time()
        while time.time() - start_time < CONNECTION_TIMEOUT:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.vehicle_status is not None:
                return
        raise Exception("Connection timeout")

    def position_callback(self, msg):
        self.current_position = np.array([msg.x, msg.y, msg.z])
        self.current_velocity = np.array([msg.vx, msg.vy, msg.vz])
        self.in_air = abs(msg.z) > 0.1

    def attitude_callback(self, msg):
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        roll = atan2(2.0*(q[0]*q[1] + q[2]*q[3]), 1.0 - 2.0*(q[1]**2 + q[2]**2))
        pitch = asin(2.0*(q[0]*q[2] - q[3]*q[1]))
        yaw = atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]**2 + q[3]**2))
        self.current_attitude = np.array([roll, pitch, yaw])

    def status_callback(self, msg):
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(time.time() * 1e6)
        self.offboard_control_pub.publish(msg)

    def publish_trajectory_setpoint(self, position, yaw):
        msg = TrajectorySetpoint()
        msg.position = position.tolist()
        msg.yaw = yaw
        msg.timestamp = int(time.time() * 1e6)
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.timestamp = int(time.time() * 1e6)
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(400, 1.0)
        start_time = time.time()
        while not self.armed and time.time() - start_time < 5:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self.armed:
            raise Exception("Arming failed")

    def disarm(self):
        self.publish_vehicle_command(400, 0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.disarm()
        time.sleep(0.5)
        
        self.step_count = 0
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        
        self.arm()
        self.publish_vehicle_command(176, 1.0, 6.0)
        time.sleep(1)
        
        return np.concatenate([
            self.current_position,
            self.current_velocity,
            self.current_attitude
        ]), {}

    def step(self, action):
        self.step_count += 1
        
        self.publish_offboard_control_mode()
        
        scaled_action = np.array([
            action[0] * MAX_POSITION_VALUE,
            action[1] * MAX_POSITION_VALUE,
            action[2] * MAX_POSITION_VALUE,
            action[3] * MAX_ANGLE_VALUE
        ])
        
        self.publish_trajectory_setpoint(scaled_action[:3], scaled_action[3])
        time.sleep(0.1)
        rclpy.spin_once(self.node)
        
        observation = np.concatenate([
            self.current_position,
            self.current_velocity,
            self.current_attitude
        ])
        
        reward = self._compute_reward()
        done = self._check_done()
        
        return observation, reward, done, False, {}

    def _compute_reward(self):
        if not self.armed or not self.in_air:
            return REWARD_CRASH
            
        height_error = abs(self.current_position[2] + DESIRED_HEIGHT)
        reward = REWARD_STEP - height_error
        if height_error < HEIGHT_THRESHOLD:
            reward += 10
        return reward

    def _check_done(self):
        return (not self.armed or 
                self.step_count >= EPISODE_LENGTH or
                np.any(np.abs(self.current_position) > MAX_POSITION_VALUE))

    def close(self):
        self.disarm()
        self.node.destroy_node()
        rclpy.shutdown()

