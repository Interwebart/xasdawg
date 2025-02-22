#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
import time
from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleStatus,
    VehicleLocalPosition
)

class DroneEnv(gym.Env):
    """Custom Environment for PX4 drone vertical takeoff task"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize state variables first
        self.vehicle_status = None
        self.local_position = None
        self.is_armed = False
        self.nav_state = None
        self.steps = 0
        
        # Initialize ROS2 context only once
        if not rclpy.ok():
            rclpy.init()
            
        self.node = Node('drone_gym_env', namespace='')
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Spaces with explicit float32 dtype
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Configuration
        self.target_height = 5.0
        
        # ROS2 Setup
        self._setup_publishers()
        self._setup_subscribers(qos_profile)
        self._wait_for_connection()
        
    def _setup_publishers(self):
        """Initialize ROS2 publishers"""
        self.command_publisher = self.node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_publisher = self.node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_publisher = self.node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
    def _setup_subscribers(self, qos_profile):
        """Initialize ROS2 subscribers"""
        self.status_sub = self.node.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self.status_callback, qos_profile)
        self.local_pos_sub = self.node.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.local_pos_callback, qos_profile)
        
    def _wait_for_connection(self):
        """Wait for PX4 connection"""
        timeout = 25.0  # Increased timeout
        start_time = time.time()
        self.node.get_logger().info('⌛ Connecting to PX4...')
        
        while (self.vehicle_status is None or self.local_position is None):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError("Failed to connect to PX4")
            time.sleep(0.1)
        self.node.get_logger().info('✅ Connection established!')

    # Rest of the class remains the same...      
    def status_callback(self, msg):
        """Handle vehicle status updates"""
        self.vehicle_status = msg
        self.is_armed = msg.arming_state == 2
        self.nav_state = msg.nav_state
        
    def local_pos_callback(self, msg):
        """Handle local position updates"""
        self.local_position = msg
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.steps = 0
        options = options or {}
        default_options = {'initial_height': 0.0, 'timeout': 3.0}
        default_options.update(options)
        
        # Send reset commands
        self._publish_setpoint([0.0, 0.0, default_options['initial_height']], 0.0)
        self._publish_offboard_mode()
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        
        if not self.is_armed:
            self._send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
        
        # Wait for initialization
        start_time = time.time()
        while self.local_position is None or not self.is_armed:
            if time.time() - start_time > default_options['timeout']:
                raise TimeoutError("Reset timeout")
            rclpy.spin_once(self.node)
            time.sleep(0.1)
        
        return self._get_observation(), {
            'initial_height': default_options['initial_height'],
            'is_armed': self.is_armed,
            'nav_state': self.nav_state
        }
    
    def step(self, action):
        """Execute one time step"""
        self._publish_offboard_mode()
        height = (action[0] + 1) * self.target_height / 2
        self._publish_setpoint([0.0, 0.0, height], 0.0)
        rclpy.spin_once(self.node)
        
        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        terminated, truncated = self._check_termination(observation)
        self.steps += 1
        
        return observation, reward, terminated, truncated, {}
    
    def _calculate_reward(self, observation):
        """Calculate step reward"""
        current_height = observation[0]
        current_velocity = observation[1]
        height_error = abs(self.target_height - current_height)
        reward = -height_error - 0.1 * abs(current_velocity)
        if height_error < 0.1 and abs(current_velocity) < 0.1:
            reward += 10.0
        return reward
    
    def _check_termination(self, observation):
        """Check termination conditions"""
        current_height = observation[0]
        current_velocity = observation[1]
        terminated = (
            current_height > 7.0 or
            current_height < 0.0 or
            abs(current_velocity) > 2.0
        )
        truncated = self.steps >= 100
        return terminated, truncated
    
    def _get_observation(self):
        """Get current observation"""
        if self.local_position is None:
            return np.zeros(3, dtype=np.float32)
        return np.array([
            -self.local_position.z,
            -self.local_position.vz,
            float(self.is_armed)
        ], dtype=np.float32)
    
    def _publish_setpoint(self, position, yaw):
        """Publish trajectory setpoint"""
        msg = TrajectorySetpoint()
        msg.position = position
        msg.yaw = yaw
        self.trajectory_publisher.publish(msg)
    
    def _publish_offboard_mode(self):
        """Publish offboard control mode"""
        msg = OffboardControlMode()
        msg.position = True
        self.offboard_publisher.publish(msg)
    
    def _send_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Send vehicle command"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.from_external = True
        self.command_publisher.publish(msg)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()