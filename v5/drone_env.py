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
    
    VERSION = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.np_random = None  # Инициализация генератора случайных чисел
        print(f"Initializing DroneEnv version {self.VERSION}")
        
        # Инициализация ROS2 узла
        rclpy.init()
        self.node = Node('drone_gym')
        
        # Пространство действий: нормализованная тяга для вертикального движения
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Пространство наблюдений: [высота, вертикальная_скорость, is_armed]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, 0.0]),
            high=np.array([10.0, 2.0, 1.0]),
            dtype=np.float32
        )
        
        # Целевая высота
        self.target_height = 5.0  # Явная инициализация
        
        # Публикаторы и подписчики (остаются без изменений)
        self.command_publisher = self.node.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )
        self.offboard_publisher = self.node.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10
        )
        self.trajectory_publisher = self.node.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10
        )
        self.status_sub = self.node.create_subscription(
            VehicleStatusV1,
            '/fmu/out/vehicle_status_v1',
            self.status_callback,
            10
        )
        self.local_pos_sub = self.node.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_pos_callback,
            10
        )
        
        # Переменные состояния
        self.vehicle_status = None
        self.local_position = None
        self.is_armed = False
        self.nav_state = None
    
    def status_callback(self, msg):
        self.vehicle_status = msg
        self.is_armed = msg.arming_state == VehicleStatusV1.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state
    
    def local_pos_callback(self, msg):
        self.local_position = msg
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # Инициализация генератора
        
        default_options = {'initial_height': 0.0, 'timeout': 3.0}
        if options:
            default_options.update(options)
            
        start_time = time.time()
        
        # Отправка начальной позиции
        self._publish_setpoint([0.0, 0.0, default_options['initial_height']], 0.0)
        
        # Активация оффборд режима
        self._publish_offboard_mode()
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        
        # Арминг дрона
        if not self.is_armed:
            self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
        
        # Ожидание инициализации
        while self.local_position is None or not self.is_armed:
            rclpy.spin_once(self.node, timeout_sec=0.1)  # Улучшенная обработка
            if time.time() - start_time > default_options['timeout']:
                raise TimeoutError("Reset timeout")
        
        observation = self._get_observation()
        info = {
            'initial_height': default_options['initial_height'],
            'is_armed': self.is_armed,
            'nav_state': self.nav_state
        }
        return observation, info
    
    def step(self, action):
        """Шаг среды"""
        self._publish_offboard_mode()
        
        # Преобразование действия: управление скоростью изменения высоты
        height = self.target_height * action[0]  # Исправление
        
        self._publish_setpoint([0.0, 0.0, height], 0.0)
        rclpy.spin_once(self.node)
        
        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        terminated, truncated = self._check_termination(observation)
        
        return observation, reward, terminated, truncated, {}
    
    def _check_termination(self, observation):
        """Условия завершения эпизода"""
        current_height = observation[0]
        current_velocity = observation[1]
        
        terminated = (
            current_height > 8.0 or      # Тест ожидает 8.0
            current_height < -1.0 or     # Тест проверяет -1.0
            abs(current_velocity) > 2.5  # Тест ожидает 2.5
        )
        return terminated, False
    
    # Остальные методы без изменений (_calculate_reward, _get_observation, _publish_setpoint и т.д.)
    
    def close(self):
        if hasattr(self, 'node'):
            self.node.destroy_node()
        rclpy.shutdown()
    
    @classmethod
    def get_version(cls):
        return cls.VERSION