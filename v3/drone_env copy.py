import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import (
    VehicleLocalPosition, 
    OffboardControlMode, 
    TrajectorySetpoint, 
    VehicleStatus,
    VehicleCommand,
    VehicleAttitude
)
import time

# Топики для подписки (входящие данные)
VEHICLE_LOCAL_POSITION = '/fmu/out/vehicle_local_position'
VEHICLE_STATUS = '/fmu/out/vehicle_status_v1'
VEHICLE_ATTITUDE = '/fmu/out/vehicle_attitude'
VEHICLE_ODOMETRY = '/fmu/out/vehicle_odometry'

# Топики для публикации (исходящие команды)
TRAJECTORY_SETPOINT = '/fmu/in/trajectory_setpoint'
OFFBOARD_CONTROL_MODE = '/fmu/in/offboard_control_mode'
VEHICLE_COMMAND = '/fmu/in/vehicle_command'

# Константы окружения
MAX_POSITION_VALUE = 10.0  # метры
MAX_VELOCITY_VALUE = 2.0   # м/с
MAX_ANGLE_VALUE = np.pi    # радианы
EPISODE_LENGTH = 1000      # шаги
DESIRED_HEIGHT = 2.0       # метры

# Константы наград
REWARD_STEP = -0.1
REWARD_CRASH = -100
REWARD_SUCCESS = 100
HEIGHT_THRESHOLD = 0.1
POSITION_THRESHOLD = 0.2

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Инициализация ROS2 node
        rclpy.init()
        self.node = Node('drone_gym_env')
        
        # QoS профиль для надежной коммуникации
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Определение пространства действий (x, y, z, yaw)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Определение пространства наблюдений
        self.observation_space = gym.spaces.Box(
            low=np.array([-MAX_POSITION_VALUE, -MAX_POSITION_VALUE, 0, 
                         -MAX_VELOCITY_VALUE, -MAX_VELOCITY_VALUE, -MAX_VELOCITY_VALUE,
                         -MAX_ANGLE_VALUE, -MAX_ANGLE_VALUE, -MAX_ANGLE_VALUE]),
            high=np.array([MAX_POSITION_VALUE, MAX_POSITION_VALUE, MAX_POSITION_VALUE,
                          MAX_VELOCITY_VALUE, MAX_VELOCITY_VALUE, MAX_VELOCITY_VALUE,
                          MAX_ANGLE_VALUE, MAX_ANGLE_VALUE, MAX_ANGLE_VALUE]),
            dtype=np.float32
        )
        
        # Подписки на топики
        self.local_position_sub = self.node.create_subscription(
            VehicleLocalPosition,
            VEHICLE_LOCAL_POSITION,
            self.position_callback,
            qos_profile)
            
        self.attitude_sub = self.node.create_subscription(
            VehicleAttitude,
            VEHICLE_ATTITUDE,
            self.attitude_callback,
            qos_profile)
            
        self.status_sub = self.node.create_subscription(
            VehicleStatus,
            VEHICLE_STATUS,
            self.status_callback,
            qos_profile)
            
        # Публикаторы
        self.trajectory_pub = self.node.create_publisher(
            TrajectorySetpoint,
            TRAJECTORY_SETPOINT,
            10)
            
        self.offboard_control_pub = self.node.create_publisher(
            OffboardControlMode,
            OFFBOARD_CONTROL_MODE,
            10)
            
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand,
            VEHICLE_COMMAND,
            10)
            
        # Состояние дрона
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        self.vehicle_status = None
        self.step_count = 0
        self.armed = False
        self.in_air = False
        
    def position_callback(self, msg):
        """Обработка данных о позиции дрона"""
        self.current_position = np.array([msg.x, msg.y, msg.z])
        self.current_velocity = np.array([msg.vx, msg.vy, msg.vz])
        self.in_air = msg.z < -0.1  # Считаем что дрон в воздухе если высота больше 0.1м

    def attitude_callback(self, msg):
        """Обработка данных об ориентации дрона"""
        # msg.q содержит кватернион [w, x, y, z]
        # Преобразуем кватернион в углы Эйлера
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        # Преобразование кватерниона в углы Эйлера
        roll = np.arctan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                          1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        pitch = np.arcsin(2.0 * (q[2] * q[0] - q[3] * q[1]))
        yaw = np.arctan2(2.0 * (q[3] * q[0] + q[1] * q[2]),
                         1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        self.current_attitude = np.array([roll, pitch, yaw])

    def status_callback(self, msg):
        """Обработка статуса дрона"""
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2  # 2 означает ARMED

    def publish_trajectory_setpoint(self, position, yaw):
        """Публикация целевой точки траектории"""
        msg = TrajectorySetpoint()
        msg.position = position.tolist()
        msg.yaw = yaw
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Публикация команд управления дроном"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        """Армирование дрона"""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        time.sleep(0.1)

    def disarm(self):
        """Дизармирование дрона"""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        time.sleep(0.1)

    def reset(self, seed=None):
        """Сброс окружения в начальное состояние"""
        super().reset(seed=seed)
        
        # Дизармируем дрон
        self.disarm()
        time.sleep(0.5)
        
        # Сброс состояния
        self.step_count = 0
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        
        # Армируем дрон
        self.arm()
        time.sleep(0.5)
        
        # Получаем начальное наблюдение
        observation = np.concatenate([
            self.current_position,
            self.current_velocity,
            self.current_attitude
        ])
        
        return observation, {}

    def step(self, action):
        """Выполнение одного шага симуляции"""
        self.step_count += 1
        
        # Масштабирование действий
        scaled_action = np.array([
            action[0] * MAX_POSITION_VALUE,
            action[1] * MAX_POSITION_VALUE,
            action[2] * MAX_POSITION_VALUE,
            action[3] * MAX_ANGLE_VALUE
        ])
        
        # Отправка команды дрону
        self.publish_trajectory_setpoint(scaled_action[:3], scaled_action[3])
        
        # Даем время на выполнение действия
        time.sleep(0.1)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        # Получение нового состояния
        observation = np.concatenate([
            self.current_position,
            self.current_velocity,
            self.current_attitude
        ])
        
        # Расчет награды
        reward = self._compute_reward()
        
        # Проверка завершения эпизода
        done = self._check_done()
        
        return observation, reward, done, False, {}

    def _compute_reward(self):
        """Расчет награды"""
        reward = REWARD_STEP  # Базовая награда за шаг
        
        # Штраф за крушение
        if not self.armed or self.current_position[2] > 0:
            return REWARD_CRASH
            
        # Награда за поддержание желаемой высоты
        height_error = abs(self.current_position[2] + DESIRED_HEIGHT)
        if height_error < HEIGHT_THRESHOLD:
            reward += 1.0
            
        # Штраф за большую скорость
        velocity_penalty = -0.1 * np.linalg.norm(self.current_velocity)
        reward += velocity_penalty
        
        return reward

    def _check_done(self):
        """Проверка условий завершения эпизода"""
        # Эпизод завершается если:
        # 1. Дрон разбился (не армирован или коснулся земли)
        # 2. Превышено максимальное количество шагов
        # 3. Дрон вышел за пределы допустимой области
        if not self.armed:
            return True
            
        if self.step_count >= EPISODE_LENGTH:
            return True
            
        if np.any(np.abs(self.current_position) > MAX_POSITION_VALUE):
            return True
            
        return False

    def close(self):
        """Закрытие окружения"""
        self.disarm()
        self.node.destroy_node()
        rclpy.shutdown()