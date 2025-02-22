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
CONNECTION_TIMEOUT = 25  # Увеличенный таймаут

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        if not rclpy.ok():
            rclpy.init(args=sys.argv)
            
        self.node = Node('drone_gym_env')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Инициализация подписчиков
        self.vehicle_status = None
        self.init_subscribers(qos)
        
        # Инициализация публикаторов
        self.init_publishers()
        
        # Пространства действий и состояний
        self.define_spaces()
        
        # Переменные состояния
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)
        self.step_count = 0
        self.armed = False
        self.nav_state = 0

        # Подключение
        self.wait_for_connection()

    def init_subscribers(self, qos):
        """Инициализация подписчиков"""
        self.local_position_sub = self.node.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos)
            
        self.attitude_sub = self.node.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos)

        self.status_sub = self.node.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.status_callback,
            qos)

    def init_publishers(self):
        """Инициализация публикаторов"""
        self.offboard_pub = self.node.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10)
            
        self.trajectory_pub = self.node.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10)

        self.cmd_pub = self.node.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10)

    def define_spaces(self):
        """Определение пространств"""
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

    def wait_for_connection(self):
        """Ожидание подключения к PX4"""
        self.node.get_logger().info("⌛ Подключение к PX4...")
        start_time = time.time()
        
        while time.time() - start_time < CONNECTION_TIMEOUT:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.vehicle_status is not None:  # Проверяем, что статус получен
                self.node.get_logger().info("✅ Соединение установлено!")
                return
                    
        raise ConnectionError(f"❌ Не удалось подключиться за {CONNECTION_TIMEOUT} секунд")


    def position_callback(self, msg):
        """Обработка позиции (NED -> ENU)"""
        self.current_position = np.array([msg.x, msg.y, -msg.z])  # Инверсия Z
        self.current_velocity = np.array([msg.vx, msg.vy, -msg.vz])

    def attitude_callback(self, msg):
        """Расчет углов Эйлера из кватерниона"""
        q = msg.q
        self.current_attitude = np.array([
            atan2(2*(q[0]*q[1] + q[2]*q[3]), 1-2*(q[1]**2 + q[2]**2)),
            asin(2*(q[0]*q[2] - q[3]*q[1])),
            atan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(q[2]**2 + q[3]**2))
        ])

    def status_callback(self, msg):
        """Обработка статуса дрона"""
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2
        self.nav_state = msg.nav_state
        self.node.get_logger().debug(
            f"ARM: {self.armed} | NAV: {self.nav_state}",
            throttle_duration_sec=2.0
        )

    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.step_count = 0

        # Дизармирование
        self.send_command(400, 0.0)
        time.sleep(1)
        
        # Армирование
        self.send_command(400, 1.0)
        
        # Активация Offboard
        for _ in range(15):
            self.publish_offboard()
            time.sleep(0.2)
        
        # Установка режима
        self.send_command(176, 1.0, 6.0)
        
        # Ожидание активации
        start_time = time.time()
        while time.time() - start_time < 5:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.nav_state == 14:
                break

        return np.concatenate([
            self.current_position,
            self.current_velocity,
            self.current_attitude
        ]), {}

    def step(self, action):
        """Выполнение шага"""
        self.step_count += 1
        
        # Публикация управления
        self.publish_offboard()
        
        # Преобразование действия
        scaled_action = np.clip(action, -0.8, 0.8) * np.array([
            MAX_POSITION_VALUE,
            MAX_POSITION_VALUE,
            -MAX_POSITION_VALUE,  # Инверсия для оси Z
            MAX_ANGLE_VALUE
        ])
        
        self.trajectory_pub.publish(
            TrajectorySetpoint(
                position=scaled_action[:3].tolist(),
                yaw=float(scaled_action[3]),
                timestamp=int(time.time()*1e6)
        ))

        # Ожидание обновления
        start_time = time.time()
        while time.time() - start_time < 0.1:
            rclpy.spin_once(self.node, timeout_sec=0.02)

        # Расчет награды
        height_error = abs(self.current_position[2] - DESIRED_HEIGHT)
        velocity_penalty = 0.2 * np.linalg.norm(self.current_velocity)
        reward = -height_error - velocity_penalty
        
        # Штраф за переворот
        if self.current_position[2] < 0.1:
            reward -= 200
            done = True
        else:
            done = self.step_count >= EPISODE_MAX_STEPS or height_error < HEIGHT_THRESHOLD

        return (
            np.concatenate([self.current_position, self.current_velocity, self.current_attitude]),
            reward,
            done,
            False,
            {}
        )

    def publish_offboard(self):
        """Публикация режима Offboard"""
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(time.time()*1e6)
        self.offboard_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        """Отправка команды"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.timestamp = int(time.time()*1e6)
        self.cmd_pub.publish(msg)

    def close(self):
        """Завершение работы"""
        self.send_command(400, 0.0)
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    env = DroneEnv()
    try:
        obs, _ = env.reset()
        for _ in range(50):
            action = [0, 0, 0.2, 0]  # Взлет
            obs, rew, done, _, _ = env.step(action)
            print(f"Высота: {obs[2]:.2f}m | Награда: {rew:.2f}")
            if done: break
    finally:
        env.close()