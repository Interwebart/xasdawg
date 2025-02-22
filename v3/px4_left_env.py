import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# Для ROS 2 / PX4 (заглушки, если хотите подписываться на одометрию и публиковать setpoints)
# import rclpy
# from rclpy.node import Node
# from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint

class PX4LeftEnv(gym.Env):
    """
    Упрощённая среда, где цель: сместиться на y=-5 (налево), удерживая высоту ~2 м (z=-2 в NED PX4).
    Демонстрирует структуру Gymnasium, а не полноценную интеграцию PX4.
    """

    def __init__(self):
        super().__init__()
        # Определяем пространство наблюдений (например, (y, z) )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        # Определяем пространство действий (допустим, action = dY ∈ [-1..+1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # rclpy.init(args=None)
        # self.node = rclpy.create_node("px4_left_env_node")
        # self.odom_sub = self.node.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, 10)
        # self.setpoint_pub = self.node.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)

        self.current_y = 0.0
        self.current_z = 0.0
        self.target_y = -5.0
        self.target_z = -2.0

        # Настройки эпизодов
        self.max_steps = 200
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        """
        Сбрасываем состояние среды:
          - возвращаем дрона в начальное положение
          - обнуляем счётчик шагов
          - публикуем ARM/Offboard если нужно
        """
        super().reset(seed=seed)

        # Здесь в реальной интеграции: сделать reset Gazebo, PX4
        self.step_count = 0
        self.current_y = 0.0
        self.current_z = 0.0
        # Могли бы подождать, пока дрон "стоит" на земле

        # Возвращаем observation
        obs = np.array([self.current_y, self.current_z], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        """
        action: np.array(shape=(1,)), ∈ [-1..1]. 
        Допустим, это приращение к setpoint Y.
        """
        self.step_count += 1

        # 1) Применяем действие
        dY = float(action[0]) * 0.1
        self.current_y += dY
        # Считаем, что высоту мы держим постоянной = -2
        self.current_z = -2.0

        # В реальности: здесь надо publish TrajectorySetpoint, дождаться обновления
        # self.publish_setpoint(self.current_y, self.current_z)

        # 2) "Ждём" и собираем новые данные одометрии
        time.sleep(0.05)
        # rclpy.spin_some(self.node)
        # self.current_y, self.current_z = ... (из VehicleOdometry)

        # 3) Вычисляем reward
        dist_y = abs(self.current_y - self.target_y)
        reward = -dist_y  # чем ближе к y=-5, тем выше награда
        # Если мы совсем близко, дадим бонус
        done = False
        truncated = False

        # Если достаточно близко, завершаем эпизод
        if dist_y < 0.2:
            done = True
            reward += 10.0  # бонус за достижение цели

        # Ограничение по количеству шагов
        if self.step_count >= self.max_steps:
            truncated = True

        # Итоговое obs
        obs = np.array([self.current_y, self.current_z], dtype=np.float32)
        info = {}
        return obs, reward, done, truncated, info

    def close(self):
        # self.node.destroy_node()
        # rclpy.shutdown()
        return super().close()

    # def publish_setpoint(self, y, z):
    #     # Пример публикации оффборд-сообщения:
    #     msg = TrajectorySetpoint()
    #     msg.position = [0.0, y, z]
    #     msg.yaw = 0.0
    #     # self.setpoint_pub.publish(msg)
    #     pass

    # def odom_cb(self, msg):
    #     # Считать реальную позицию
    #     # self.current_y = ...
    #     # self.current_z = ...
    #     pass
