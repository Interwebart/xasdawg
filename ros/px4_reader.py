import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from px4_msgs.msg import (
    VehicleStatus, BatteryStatus, EstimatorStatusFlags, FailsafeFlags,
    ManualControlSetpoint, PositionSetpointTriplet, SensorCombined,
    TimesyncStatus, VehicleAttitude, VehicleCommandAck,
    VehicleControlMode, VehicleGlobalPosition,
    VehicleLandDetected, VehicleLocalPosition, VehicleOdometry
)

class PX4Reader(Node):
    def __init__(self):
        super().__init__('px4_reader')
        
        # Инициализация переменных для хранения данных с топиков
        self.vehicle_status = None
        self.battery_status = None
        self.estimator_status_flags = None
        self.failsafe_flags = None
        self.manual_control_setpoint = None
        self.position_setpoint_triplet = None
        self.sensor_combined = None
        self.timesync_status = None
        self.vehicle_attitude = None
        self.vehicle_command_ack = None
        self.vehicle_control_mode = None
        self.vehicle_global_position = None
        self.vehicle_gps_position = None
        self.vehicle_land_detected = None
        self.vehicle_local_position = None
        self.vehicle_odometry = None

        # Настройка QoS (используем BEST_EFFORT для совместимости)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Создание подписок на все топики
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.vehicle_status_callback, qos_profile)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status", self.battery_status_callback, qos_profile)
        self.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", self.estimator_status_flags_callback, qos_profile)
        self.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags", self.failsafe_flags_callback, qos_profile)
        self.create_subscription(ManualControlSetpoint, "/fmu/out/manual_control_setpoint", self.manual_control_setpoint_callback, qos_profile)
        self.create_subscription(PositionSetpointTriplet, "/fmu/out/position_setpoint_triplet", self.position_setpoint_triplet_callback, qos_profile)
        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined", self.sensor_combined_callback, qos_profile)
        self.create_subscription(TimesyncStatus, "/fmu/out/timesync_status", self.timesync_status_callback, qos_profile)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleCommandAck, "/fmu/out/vehicle_command_ack", self.vehicle_command_ack_callback, qos_profile)
        self.create_subscription(VehicleControlMode, "/fmu/out/vehicle_control_mode", self.vehicle_control_mode_callback, qos_profile)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.vehicle_global_position_callback, qos_profile)
        self.create_subscription(VehicleLandDetected, "/fmu/out/vehicle_land_detected", self.vehicle_land_detected_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.vehicle_local_position_callback, qos_profile)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.vehicle_odometry_callback, qos_profile)

    # Коллбэки для обработки сообщений из топиков
    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg
        # self.get_logger().info(f"Status received: {msg.arming_state}")

    def battery_status_callback(self, msg):
        self.battery_status = msg
        # self.get_logger().info(f"Battery remaining: {msg.remaining * 100:.1f}%")

    def estimator_status_flags_callback(self, msg):
        self.estimator_status_flags = msg

    def failsafe_flags_callback(self, msg):
        self.failsafe_flags = msg

    def manual_control_setpoint_callback(self, msg):
        self.manual_control_setpoint = msg

    def position_setpoint_triplet_callback(self, msg):
        self.position_setpoint_triplet = msg

    def sensor_combined_callback(self, msg):
        self.sensor_combined = msg

    def timesync_status_callback(self, msg):
        self.timesync_status = msg

    def vehicle_attitude_callback(self, msg):
        self.vehicle_attitude = msg

    def vehicle_command_ack_callback(self, msg):
        self.vehicle_command_ack = msg

    def vehicle_control_mode_callback(self, msg):
        self.vehicle_control_mode = msg

    def vehicle_global_position_callback(self, msg):
        self.vehicle_global_position = msg

    def vehicle_gps_position_callback(self, msg):
        self.vehicle_gps_position = msg

    def vehicle_land_detected_callback(self, msg):
        self.vehicle_land_detected = msg

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def vehicle_odometry_callback(self, msg):
        self.vehicle_odometry = msg


# Пример использования
def main():
    rclpy.init()
    reader = PX4Reader()
    rclpy.spin(reader)
    rclpy.shutdown()

if __name__ == "__main__":
    main()