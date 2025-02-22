import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from px4_msgs.msg import VehicleStatus, BatteryStatus, VehicleAttitude, VehicleGlobalPosition
from .topics_config import topics

class OutHandler(Node):
    def __init__(self):
        super().__init__('out_handler')
        self._data = {}
        self._msg_types = {
            'vehicle_status_v1': VehicleStatus,
            'battery_status': BatteryStatus,
            'vehicle_attitude': VehicleAttitude,
            'vehicle_global_position': VehicleGlobalPosition
        }
        
        self._init_subscriptions()

    def _init_subscriptions(self):
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        for topic_name in topics["outbound"]:
            if msg_type := self._msg_types.get(topic_name):
                self.create_subscription(
                    msg_type,
                    topics["outbound"][topic_name],
                    lambda msg, tn=topic_name: self._callback(msg, tn),
                    qos
                )
                self._data[topic_name] = None

    def _callback(self, msg, topic_name: str):
        self._data[topic_name] = {
            field: getattr(msg, field)
            for field in msg.__slots__
            if not field.startswith('_')
        }

    def get(self, topic_name: str) -> dict:
        """Возвращает словарь данных топика или пустой словарь"""
        return self._data.get(topic_name, {})