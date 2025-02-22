import yaml
import rclpy
from rclpy.node import Node
from threading import Lock
from std_msgs.msg import String, Float32, Int32, Bool
from px4_msgs.msg import *

class TopicHandler(Node):
    def __init__(self, yaml_path="topic.yml"):
        super().__init__('topic_handler')
        with open(yaml_path, "r") as file:
            self.topics = yaml.safe_load(file)["topics"]
        
        self.data = {}
        self.lock = Lock()
        self.create_subscriptions()
    
    def create_subscriptions(self):
        """Создает подписки на все топики из конфигурационного файла."""
        message_types = {
            "battery_status": BatteryStatus,
            "vehicle_attitude": VehicleAttitude,
            "vehicle_global_position": VehicleGlobalPosition,
            "vehicle_local_position": VehicleLocalPosition,
            "vehicle_status_v1": VehicleStatus
        }
        
        for category in self.topics.values():
            for topic_name, topic_path in category.items():
                msg_type = message_types.get(topic_name, String)  # По умолчанию String для неизвестных топиков
                self.create_subscription(msg_type, topic_path, self.generic_callback(topic_name), 10)
    
    def generic_callback(self, topic_name):
        """Обработчик сообщений для всех подписанных топиков."""
        def callback(msg):
            with self.lock:
                self.data[topic_name] = self.msg_to_dict(msg)
        return callback
    
    def msg_to_dict(self, msg):
        """Преобразует сообщение ROS2 в словарь."""
        return {field: getattr(msg, field) for field in dir(msg) if not field.startswith('_') and not callable(getattr(msg, field))}
    
    def get_topic_value(self, topic_name):
        """Возвращает последнее полученное значение из топика мгновенно."""
        with self.lock:
            return self.data.get(topic_name, "Нет данных")

# Пример использования
if __name__ == "__main__":
    rclpy.init()
    handler = TopicHandler()
    rclpy.spin(handler)
    rclpy.shutdown()