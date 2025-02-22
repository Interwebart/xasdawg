import rclpy
from ros.ros_handler import TopicHandler

def main():
    rclpy.init()
    handler = TopicHandler()
    rclpy.spin(handler)
    rclpy.shutdown()

if __name__ == "__main__":
    main()