import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import struct
import shutil
import time

count = 0

ros2_ws_path = os.path.expanduser("~/ros2_ws")
frames_dir = os.path.join(ros2_ws_path, 'small_house/data/play_room/')

def clear_folder(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    


def image_callback(msg):
    global count
    count += 1
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    frame_filename = os.path.join(frames_dir, f"{count}.jpg")
    cv2.imwrite(frame_filename, cv_image)

    print(f"frame saved {frame_filename}")
    
    




def main(args=None):
    rclpy.init(args=args)
    node = Node('room_mapper')

    clear_folder(frames_dir)

    node.create_subscription(Image, '/web_camera/image_raw', image_callback, 10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
