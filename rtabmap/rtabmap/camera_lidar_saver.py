#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import struct
import shutil
import time

ros2_ws_path = os.path.expanduser("~/ros2_ws")
frames_dir = os.path.join(ros2_ws_path, 'map_data/small_house_FINAL/frames')
lidar_dir = os.path.join(ros2_ws_path, 'map_data/small_house_FINAL/lidar_scans')
pose_dir = os.path.join(ros2_ws_path, 'map_data/small_house_FINAL/poses')

last_pose_vec = None

def clear_folder(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    
    

def get_timestamp():
    return f"{time.time():.6f}"  



def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    timestamp = get_timestamp()
    frame_filename = os.path.join(frames_dir, f"{timestamp}.jpg")
    cv2.imwrite(frame_filename, cv_image)

    print(f"frame saved {frame_filename}")
    
    

# def lidar_callback(msg):
#     lidar_data = pointcloud2_to_array(msg)

#     timestamp = get_timestamp()
#     lidar_filename = os.path.join(lidar_dir, f"{timestamp}.npz")
#     np.savez_compressed(lidar_filename, lidar_data)

#     print(f"lidar saved{lidar_filename}")
    
    

# def pose_callback(msg):
#   #  pose = msg.pose
#     pose = msg.pose.pose


#     timestamp = get_timestamp()
#     # pose_filename = os.path.join(pose_dir, f"{timestamp}.txt")
#     # with open(pose_filename, "w") as pose_file:
#     #     pose_file.write(f"{pose.position.x} {pose.position.y} {pose.position.z} "
#     #                     f"{pose.orientation.x} {pose.orientation.y} {pose.orientation.z} {pose.orientation.w}")
    
#     pose_log_file = os.path.join(pose_dir, "poses.csv")
    
#     print("POSE : ", pose)

#     with open(pose_log_file, "a") as f:
#         f.write(f"{timestamp},{pose.position.x},{pose.position.y},{pose.position.z},"
#                 f"{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n")

#     print(f"pose saved at time {timestamp}")
    
    
def pose_callback(msg):
    pose = msg.pose.pose
    x = float(pose.position.x)
    y = float(pose.position.y)
    z = float(pose.position.z)
    qx = float(pose.orientation.x)
    qy = float(pose.orientation.y)
    qz = float(pose.orientation.z)
    qw = float(pose.orientation.w)

    pose_vec = (x, y, z, qx, qy, qz, qw)

    global last_pose_vec
    if last_pose_vec is not None and pose_vec == last_pose_vec:
        print("pose duplicate skipped")
        return
    else:
        timestamp = get_timestamp()
        pose_log_file = os.path.join(pose_dir, "poses.csv")

        cov = list(msg.pose.covariance)

        print("POSE : ", pose_vec)

        with open(pose_log_file, "a") as f:
            f.write(
                f"{timestamp},{x},{y},{z},{qx},{qy},{qz},{qw}," +
                ",".join(str(c) for c in cov) + "\n"
            )

        last_pose_vec = pose_vec
        print(f"pose saved at time {timestamp}")

# def pointcloud2_to_array(cloud_msg):
    
#     point_step = cloud_msg.point_step       #   gets size of each point 
#     data = np.frombuffer(cloud_msg.data, dtype=np.uint8)        #  get it into format numpy can work with
#     num_points = len(data) // point_step   # size of data divided by size of each point gets num of points ( dont want decimal)
    
#     points = []

#     for i in range(num_points):        
#         start = i * point_step             # get start of each point
#         x, y, z = struct.unpack_from('fff', data, start)  # extract x y z
#         points.append([x, y, z]) 

#     return np.array(points)



def main(args=None):
    rclpy.init(args=args)
    node = Node('camera_lidar_saver')

    clear_folder(frames_dir)
    clear_folder(lidar_dir)
    clear_folder(pose_dir)

    node.create_subscription(Image, '/web_camera/image_raw', image_callback, 10)
  #  node.create_subscription(PointCloud2, '/ouster/points', lidar_callback, 10)
  #  node.create_subscription(PoseStamped, '/localization_pose', pose_callback, 10)
    node.create_subscription(Odometry, '/odometry/filtered', pose_callback, 10)


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
