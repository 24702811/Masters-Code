#!/usr/bin/env python3

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import faiss
import numpy as np
import pickle
from PIL import Image
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import struct
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as RosImage
from PIL import Image
import cv2
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation as R
#from torchvision.models import ResNet50_Weights
import csv
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


curr_image = None
curr_scan = None
curr_pose = None

def publish_pose(pose, covariance, node, pose_publisher):
    
        
    estimated_pose = PoseWithCovarianceStamped()
    estimated_pose.header.stamp = node.get_clock().now().to_msg()
    estimated_pose.header.frame_id = "map"

    estimated_pose.pose.pose.position.x = pose[0, 3]
    estimated_pose.pose.pose.position.y = pose[1, 3]
    estimated_pose.pose.pose.position.z = pose[2, 3]

    q = R.from_matrix(pose[:3, :3]).as_quat()
    estimated_pose.pose.pose.orientation.x = q[0]
    estimated_pose.pose.pose.orientation.y = q[1]
    estimated_pose.pose.pose.orientation.z = q[2]
    estimated_pose.pose.pose.orientation.w = q[3]

    estimated_pose.pose.covariance = covariance.reshape(-1).astype(float).tolist()

    pose_publisher.publish(estimated_pose)

def pointcloud2_to_array(cloud_msg):
    point_step = cloud_msg.point_step
    data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    num_points = len(data) // point_step
    points = []

    for i in range(num_points):
        start = i * point_step
        x, y, z = struct.unpack_from('fff', data, start)
        points.append([x, y, z])

    return np.array(points)


def image_callback(msg):
    global curr_image
    bridge = CvBridge()
    curr_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    
def lidar_callback(msg):
    global curr_scan
    curr_scan = pointcloud2_to_array(msg)
    
def pose_callback(msg):
    global curr_pose
    curr_pose = msg.pose


def load_model(model_path, mean_path):
    
    
    # model = models.resnet50(pretrained=True)
    # model = torch.nn.Sequential(*list(model.children())[:-1])  
    # model.eval() 
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Identity()  

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    
    
    with open(mean_path, "rb") as f:
        norm_values = pickle.load(f)

    mean = norm_values["mean"]
    std = norm_values["std"]
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=mean, std=std) ])
    return model, transform


def load_index(index_path, image_paths_file):
    index = faiss.read_index(index_path) 
    with open(image_paths_file, "rb") as f:
        image_paths = pickle.load(f)   
    return index, image_paths


def load_confidence_function(path_npz):
    data = np.load(path_npz)
    xs = data["xs"].astype(np.float32)
    ys = data["ys"].astype(np.float32)
    return xs, ys

def preprocess_image(image, transform):
    #image = Image.open(image_path).convert("RGB")  
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0).to(device)


def extract_descriptor(processed_image, model):
    with torch.no_grad():
        descriptor = model(processed_image).squeeze(0).cpu()
        #descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=0)  
    return descriptor


def match_descriptor(index, image_paths, query_desc):
    distances, indices = index.search(query_desc, 1) 

    matched_places = [image_paths[idx] for idx in indices[0]]
    image = cv2.imread(matched_places[0])
    print(matched_places[0])


    return matched_places, distances



def get_match_info(matched_places, lidar_scans_path, poses_csv_path):

    image_timestamp = float(os.path.splitext(os.path.basename(matched_places[0]))[0])
    

    lidar_files = sorted(os.listdir(lidar_scans_path), key=lambda f: float(os.path.splitext(f)[0]))
   # pose_files = sorted(os.listdir("ros2_ws/small_house/poses/"), key=lambda f: float(os.path.splitext(f)[0]))

    lidar_timestamps = [float(os.path.splitext(f)[0]) for f in lidar_files]
  #  pose_timestamps = [float(os.path.splitext(f)[0]) for f in pose_files]
    
    closest_lidar_id = None
    min = float('inf')  

    for i in range(len(lidar_timestamps)):
        diff = abs(lidar_timestamps[i] - image_timestamp)
        
        if diff < min:
            min = diff
            closest_lidar_id = i
        
    closest_lidar_file = lidar_files[closest_lidar_id]
    npz = np.load(os.path.join(lidar_scans_path, closest_lidar_file))
   # print(f"Keys in {closest_lidar_file}: {npz.files}")
    matched_scan = npz['arr_0']  
    
    
    # with np.load(os.path.join("ros2_ws/mapping/data/lidar_scans/", closest_lidar_file)) as data:
    #     matched_scan = data['a']
    
    
    # closest_pose_id = None
    # min = float('inf')  

    # for i in range(len(pose_timestamps)):
    #     diff = abs(pose_timestamps[i] - image_timestamp)
        
    #     if diff < min:
    #         min = diff
    #         closest_pose_id = i
            
            
    # closest_pose_file = pose_files[closest_pose_id]
    # with open(os.path.join("ros2_ws/mapping/data/poses/", closest_pose_file), "r") as file:
    #     line = file.readline().strip()  
    #     matches_translation = list(map(float, line.split()[:3]))  
    #     matches_rotation = list(map(float, line.split()[3:]))  
    
    
    
    closest_pose = None
    min_diff = float('inf')

    with open(poses_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            ts = float(row[0])
            diff = abs(ts - image_timestamp)
            if diff < min_diff:
                min_diff = diff
                matches_translation = list(map(float, row[1:4]))
                matches_rotation = list(map(float, row[4:]))
        
   # print(matched_places)
    # print(closest_lidar_file)
    # print(closest_pose_file)
    
    
    return matched_scan, matches_translation, matches_rotation


def interpolate_pose(timestamp, poses_csv_path):

    data = np.loadtxt(poses_csv_path, delimiter=",")
    T = data[:, 0]        
    P = data[:, 1:4]      
    Q = data[:, 4:8]      
    C = data[:, 8:44] 

    order =np.argsort(T)
    T = T[order]
    P = P[order]
    Q = Q[order]
    C = C[order]
    
    later_pose_index = int(np.searchsorted(T, timestamp, side='left'))
    if later_pose_index <= 0:
        earlier_pose = 0
        later_pose = 0
    elif later_pose_index >= len(T):
        earlier_pose = len(T) - 1
        later_pose = len(T) - 1
    else:
        earlier_pose = later_pose_index - 1
        later_pose = later_pose_index


    if earlier_pose == later_pose:
        near = earlier_pose
    else: 
        if abs(T[earlier_pose] - timestamp) <= abs(T[later_pose] - timestamp):
            near = earlier_pose
        else:
            near = later_pose
    covariance = C[near].reshape(6, 6)

    # diffs = np.abs(T - timestamp)
    # diffs[T == T[closest_pose]] = np.inf
    # other_pose = int(np.argmin(diffs))


    # if T[closest_pose] <= T[other_pose]:
    #     earlier_pose = closest_pose
    #     later_pose = other_pose
    # else:
    #     earlier_pose = other_pose
    #     later_pose = closest_pose
    
    if timestamp < T[earlier_pose]:
        pose = (T[earlier_pose], *P[earlier_pose], *Q[earlier_pose])
        translation = P[earlier_pose]
        rotation = Q[earlier_pose]
        
    elif timestamp > T[later_pose]:
        pose = (T[later_pose], *P[later_pose], *Q[later_pose])
        translation = P[later_pose]
        rotation = Q[later_pose]
        
    else:
        p = interp1d([T[earlier_pose], T[later_pose]], P[[earlier_pose, later_pose]], axis=0)(timestamp)
        slerp = Slerp([T[earlier_pose], T[later_pose]], R.from_quat(Q[[earlier_pose, later_pose]]))
        q = slerp([timestamp]).as_quat()[0]
        pose = (timestamp, *p, *q)
        translation = p
        rotation = q
    
    print(T[earlier_pose], "  ", T[later_pose])
    
    return translation, rotation, covariance



def main(args=None):
    rclpy.init(args=args)
    node = Node('find_initial_pose')
    node.create_subscription(RosImage, '/web_camera/image_raw', image_callback, 10)
    node.create_subscription(PointCloud2, '/ouster/points', lidar_callback, 10)

    pose_publisher = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    
    model_path = "ros2_ws/mapping/SMALL_HOUSE_FINAL_MODEL.pth"
    mean_path = "ros2_ws/mapping/SMALL_HOUSE_FINAL_MEAN.pkl"
    index_path = "ros2_ws/mapping/SMALL_HOUSE_FINAL_INDEX.bin"
    image_paths_file = "ros2_ws/mapping/SMALL_HOUSE_FINAL_IMAGE_PATHS.pkl"
   # lidar_scans_path = "ros2_ws/small_house/lidar_scans/"
    poses_csv_path = "ros2_ws/map_data/small_house_FINAL/poses/poses.csv"
    confidence_function_path = "ros2_ws/mapping/SMALL_HOUSE_CONF_FINAL.npz"


    model, transform = load_model(model_path, mean_path)
    print("model loaded")
    index, image_paths = load_index(index_path, image_paths_file)
    print("index loaded")
    d_conf, y_conf = load_confidence_function(confidence_function_path)

    global curr_image, curr_scan

    while rclpy.ok() and (curr_image is None or curr_scan is None):
        rclpy.spin_once(node, timeout_sec=0.1)

    processed_image = preprocess_image(curr_image, transform)
    descriptor = extract_descriptor(processed_image, model).reshape(1, -1)
    matched_places, distances = match_descriptor(index, image_paths, descriptor)
   # matched_scan, initial_translation, initial_rotation = get_match_info(matched_places, lidar_scans_path, poses_csv_path)\
    image_timestamp = float(os.path.splitext(os.path.basename(matched_places[0]))[0])
    translation, rotation, covariance = interpolate_pose(image_timestamp, poses_csv_path)
    
    distance = np.clip(distances[0], d_conf[0], d_conf[-1])
    confidence = np.interp(distance, d_conf, y_conf)
    
    # confidence = ((4 - distances[0]) / 4) * 100
    print("distance", distances[0], "and confidence", confidence)
    print(distance)

    r = R.from_quat(rotation).as_matrix()
    T_init = np.eye(4)
    T_init[:3, :3] = r
    T_init[:3, 3] = translation

    if confidence > 95:
        print("POSE FOUND")
        print(T_init)
        publish_pose(T_init, covariance, node, pose_publisher)
    else:
        print("confidence too low please retry in a new location")
        print(T_init)
    rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()
