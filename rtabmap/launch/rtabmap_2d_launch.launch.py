# Requirements:
#   Download rosbag:
#    * demo_mapping.db3: https://drive.google.com/file/d/1v9qJ2U7GlYhqBJr7OQHWbDSCfgiVaLWb/view?usp=drive_link
#
# Example:
#
#   SLAM:
#     $ ros2 launch rtabmap_demos robot_mapping_demo.launch.py rviz:=true rtabmap_viz:=true
#
#   Rosbag:
#     $ ros2 bag play demo_mapping.db3 --clock
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.actions import SetParameter
import os
from ament_index_python.packages import get_package_share_directory
import math

def generate_launch_description():

    localization = LaunchConfiguration('localization')

    # parameters={
    #       'frame_id':'base_link',
    #       'qos': 2,
    #       'odom_frame_id':'odom',
    #       'odom_tf_linear_variance':0.001,
    #       'odom_tf_angular_variance':0.001,
    #       'subscribe_rgbd':False,
    #       'subscribe_rgb':False,
    #       'subscribe_depth': False,
    #       'subscribe_scan':True, 
    #       'subscribe_odom_info':False,
    #       'approx_sync':False,
    #       'sync_queue_size': 10,
    #       # RTAB-Map's internal parameters should be strings
    #       'RGBD/NeighborLinkRefining': 'true',    # Do odometry correction with consecutive laser scans
    #       'RGBD/ProximityBySpace':     'true',    # Local loop closure detection (using estimated position) with locations in WM
    #       'RGBD/ProximityByTime':      'false',   # Local loop closure detection with locations in STM
    #       'RGBD/ProximityPathMaxNeighbors': '10', # Do also proximity detection by space by merging close scans together.
    #       'RGBD/CreateOccupancyGrid': 'true',
    #       'Grid/PublishMap':          'true',     # Publish occupancy grid map
    #       'Reg/Strategy':              '1',       # 0=Visual, 1=ICP, 2=Visual+ICP
    #       'Vis/MinInliers':            '12',      # 3D visual words minimum inliers to accept loop closure
    #       'RGBD/OptimizeFromGraphEnd': 'false',   # Optimize graph from initial node so /map -> /odom transform will be generated
    #       'RGBD/OptimizeMaxError':     '4',       # Reject any loop closure causing large errors (>3x link's covariance) in the map
    #       'Reg/Force3DoF':             'true',    # 2D SLAM
    #       'Grid/FromDepth':            'false',   # Create 2D occupancy grid from laser scan
    #       'Mem/STMSize':               '30',      # increased to 30 to avoid adding too many loop closures on just seen locations
    #       'RGBD/LocalRadius':          '5',       # limit length of proximity detections
    #       'Icp/CorrespondenceRatio':   '0.2',     # minimum scan overlap to accept loop closure
    #       'Icp/PM':                    'false',
    #       'Icp/PointToPlane':          'false',
    #       'Icp/MaxCorrespondenceDistance': '0.15',
    #       'Icp/VoxelSize':             '0.05',
    #       'database_path' : LaunchConfiguration('database_path'),
    # }
#     parameters={
#           'frame_id':'base_link',
#           'qos': 2,
#           'odom_frame_id':'odom',   
#           'publish_tf':True,
#           'odom_tf_linear_variance':0.001,
#           'odom_tf_angular_variance':0.001,
#           'subscribe_rgbd':False,
#           'subscribe_rgb':False,
#           'subscribe_depth': False,
#           'subscribe_scan':True, 
#           'subscribe_odom_info':False,
#           'approx_sync':False,
#           'sync_queue_size': 10,
#           # RTAB-Map's internal parameters should be strings
#           'RGBD/NeighborLinkRefining': 'true',    # Do odometry correction with consecutive laser scans
#       #    'RGBD/ProximityBySpace':     'true',    # Local loop closure detection (using estimated position) with locations in WM
#         #  'RGBD/ProximityByTime':      'false',   # Local loop closure detection with locations in STM
#      #     'RGBD/ProximityPathMaxNeighbors': '0', # Do also proximity detection by space by merging close scans together.
#           'RGBD/CreateOccupancyGrid': 'true',
#         #   'RGBD/LinearUpdate':         '0.05',     # Update grid every 5cm
#         #   'RGBD/AngularUpdate':        '0.05',      # Update grid every
#           'Grid/PublishMap':          'true',     # Publish occupancy grid map
#           'Grid/RangeMin':        '0.1',      
#           'Grid/RangeMax':        '10.0',     
#           'Grid/CellSize':        '0.025',     
#        #   'Reg/Strategy':              '1',       # 0=Visual, 1=ICP, 2=Visual+ICP
#        #   'Vis/MinInliers':            '12',      # 3D visual words minimum inliers to accept loop closure
#           'RGBD/OptimizeFromGraphEnd': 'false',   # Optimize graph from initial node so /map -> /odom transform will be generated
#        #   'RGBD/OptimizeMaxError':     '4',       # Reject any loop closure causing large errors (>3x link's covariance) in the map
#           'Reg/Force3DoF':             'true',    # 2D SLAM
#           'Grid/FromDepth':            'false',   # Create 2D occupancy grid from laser scan
#     #      'Mem/STMSize':               '30',      # increased to 30 to avoid adding too many loop closures on just seen locations
#       #    'RGBD/LocalRadius':          '5',       # limit length of proximity detections
#           'Icp/CorrespondenceRatio':   '0.1',     # minimum scan overlap to accept loop closure
#      #    'Icp/PM':                    'false',
#    #       'Icp/PointToPlane':          'false',
#           'Icp/MaxCorrespondenceDistance': '0.15',
#           'Icp/VoxelSize':             '0.0',
#           'database_path' : LaunchConfiguration('database_path'),
#           'Rtabmap/DetectionRate': '5',
#     }
    
    # parameters={
    #       'frame_id':'base_link',
    #       'qos': 2,
    #       'odom_frame_id':'odom',   
    #       'publish_tf':True,
    #       'odom_tf_linear_variance':0.001,
    #       'odom_tf_angular_variance':0.001,
    #       'subscribe_rgbd':False,
    #       'subscribe_rgb':False,
    #       'subscribe_depth': False,
    #       'subscribe_scan':True, 
    #       'subscribe_odom_info':False,
    #       'approx_sync':False,
    #       'sync_queue_size': 10,
    #       # RTAB-Map's internal parameters should be strings
    #       'RGBD/NeighborLinkRefining': 'true',    # Do odometry correction with consecutive laser scans
    #     #  'RGBD/ProximityBySpace':     'true',    # Local loop closure detection (using estimated position) with locations in WM
    #     #  'RGBD/ProximityByTime':      'false',   # Local loop closure detection with locations in STM
    # #      'RGBD/ProximityPathMaxNeighbors': '10', # Do also proximity detection by space by merging close scans together.
    #       'RGBD/CreateOccupancyGrid':  'true',
    #   #    'RGBD/LinearUpdate':         '0.05',     # Update grid every 5cm
    #    #   'RGBD/AngularUpdate':        '0.05',      # Update grid every
    #       'Grid/PublishMap':           'true',     # Publish occupancy grid map
    #       'Grid/RangeMin':             '0.1',      
    #       'Grid/RangeMax':             '10.0',     
    #       'Grid/CellSize':             '0.025',     
    #  #     'Grid/MinClusterSize':       '10',      # minimum number of cells to form a cluster
    #       'Grid/FromDepth':            'false',   # Create 2D occupancy grid from laser scan
    #    #   'Reg/Strategy':              '1',       # 0=Visual, 1=ICP, 2=Visual+ICP
    #       'RGBD/OptimizeFromGraphEnd': 'false',   # Optimize graph from initial node so /map -> /odom transform will be generated
    #  #     'RGBD/OptimizeMaxError':     '3',       # Reject any loop closure causing large errors (>3x link's covariance) in the map
    #       'Reg/Force3DoF':             'true',    # 2D SLAM
    #     #  'Mem/STMSize':               '30',      # increased to 30 to avoid adding too many loop closures on just seen locations
    #   #    'RGBD/LocalRadius':          '5',       # limit length of proximity detections    
    # #     'Icp/CorrespondenceRatio':   '0.5',     # minimum scan overlap to accept loop closure
    #   #   'Icp/PM':                    'false',
    #   #   'Icp/PointToPlane':          'false',
    #       'Icp/MaxCorrespondenceDistance': '0.1',
    #       'Icp/VoxelSize':             '0.0',
    #       'database_path' : LaunchConfiguration('database_path'),
    #       'Rtabmap/CreateIntermediateNodes': 'true',  # create intermediate nodes in graph when odometry is lost
    #       'Rtabmap/DetectionRate': '5',  
    # }
    
    parameters = { # ---- Basics / I/O ---- 
    'frame_id': 'base_link',
    'odom_frame_id': 'odom',
    'publish_tf': True,
    'qos': 2,
    'subscribe_rgbd': False,
    'subscribe_rgb': False, 
    'subscribe_depth': False,
    'subscribe_scan': True,
    'subscribe_odom_info': False,
    'approx_sync': False, 
    'sync_queue_size': 10, 
    'odom_tf_linear_variance': 0.001,
    'odom_tf_angular_variance': 0.001,
    'database_path': LaunchConfiguration('database_path'),
# ---- Registration: ICP in 2D for odometry / neighbor links ---- 
    'Reg/Strategy': '1',  # 1 = ICP 
    'Reg/Force3DoF': 'true', # planar # Moderately permissive so neighbors + LC can succeed 
    'Icp/MaxCorrespondenceDistance': '0.1', 
    'Icp/CorrespondenceRatio': '0.6',
    'Icp/MaxIterations': '40', 
    'Icp/Epsilon': '0.0015',
    'Icp/VoxelSize': '0.0', 
    'Icp/PointToPlane': 'false', # ---- Keyframe cadence ----
    'Icp/MaxTranslation': '1.5',  # accept if within 0.6 m 
    'Icp/MaxRotation': '0.52', # ~30 deg 
    'Rtabmap/DetectionRate': '5', # denser nodes => more LC chances 
    'RGBD/LinearUpdate': '0.0', # don’t gate on motion (avoid stalls)
    'RGBD/AngularUpdate': '0.0', # ---- Local loop closures (proximity) ---- 
    'RGBD/ProximityBySpace': 'true',  # spatial proximity 
    'RGBD/ProximityByTime': 'true', # recent nodes in STM 
    'RGBD/LocalRadius': '6',  # meters (search window) 
    'RGBD/ProximityMaxGraphDepth': '0', # 0 = search whole WM 
    'RGBD/ProximityPathMaxNeighbors': '12',
    'RGBD/NeighborLinkRefining': 'true',
    # ---- Loop-closure validation using ICP (separate “LccIcp/*”) ---- # These govern how candidate LCs are checked & accepted. 
    'LccIcp/Type': '1', # point-to-point ICP
    'LccIcp/MaxTranslation': '1.5',  # accept if within 0.6 m 
    'LccIcp/MaxRotation': '0.52', # ~30 deg 
    'LccIcp/CorrespondenceRatio': '0.45',# ≥40% inliers to accept LC 
    'LccIcp/MaxCorrespondenceDistance':'0.15', # slightly looser than odom ICP
    'LccIcp/Iterations': '50', 
    'LccIcp/Epsilon': '0.0015', 
    'LccIcp/VoxelSize': '0.01', 
    # ---- Graph optimization ----
    'Optimizer/Strategy': '1', # g2o 
    'Optimizer/Iterations':'25', 
    'Optimizer/Epsilon': '0.0001', 
    'RGBD/OptimizeFromGraphEnd': 'false', # stable map->odom 
    'RGBD/OptimizeMaxError': '3', # a bit lenient so LCs stick 
    # ---- Occupancy grid ---- 
    'RGBD/CreateOccupancyGrid': 'true',
    'Grid/PublishMap': 'true', 
    'Grid/FromDepth': 'false', 
    'Grid/CellSize': '0.025', 
    'Grid/RangeMin': '0.1', 
    'Grid/RangeMax': '10.0', 
    'Grid/MinClusterSize': '5', 
    # ---- Memory / resiliency ---- 
    'Mem/STMSize': '60', # keep more recent nodes in STM -> more LC-by-time hits 
    'Rtabmap/CreateIntermediateNodes': 'true', 
    'Rtabmap/StartNewMapOnLoopClosure': 'false', 
    }
    
    # parameters.update({
    # # --- ICP for neighbor/odom links (stability first) ---
    # 'Icp/VoxelSize': '0.05',                 # downsample → less noise, faster/more stable ICP
    # 'Icp/MaxCorrespondenceDistance': '0.12', # was 0.7 (very loose). 0.10–0.15 is typical indoors
    # 'Icp/CorrespondenceRatio': '0.35',       # was 0.6 (too strict with real noise). 0.30–0.40 balances precision/recall
    # 'Icp/MaxIterations': '30',               # was 40; avoids overfitting/overrun
    # 'Icp/Epsilon': '0.0005',                 # was 0.0015; tighter convergence

    # # --- Keyframe cadence (reduce duplicates, increase overlap per node) ---
    # 'Rtabmap/DetectionRate': '3',            # was 10
    # 'RGBD/LinearUpdate': '0.05',             # was 0.0
    # 'RGBD/AngularUpdate': '0.05',            # was 0.0
    # 'Rtabmap/CreateIntermediateNodes': 'false',  # was true; fewer weak pose-only nodes

    # # --- Proximity (candidate generation; keep it meaningful, not global) ---
    # 'RGBD/ProximityBySpace': 'true',         # keep
    # 'RGBD/ProximityByTime': 'false',         # was true; time-based often noisy on bags
    # 'RGBD/LocalRadius': '5',                 # was 6; tune 4–6
    # 'RGBD/ProximityMaxGraphDepth': '60',     # was 0 (whole WM). Bound the search
    # 'RGBD/ProximityPathMaxNeighbors': '12',  # keep

    # # --- Loop-closure ICP validator (precision gate) ---
    # 'LccIcp/VoxelSize': '0.05',              # was 0.01; match odom ICP
    # 'LccIcp/MaxCorrespondenceDistance': '0.12',  # was 0.8; too loose
    # 'LccIcp/CorrespondenceRatio': '0.40',    # was 0.45; still strict but realistic
    # 'LccIcp/MaxTranslation': '0.25',         # was 0.5; reject big jumps
    # 'LccIcp/MaxRotation': '0.35',            # was 0.52 rad (~20° vs 30°)
    # 'LccIcp/Iterations': '40',               # was 50
    # 'LccIcp/Epsilon': '0.0005',              # was 0.0015

    # # --- Graph optimization (outlier guard) ---
    # 'RGBD/OptimizeMaxError': '2',            # was 3; stricter post-optimization rejection

    # # --- Map cleanliness (optional but helps downstream) ---
    # 'Grid/MinClusterSize': '10',             # was 5; remove speckle in occupancy map

    # # --- Memory window (optional) ---
    # 'Mem/STMSize': '30',                     # was 60; smaller STM reduces near-duplicate LC attempts
    # })
#     parameters.update({
#   #  'RGBD/ProximityAngMax': '0.79',
#     'Icp/CorrespondenceRatio': '0.30',
#    # 'Icp/MaxIterations': '30',
#     'LccIcp/CorrespondenceRatio': '0.30',


#     })

    # parameters.update({
    # 'RGBD/ProximityBySpace': 'true',
    # 'RGBD/LocalRadius': '6',
    # 'RGBD/ProximityPathMaxNeighbors': '20',
    # 'RGBD/ProximityMaxGraphDepth': '100',
    # 'RGBD/ProximityAngMax': '0.79',
    # 'Rtabmap/DetectionRate': '3',
    # 'RGBD/LinearUpdate': '0.05',
    # 'RGBD/AngularUpdate': '0.05',
    # 'Icp/VoxelSize': '0.05',
    # 'Icp/MaxCorrespondenceDistance': '0.12',
    # 'Icp/CorrespondenceRatio': '0.30',
    # 'Icp/MaxIterations': '30',
    # 'Icp/Epsilon': '0.0005',
    # 'LccIcp/VoxelSize': '0.05',
    # 'LccIcp/MaxCorrespondenceDistance': '0.15',
    # 'LccIcp/CorrespondenceRatio': '0.30',
    # 'LccIcp/MaxTranslation': '0.30',
    # 'LccIcp/MaxRotation': '0.52',
    # 'LccIcp/Iterations': '40',
    # 'LccIcp/Epsilon': '0.0005',
    # 'Rtabmap/PublishStats': 'true',
    # 'Rtabmap/PublishLikelihood': 'true',
    # })
    
    remappings=[
         ('scan', '/scan_out'), ('odom', '/odom')]
    
    config_rviz = os.path.join(
        get_package_share_directory('rtabmap'), 'config', 'demo_robot_mapping.rviz'
    )

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument('rtabmap_viz',  default_value='true',  description='Launch RTAB-Map UI (optional).'),
        DeclareLaunchArgument('rviz',         default_value='false', description='Launch RVIZ (optional).'),
        DeclareLaunchArgument('localization', default_value='false', description='Launch in localization mode.'),
        DeclareLaunchArgument('rviz_cfg', default_value=config_rviz,  description='Configuration path of rviz2.'),
        DeclareLaunchArgument('database_path', default_value='',description='Path to the RTAB-Map database to load or save'),
        DeclareLaunchArgument('cam_saver', default_value='false',description='camera lidar saver'),
      #  DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock if true'),

        SetParameter(name='use_sim_time', value=True),

        # # Nodes to launch
        # Node(
        #     package='rtabmap_sync', executable='rgbd_sync', output='screen',
        #     parameters=[parameters,
        #       {'rgb_image_transport':'compressed',
        #        'depth_image_transport':'compressedDepth',
        #        'approx_sync_max_interval': 0.02}],
        #     remappings=remappings),
        
        # SLAM mode:
        Node(
            condition=UnlessCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=['-d']), # This will delete the previous database (~/.ros/rtabmap.db)
            
        # Localization mode:
        Node(
            condition=IfCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters,
              {'Mem/IncrementalMemory':'false',
               'Mem/InitWMWithAllNodes':'true',
                'RGBD/StartAtOrigin':'true',
                'Icp/MaxCorrespondenceDistance': '0.3', 
                'Icp/CorrespondenceRatio': '0.2',
                'Icp/MaxIterations': '40', 
                'Icp/Epsilon': '0.0015',
                'Icp/VoxelSize': '0.0', 
                'Icp/PointToPlane': 'false', # ---- Keyframe cadence ----
                'Icp/MaxTranslation': '1.5',  # accept if within 0.6 m 
                'Icp/MaxRotation': '0.52', # ~30 deg 
                }],
            remappings=remappings),

        # Visualization:
        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            parameters=[parameters],
            remappings=remappings),
        Node(
            package='rviz2', executable='rviz2', name="rviz2", output='screen',
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=[["-d"], [LaunchConfiguration("rviz_cfg")]]),
        
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan_node',
            output='screen',
            remappings=[
                ('cloud_in', '/ouster/points'),
                ('scan', '/scan_out')
            ],
            parameters=[{
                'target_frame': 'base_link',
                'transform_tolerance': 0.01,
                'min_height': 0.05,
                'max_height': 0.7,
                'angle_min': -3.14,
                'angle_max': 3.14,
                'angle_increment': 1*math.pi/180,
                'scan_time': 0.1,
                'range_min': 0.2,
                'range_max': 10.0,
                'use_inf': True
            }]
        ),
        Node(
        package='rtabmap',    
        condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("cam_saver"), "' == 'true' and ",
                "'", LaunchConfiguration("localization"), "' == 'false'"
            ])),
        executable='camera_lidar_saver',  
        name='camera_lidar_saver',
        output='screen',),
        
        Node(
        condition=UnlessCondition(localization),
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=['/home/alex/ros2_ws/src/rtabmap/params/ekf_2d_params.yaml'],

        ),
        Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_odom_node',
        output='screen',
        parameters=['/home/alex/ros2_ws/src/rtabmap/params/ekf_2d_odom_params.yaml'],
        remappings=[('odometry/filtered', '/odometry/filtered_odom')],
        ),
        Node(
        package='rtabmap',
        executable='pub_gt_pose',
        name='pub_gt_pose',
        output='screen',
        )
        
    ])