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
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.actions import SetParameter
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    localization = LaunchConfiguration('localization')

    parameters={
          'frame_id':'base_link',
          'qos': 2,
          'odom_frame_id':'odom',   
          'publish_tf':True,
          'odom_tf_linear_variance':0.001,
          'odom_tf_angular_variance':0.001,
          'subscribe_rgbd':False,
          'subscribe_rgb':False,
          'subscribe_depth': False,
          'subscribe_scan':True, 
          'subscribe_odom_info':False,
          'approx_sync':False,
          'sync_queue_size': 10,
          # RTAB-Map's internal parameters should be strings
          'RGBD/NeighborLinkRefining': 'true',    # Do odometry correction with consecutive laser scans
          'RGBD/ProximityBySpace':     'true',    # Local loop closure detection (using estimated position) with locations in WM
        #  'RGBD/ProximityByTime':      'false',   # Local loop closure detection with locations in STM
          'RGBD/ProximityPathMaxNeighbors': '10', # Do also proximity detection by space by merging close scans together.
          'RGBD/CreateOccupancyGrid': 'true',
          'Grid/PublishMap':          'true',     # Publish occupancy grid map
          'Grid/RangeMin':        '0.1',      
          'Grid/RangeMax':        '10.0',     
          'Grid/CellSize':        '0.025',     
       #   'Reg/Strategy':              '1',       # 0=Visual, 1=ICP, 2=Visual+ICP
       #   'Vis/MinInliers':            '12',      # 3D visual words minimum inliers to accept loop closure
          'RGBD/OptimizeFromGraphEnd': 'false',   # Optimize graph from initial node so /map -> /odom transform will be generated
          'RGBD/OptimizeMaxError':     '4',       # Reject any loop closure causing large errors (>3x link's covariance) in the map
          'Reg/Force3DoF':             'true',    # 2D SLAM
          'Grid/FromDepth':            'false',   # Create 2D occupancy grid from laser scan
          'Mem/STMSize':               '30',      # increased to 30 to avoid adding too many loop closures on just seen locations
          'RGBD/LocalRadius':          '5',       # limit length of proximity detections
          'Icp/CorrespondenceRatio':   '0.2',     # minimum scan overlap to accept loop closure
          'Icp/PM':                    'false',
          'Icp/PointToPlane':          'false',
          'Icp/MaxCorrespondenceDistance': '0.15',
          'Icp/VoxelSize':             '0.05',
          'database_path' : LaunchConfiguration('database_path'),
    }
    
    remappings=[
         ('scan', '/scan_out'), ('odom', '/odometry')]
    
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
              {'Mem/IncrementalMemory':'False',
               'Mem/InitWMWithAllNodes':'True'}],
            remappings=remappings),

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
                'min_height': 0.1,
                'max_height': 1.5,
                'angle_min': -3.14,
                'angle_max': 3.14,
                'angle_increment': 0.0087,
                'scan_time': 0.1,
                'range_min': 0.2,
                'range_max': 10.0,
                'use_inf': True
            }]
        ),
        Node(
        package='rtabmap',         
        condition=IfCondition(LaunchConfiguration("cam_saver")),
        executable='camera_lidar_saver_voyager',   
        name='camera_lidar_saver',
        output='screen',),
        
        Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=['/home/alex/ros2_ws/src/rtabmap/params/test.yaml'])
        
    ])