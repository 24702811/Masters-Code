"""Launch Nav2, RViz2, and RTAB-Map"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_rtabmap = get_package_share_directory('rtabmap')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    # File paths
    nav2_params_file = os.path.join(pkg_rtabmap, 'params', 'nav2_params.yaml')
    rviz_config_file = os.path.join(pkg_nav2_bringup, 'rviz', 'nav2_default_view.rviz')
    rtabmap_launch_file = os.path.join(pkg_rtabmap, 'launch', 'rtabmap_launch.launch.py')

    # Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': nav2_params_file,
            'use_amcl': 'false',
            'map':'/home/alex/ros2_ws/mapping/house3.yaml',  # 3d mapping
           # 'map':'/home/alex/ros2_ws/maps/2d_mapping/small_house_2d_mapping.yaml',  # 2d mapping
        }.items()
    )

    # RTAB-Map SLAM
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_launch_file),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'localization': 'true',
            'database_path': '/home/alex/ros2_ws/mapping/dbs/small_house.db', # 3d mapping
          #  'database_path': '/home/alex/ros2_ws/maps/2d_mapping/small_house_2d_mapping.db',   # 2d mapping
            'frame_id': 'base_link',
            'lidar_topic': '/ouster/points',
            'imu_topic': '/imu/data',
          #  'publish_tf_map': 'true',
        }.items()
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan_node',
        output='screen',
        remappings=[
            ('cloud_in', '/ouster/points'),
            ('scan', '/ouster/scan')
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
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        nav2_bringup,
        rtabmap,
        rviz_node, 
        pointcloud_to_laserscan_node
    ])
