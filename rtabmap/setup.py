from setuptools import find_packages, setup

package_name = 'rtabmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/rtabmap/launch', ['launch/rtabmap_launch.launch.py', 'launch/nav_launch.launch.py', 'launch/rtabmap_2d_launch.launch.py', 'launch/nav_2d_launch.launch.py', 'launch/esl_test.launch.py', 'launch/rtabmap_voyager_2d_launch.py']),
        ('share/' + package_name + '/params', ['params/nav2_params.yaml', 'params/nav2_2d_params.yaml', 'params/ekf_params.yaml', 'params/ekf_2d_params.yaml', 'params/test.yaml', 'params/ekf_voyager_params.yaml', 'params/ekf_voyager_odom_params.yaml', 'params/ekf_2d_odom_params.yaml']),
        ('share/' + package_name + '/config', ['config/demo_robot_mapping.rviz', 'config/nav2_rviz_config.rviz']),
        
 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='24702811@sun.ac.za',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],                                                                                               
    entry_points={
        'console_scripts': [
            'camera_lidar_saver = rtabmap.camera_lidar_saver:main',
            'camera_lidar_saver_voyager = rtabmap.camera_lidar_saver_voyager:main',
            'find_initial_pose = rtabmap.find_initial_pose:main',
            'pub_gt_pose = rtabmap.pub_gt_pose:main',
        ],
    },
)
