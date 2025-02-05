from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    config_pkg_path = get_package_share_directory('config_pkg')
    lightglue_pkg_path = get_package_share_directory('lightglue_feature_tracker')
    loop_fusion_pkg_path = get_package_share_directory('loop_fusion')
    
    config_path = PathJoinSubstitution([
        config_pkg_path, 'config/euroc/euroc_config.yaml'
    ])
    vins_path = PathJoinSubstitution([
        config_pkg_path, 'config/../'
    ])
    support_path = PathJoinSubstitution([
        config_pkg_path, 'support_files'
    ])
    
    feature_tracker_node = Node(
        package='lightglue_feature_tracker',
        executable='feature_tracker_node',
        name='lightglue_feature_tracker',
        namespace='feature_tracker',
        output='screen',
        parameters=[{
            'config_file': config_path,
            'vins_folder': vins_path,
            'cam_config_file': PathJoinSubstitution([
                lightglue_pkg_path, 'config/euroc.yaml'
            ]),
        }]
    )
    rviz_config_path = PathJoinSubstitution([
        lightglue_pkg_path, 'config/LightGlueRViz2.rviz'
    ])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    return LaunchDescription([
        LogInfo(msg=['[launch] Launching feature tracker and loop fusion...']),
        feature_tracker_node,
        # pose_graph_node,
        rviz_node
    ])
