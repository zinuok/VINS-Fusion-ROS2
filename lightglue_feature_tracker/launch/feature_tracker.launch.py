from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():

    config_pkg_path = get_package_share_directory('config_pkg')
    LogInfo(msg=['[feature tracker launch] Obtaining config package path: ', config_pkg_path])

    config_path = PathJoinSubstitution([
        config_pkg_path,
        'config/euroc/euroc_config.yaml'
    ])
    LogInfo(msg=['[feature tracker launch] Config path: ', config_path])

    vins_path = PathJoinSubstitution([
        config_pkg_path,
        'config/../'
    ])
    LogInfo(msg=['[feature tracker launch] VINS path: ', vins_path])

    support_path = PathJoinSubstitution([
        config_pkg_path,
        'support_files'
    ])
    LogInfo(msg=['[feature tracker launch] Support path: ', support_path])

    # Define the feature tracker node
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
            get_package_share_directory('lightglue_feature_tracker'),
            'config/euroc.yaml'
        ]),
        }]
    )
    LogInfo(msg=['[feature tracker launch] Feature tracker node defined.'])

    # Chemin vers le fichier RViz
    rviz_config_path = PathJoinSubstitution([
        config_pkg_path,
        'config/vins_euroc_rviz.rviz'
    ])
    LogInfo(msg=['[feature tracker launch] RViz config path: ', rviz_config_path])

    # # Define the vins estimator node
    # vins_estimator_node = Node(
    #     name='vins_estimator',
    #     package='vins_estimator',
    #     executable='vins_estimator',
    #     namespace='vins_estimator',
    #     output='screen',
    #     parameters=[{
    #         'config_file': config_path,
    #         'vins_folder': vins_path,
    #         'cam0_config_file': PathJoinSubstitution([
    #         get_package_share_directory('lightglue_feature_tracker'),
    #         'config/euroc.yaml'
    #         ])
    #     }]
    # )
    # LogInfo(msg=['[feature tracker launch] VINS estimator node defined.'])

    # Define the pose graph node
    # pose_graph_node = Node(
    #     name='pose_graph',
    #     package='pose_graph',
    #     executable='pose_graph',
    #     namespace='pose_graph',
    #     output='screen',
    #     parameters=[{
    #         'config_file': config_path,
    #         'support_file': support_path,
    #         'visualization_shift_x': 0,
    #         'visualization_shift_y': 0,
    #         'skip_cnt': 0,
    #         'skip_dis': 0.0
    #     }]
    # )
    # LogInfo(msg=['[feature tracker launch] Pose graph node defined.'])

    # Define the RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    LogInfo(msg=['[feature tracker launch] RViz node defined.'])

    return LaunchDescription([
        LogInfo(msg=['[feature tracker launch] Launching nodes...']),
        LogInfo(msg=['[feature tracker launch] Config path: ', config_path]),
        # vins_estimator_node,
        # pose_graph_node,
        rviz_node,
        feature_tracker_node
    ])