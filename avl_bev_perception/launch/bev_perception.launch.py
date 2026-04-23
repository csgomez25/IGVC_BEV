"""
AVL BEV Perception (v3) — IGVC AutoNav launch
=============================================
Starts the BEV perception node only. Bring up the 3 ZED X cameras with
your existing ZED launch first.

Usage:
  ros2 launch avl_bev_perception bev_perception.launch.py
  ros2 launch avl_bev_perception bev_perception.launch.py viz_enabled:=false
  ros2 launch avl_bev_perception bev_perception.launch.py seg_enabled:=false
  ros2 launch avl_bev_perception bev_perception.launch.py use_rviz:=true

Arguments:
  seg_enabled  : Enable HSV+ONNX segmentation               (default: true)
  viz_enabled  : Publish BGR images for RViz/rqt            (default: true)
                 Set false for race runs to save CPU/network.
  perc_fps     : Perception loop rate (Hz)                  (default: 20)
  viz_fps      : Visualization publish rate (Hz)            (default: 2)
  use_rviz     : Open RViz with the bundled BEV layout      (default: false)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('avl_bev_perception')

    args = [
        DeclareLaunchArgument('seg_enabled', default_value='true'),
        DeclareLaunchArgument('viz_enabled', default_value='true'),
        DeclareLaunchArgument('perc_fps',    default_value='20.0'),
        DeclareLaunchArgument('viz_fps',     default_value='2.0'),
        DeclareLaunchArgument('use_rviz',    default_value='false'),
    ]

    bev_node = Node(
        package='avl_bev_perception',
        executable='bev_perception_node',
        name='bev_perception_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'bev_config.yaml']),
            {
                'segmentation.enabled': LaunchConfiguration('seg_enabled'),
                'viz.enabled':          LaunchConfiguration('viz_enabled'),
                'perception.fps':       LaunchConfiguration('perc_fps'),
                'viz.fps':              LaunchConfiguration('viz_fps'),
            },
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_bev',
        output='screen',
        arguments=[
            '-d', PathJoinSubstitution([pkg_share, 'rviz', 'bev_perception.rviz'])
        ],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(args + [bev_node, rviz])
