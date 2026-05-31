#!/usr/bin/env python3
"""
vision_test.launch.py — Standalone BEV-perception bringup (no robot needed).

Brings up ONLY the perception pieces: ZED cameras (or a bag replay), the
BEV perception node, and (optionally) RViz. No Xsens, no Velodyne, no Nav2,
no actuator, no localization — this is the "test the eyes without turning
the robot on" launch.

Two modes:

  Live cameras (default):
    ros2 launch avl_bev_perception vision_test.launch.py

  Bag replay (developing off-Jetson, no cameras attached):
    ros2 launch avl_bev_perception vision_test.launch.py \
        use_bag:=true bag_path:=/data/run_42

Arguments:
  use_bag:bool    Replay a recorded ros2 bag instead of opening live ZEDs.
                  Skips zed_cameras.launch.py entirely and enables sim time.
                  Default: false.
  bag_path:str    Path to the ros2 bag directory (required if use_bag:=true).
  bag_loop:bool   Pass --loop to `ros2 bag play` so the session repeats
                  forever. Default: true.
  bag_rate:float  Playback rate multiplier. Default: 1.0.
  use_rviz:bool   Open RViz with the bundled BEV layout. Default: true.
  perc_fps:float  /bev/* publish rate. Default: 20.0.
  viz_fps:float   /bev/image_raw etc publish rate. Default: 2.0.

Topic contract (mode-independent — the BEV node doesn't care whether the
ZED topics come from the live wrapper or `ros2 bag play`):
  Subscribes (per camera <cam>):
    /zed_<cam>/zed_node/rgb/color/rect/image
    /zed_<cam>/zed_node/rgb/color/rect/camera_info
    /zed_<cam>/zed_node/depth/depth_registered
  Publishes:
    /bev/segmentation /bev/drivable_mask /bev/obstacle_mask
    /bev/lane_lines_detected /bev/perception_latency_ms
    /bev/image_raw /bev/fused /bev/debug/<cam>   (if viz enabled)

Record a bag suitable for replay with tools/record_session.sh.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('avl_bev_perception')

    args = [
        DeclareLaunchArgument(
            'use_bag', default_value='false',
            description='Replay a ros2 bag instead of opening live ZED cameras.'),
        DeclareLaunchArgument(
            'bag_path', default_value='',
            description='Path to the ros2 bag directory (required if use_bag:=true).'),
        DeclareLaunchArgument(
            'bag_loop', default_value='true',
            description='Loop bag playback forever.'),
        DeclareLaunchArgument(
            'bag_rate', default_value='1.0',
            description='Bag playback rate multiplier.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Open RViz with the bundled BEV layout.'),
        DeclareLaunchArgument(
            'perc_fps', default_value='20.0',
            description='/bev/* perception loop rate (Hz).'),
        DeclareLaunchArgument(
            'viz_fps', default_value='2.0',
            description='/bev/image_raw etc viz loop rate (Hz).'),
    ]

    use_bag = LaunchConfiguration('use_bag')
    use_rviz = LaunchConfiguration('use_rviz')
    bag_path = LaunchConfiguration('bag_path')
    bag_loop = LaunchConfiguration('bag_loop')
    bag_rate = LaunchConfiguration('bag_rate')

    # When replaying a bag, use_sim_time MUST be true so message-filter
    # synchronization and our timers use bag-stamped clocks. With live
    # cameras, stay on wall clock.
    use_sim_time = PythonExpression(["'", use_bag, "'.lower() == 'true'"])

    # ---- Mode A: live cameras (use_bag:=false) ---------------------------
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            pkg_share, '/launch/zed_cameras.launch.py',
        ]),
        condition=UnlessCondition(use_bag),
    )

    # ---- Mode B: bag replay (use_bag:=true) ------------------------------
    # `ros2 bag play` reads /tf and /tf_static along with the topics and
    # publishes them under sim-time. We rely on the bag containing the
    # full ZED topic set (see tools/record_session.sh for what to capture).
    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            bag_path,
            '--clock',
            '--rate', bag_rate,
            PythonExpression(["'--loop' if '", bag_loop, "'.lower()=='true' else ''"]),
        ],
        output='screen',
        condition=IfCondition(use_bag),
    )

    bag_warn = LogInfo(
        msg=['Bag replay mode enabled. Playing: ', bag_path,
             ' (loop=', bag_loop, ' rate=', bag_rate, ').'],
        condition=IfCondition(use_bag),
    )

    # ---- BEV perception node (mode-independent) --------------------------
    bev_node = Node(
        package='avl_bev_perception',
        executable='bev_perception_node',
        name='bev_perception_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'bev_config.yaml']),
            {
                'use_sim_time': use_sim_time,
                'perception.fps': LaunchConfiguration('perc_fps'),
                'viz.fps':        LaunchConfiguration('viz_fps'),
            },
        ],
    )

    # ---- RViz ------------------------------------------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_vision_test',
        output='screen',
        arguments=[
            '-d', PathJoinSubstitution([pkg_share, 'rviz', 'bev_perception.rviz']),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(args + [zed_launch, bag_warn, bag_play, bev_node, rviz])
