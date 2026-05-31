#!/usr/bin/env python3
"""
zed_cameras.launch.py — IGVC AutoNav (v3.2.2)

Brings up all three ZED X cameras with EXPLICIT serial-to-namespace binding.

Why this matters: the ZED ROS 2 wrapper, by default, opens whichever camera
the OS happened to enumerate first. If a team member unplugs and replugs a
USB cable in a different order, the namespace assignments silently swap and
the BEV node ends up using (e.g.) the right camera's mount yaw to project
the actual-left camera's images. The result is a stitched BEV that looks
'almost right' but is geometrically wrong.

Hard-pinning the serial here means a wrong cable produces a loud failure
('camera S/N XXX not found') instead of silent garbage data.

Camera serial mapping (matches bev_config.yaml and the team's IGVC_ROS2
stack — verified per-port 2026-04-24):
  Left   : ZED X (S/N 43779087)  ->  namespace /zed_left/zed_node/...
  Front  : ZED X (S/N 42569280)  ->  namespace /zed_front/zed_node/...
  Right  : ZED X (S/N 49910017)  ->  namespace /zed_right/zed_node/...

Usage:
  ros2 launch avl_bev_perception zed_cameras.launch.py
  ros2 launch avl_bev_perception zed_cameras.launch.py resolution:=HD720

Then in another terminal:
  ros2 launch avl_bev_perception bev_perception.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# Authoritative mapping. Edit ONLY if you physically swap cameras between
# mount positions. Do NOT edit this to "fix" a missing camera at runtime —
# fix the cable instead.
#
# v3.2.2 (2026-05-13): aligned with the team's IGVC_ROS2 stack at
# references/parsa_igvc/src/avros_bringup/launch/sensors.launch.py, which
# notes the mapping was "Verified 2026-04-24 via per-port enumeration."
# Pre-v3.2.2 this file had left/right swapped — see the docs sketch +
# physical mount positions if you need to re-verify.
CAMERA_BINDINGS = [
    # (namespace, serial_number, friendly_name)
    ('zed_left',  43779087, 'left'),
    ('zed_front', 42569280, 'front'),
    ('zed_right', 49910017, 'right'),
]

# ZED X camera model string used by the wrapper. Must be 'zedx' for ZED X
# units; 'zed2i' / 'zed' / etc. for other models.
CAMERA_MODEL = 'zedx'


def generate_launch_description():
    resolution = LaunchConfiguration('resolution')
    fps        = LaunchConfiguration('fps')

    actions = [
        DeclareLaunchArgument(
            'resolution', default_value='HD1080',
            description='ZED resolution (HD2K / HD1080 / HD720 / VGA). '
                        'HD720 recommended for full 30 FPS on 3 cameras.'),
        DeclareLaunchArgument(
            'fps', default_value='15',
            description='ZED frame rate. 15 is conservative for 3-camera '
                        'sustained operation on AGX Orin.'),
    ]

    # Find the official zed_wrapper launch file. Requires the zed-ros2-wrapper
    # package to be installed:  https://github.com/stereolabs/zed-ros2-wrapper
    zed_wrapper_launch = PathJoinSubstitution([
        FindPackageShare('zed_wrapper'),
        'launch',
        'zed_camera.launch.py',
    ])

    for namespace, serial, friendly in CAMERA_BINDINGS:
        actions.append(GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(zed_wrapper_launch),
                launch_arguments={
                    # Pin the serial — this is the whole point of this file.
                    'serial_number': str(serial),
                    # Distinct namespace so each camera publishes independently.
                    'camera_name':   namespace,
                    'camera_model':  CAMERA_MODEL,
                    # Reasonable defaults for IGVC outdoor use.
                    'grab_resolution':         resolution,
                    'grab_frame_rate':         fps,
                    'publish_tf':              'false',  # we publish a clean
                                                         # static TF tree from
                                                         # tf_static.launch.py
                    'publish_map_tf':          'false',
                    'pos_tracking_enabled':    'false',  # planner owns localization
                    'mapping.mapping_enabled': 'false',  # we only need RGB+depth+info
                    'object_detection_enabled':'false',  # tier 2 lives in our node
                }.items(),
            ),
        ]))

    return LaunchDescription(actions)
