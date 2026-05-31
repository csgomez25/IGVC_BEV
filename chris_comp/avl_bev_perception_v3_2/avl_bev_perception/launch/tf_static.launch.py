#!/usr/bin/env python3
"""
tf_static.launch.py — IGVC AutoNav (v3.2.1)

Publishes the full static TF tree for the AVL bot. All other packages
(LiDAR, planner, RViz) consume this — the BEV perception node uses its
YAML mount poses directly and does NOT depend on this tree, but having
one consistent source of truth across the stack avoids drift between
hand-edited config files.

Frame conventions (REP-103):
  X = forward, Y = left, Z = up

Origin: base_link sits at the IMU-based footprint (0, 0, 0) per the
team measurement sketch. Sensor poses are converted from the sketch's
(X=right, Y=forward, Z=up) inch coordinates to REP-103 meters.

Conversion:
  X_rep = Y_sketch * 0.0254
  Y_rep = -X_sketch * 0.0254
  Z_rep = Z_sketch * 0.0254

Topology:
  base_link
    |-- imu_link            (Xsens MTi-680G body)
    |-- gps_link            (Xsens internal GPS antenna)
    |-- velodyne            (VLP-16 mount)
    |-- zed_left_camera_center
    |-- zed_front_camera_center
    `-- zed_right_camera_center
"""

from launch import LaunchDescription
from launch_ros.actions import Node


# Convenience: inches -> meters
def IN(x):
    return float(x) * 0.0254


def static_tf(name, parent, child, x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
    """Build a static_transform_publisher Node. Args are floats."""
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=[
            '--x', str(x), '--y', str(y), '--z', str(z),
            '--yaw',   str(yaw),
            '--pitch', str(pitch),
            '--roll',  str(roll),
            '--frame-id',       parent,
            '--child-frame-id', child,
        ],
    )


def generate_launch_description():
    return LaunchDescription([

        # ---- IMU (Xsens MTi-680G body) ----
        # Sketch IMU(0, 0, 19.5) in. Centered, 19.5 in above footprint.
        static_tf('tf_imu',
                  parent='base_link', child='imu_link',
                  x=IN(0), y=IN(0), z=IN(19.5)),

        # ---- GPS antenna (internal to Xsens MTi-680G) ----
        # Sketch GPS(0, 26, -1) in. NOTE: Z = -1 in the sketch puts the
        # antenna 1 inch BELOW the ground footprint, which is physically
        # impossible. The team confirmed -1 is correct as drawn; treating
        # this as 'just below the IMU baseplate' relative to ground level.
        # If the GPS is actually mounted higher (e.g. on the GPS fin), update
        # Z here AND remeasure — antenna height affects RTK fix accuracy.
        static_tf('tf_gps',
                  parent='base_link', child='gps_link',
                  x=IN(26), y=IN(0), z=IN(-1)),

        # ---- Velodyne VLP-16 ----
        # Sketch Lidar(0, 4, 25.5) in.
        static_tf('tf_velodyne',
                  parent='base_link', child='velodyne',
                  x=IN(4), y=IN(0), z=IN(25.5)),

        # ---- ZED X Front (S/N 42569280) ----
        # Sketch F_camera(0, 16, 19.5) in. yaw=0 (looking straight ahead).
        static_tf('tf_zed_front',
                  parent='base_link', child='zed_front_camera_center',
                  x=IN(16), y=IN(0), z=IN(19.5),
                  yaw=0.0),

        # ---- ZED X Left (S/N 43779087) ----
        # Sketch L_camera(-11.5, 4, 19.5) in. yaw=+90deg, looking due left.
        # Team measured ~exactly 90deg; if a future calibration shows a slight
        # forward toe-in (e.g. 75deg) update yaw here AND in bev_config.yaml.
        static_tf('tf_zed_left',
                  parent='base_link', child='zed_left_camera_center',
                  x=IN(4), y=IN(11.5), z=IN(19.5),
                  yaw=1.5708),

        # ---- ZED X Right (S/N 49910017) ----
        # Sketch R_camera(11.5, 4, 19.5) in. yaw=-90deg, looking due right.
        static_tf('tf_zed_right',
                  parent='base_link', child='zed_right_camera_center',
                  x=IN(4), y=IN(-11.5), z=IN(19.5),
                  yaw=-1.5708),
    ])
