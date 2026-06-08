#!/usr/bin/env python3
"""
ROS 2 lane-detection adapter for the ZED X cameras (Parsa's IGVC stack).

Subscribes the ZED wrapper's rectified RGB topic for one or more cameras, runs a
per-camera `LaneDetector`, and republishes an overlay + a mono8 lane mask. This
is the *visual bring-up* path — it proves the CV works on the real cameras
before any BEV/costmap wiring. All CV logic stays in the ROS-free `lane_cv`
package; this file is pure glue (rclpy + cv_bridge).

Topic paths default to the **ZED v5.x** convention used in the references
(`references/parsa_igvc/CLAUDE.md`, "ZED v5 topic names differ from v4"):

    /zed_<cam>/zed_node/rgb/color/rect/image          <- v5 (default here)
    /zed_<cam>/zed_node/rgb/image_rect_color          <- v4 (override if needed)

Camera names match the serial-pinned namespaces (left / front / right).

Run — just the front camera:
    python3 adapters/ros2_zed.py --cameras front

Run — all three:
    python3 adapters/ros2_zed.py --cameras front left right

Override the topic template (e.g. a v4 bringup):
    python3 adapters/ros2_zed.py --cameras front --ros-args \
        -p rgb_topic_template:='/zed_{cam}/zed_node/rgb/image_rect_color'

View results:
    ros2 topic hz   /lane/front/mask
    ros2 run rqt_image_view rqt_image_view /lane/front/overlay
"""

import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lane_cv import LaneConfig, LaneDetector  # noqa: E402


class LaneZedNode(Node):
    """One detector per camera; one RGB sub + overlay/mask pubs per camera."""

    def __init__(self, cameras):
        super().__init__("lane_cv_zed")
        self.bridge = CvBridge()

        # Parameters (all overridable with -p ... at launch).
        default_cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
        self.declare_parameter("config", default_cfg)
        self.declare_parameter("rgb_topic_template", "/zed_{cam}/zed_node/rgb/color/rect/image")
        self.declare_parameter("overlay_topic_template", "/lane/{cam}/overlay")
        self.declare_parameter("mask_topic_template", "/lane/{cam}/mask")
        self.declare_parameter("publish_overlay", True)

        cfg_path = self.get_parameter("config").value
        rgb_tpl = self.get_parameter("rgb_topic_template").value
        ov_tpl = self.get_parameter("overlay_topic_template").value
        mask_tpl = self.get_parameter("mask_topic_template").value
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)

        try:
            base_cfg = LaneConfig.from_yaml(cfg_path)
        except Exception as e:  # fall back to built-in defaults
            self.get_logger().warn(f"could not load {cfg_path} ({e}); using built-in defaults")
            base_cfg = LaneConfig()

        self._cams = {}
        for cam in cameras:
            # Each camera gets its own detector (independent adaptive + temporal
            # state) so one camera's exposure swing can't perturb another.
            det = LaneDetector(LaneConfig.from_dict(base_cfg.to_dict()))
            rgb_topic = rgb_tpl.format(cam=cam)
            sub = self.create_subscription(
                Image, rgb_topic,
                lambda msg, c=cam: self._on_image(c, msg),
                qos_profile_sensor_data)
            ov_pub = (self.create_publisher(Image, ov_tpl.format(cam=cam), 5)
                      if self.publish_overlay else None)
            mask_pub = self.create_publisher(Image, mask_tpl.format(cam=cam), 5)
            self._cams[cam] = dict(det=det, sub=sub, ov_pub=ov_pub,
                                   mask_pub=mask_pub, count=0)
            self.get_logger().info(f"[{cam}] subscribing {rgb_topic}")

        self.get_logger().info(
            f"lane_cv_zed up on {len(self._cams)} camera(s): {', '.join(cameras)}")

    def _on_image(self, cam, msg):
        st = self._cams[cam]
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[{cam}] cv_bridge failed: {e}", throttle_duration_sec=5.0)
            return

        result = st["det"].process(bgr)

        mask_msg = self.bridge.cv2_to_imgmsg(result.lane_mask, encoding="mono8")
        mask_msg.header = msg.header           # keep stamp + frame_id from source
        st["mask_pub"].publish(mask_msg)

        if st["ov_pub"] is not None:
            overlay = st["det"].draw_overlay(bgr, result)
            ov_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            ov_msg.header = msg.header
            st["ov_pub"].publish(ov_msg)

        st["count"] += 1
        if st["count"] % 60 == 0:
            self.get_logger().info(f"[{cam}] {st['count']} frames, "
                                   f"{len(result.segments)} segments last frame")


def main():
    ap = argparse.ArgumentParser(description="ROS 2 lane_cv adapter for ZED X.")
    ap.add_argument("--cameras", nargs="+", default=["front"],
                    choices=["front", "left", "right"],
                    help="which ZED cameras to run (1 or all 3)")
    # Let ROS args (e.g. --ros-args -p ...) pass through untouched.
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = LaneZedNode(args.cameras)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
