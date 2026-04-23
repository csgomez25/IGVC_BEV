#!/usr/bin/env python3
"""
AVL BEV Perception Node — IGVC AutoNav (v3)
===========================================
Bird's Eye View from 3 ZED X cameras, IGVC-tuned segmentation, and
decoupled perception/visualization timers for full-speed operation.

Cameras:
  Left   : ZED X (S/N 43779087)
  Front  : ZED X (S/N 42569280)
  Right  : ZED X (S/N 49910017)

Key design points:

  * Two timers. The PERCEPTION loop runs at perception_fps (default 20 Hz)
    and publishes machine-consumable outputs the planner needs every
    frame. The VIZ loop runs at viz_fps (default 2 Hz) and publishes the
    pretty RGB images for RViz / rqt. Viz can be disabled entirely.

  * Two output classes of topic:
      Machine outputs (always on, lightweight):
        /bev/segmentation       sensor_msgs/Image  mono8  class IDs
        /bev/drivable_mask      sensor_msgs/Image  mono8  255 = drivable
        /bev/obstacle_mask      sensor_msgs/Image  mono8  255 = obstacle
      Viz outputs (slow, optional):
        /bev/image_raw          sensor_msgs/Image  bgr8
        /bev/fused              sensor_msgs/Image  bgr8
        /bev/debug/<cam>        sensor_msgs/Image  bgr8

  * IGVC class set (see seg_inference.py):
      0 background  1 lane line  2 barrel  3 person  4 pothole  5 drivable

  * Pothole-friendly projection. Potholes are flat circles painted on
    asphalt, so they have no height. The standard height filter would
    discard them. We carry a separate low-Z pass for any pixel whose seg
    class is in GROUND_PLANE_CLASSES.

  * Offroad / waypoint section. When the course has no lane lines (the
    GPS-waypoint section), the planner ignores /bev/drivable_mask and
    plans straight toward the next waypoint using only /bev/obstacle_mask.
    The mode switch lives in the planner package, not here. This node
    publishes the same topics the same way regardless.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

try:
    from .seg_inference import (
        SegmentationEngine,
        CLASS_BACKGROUND, CLASS_LANE_LINE, CLASS_BARREL,
        CLASS_PERSON, CLASS_POTHOLE, CLASS_DRIVABLE,
        OBSTACLE_CLASSES,
    )
    HAS_SEG = True
except ImportError:
    HAS_SEG = False
    CLASS_BACKGROUND = 0
    CLASS_LANE_LINE  = 1
    CLASS_BARREL     = 2
    CLASS_PERSON     = 3
    CLASS_POTHOLE    = 4
    CLASS_DRIVABLE   = 5
    OBSTACLE_CLASSES = (1, 2, 3, 4)


# Classes that live on the ground plane — projected without the normal
# height-filter floor so painted lines and potholes survive.
GROUND_PLANE_CLASSES = (CLASS_LANE_LINE, CLASS_POTHOLE)


# =============================================================================
#  Per-camera state
# =============================================================================

@dataclass
class CameraState:
    name: str
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    img_w: int = 0
    img_h: int = 0
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_z: float = 0.0
    mount_yaw: float = 0.0
    min_depth: float = 0.3
    max_depth: float = 15.0
    got_rgb: bool = False
    got_depth: bool = False
    got_info: bool = False


# =============================================================================
#  Main node
# =============================================================================

class BevPerceptionNode(Node):
    def __init__(self):
        super().__init__('bev_perception_node')
        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self._declare_params()

        # Sensor topics from the ZED wrapper are BEST_EFFORT.
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.cameras: Dict[str, CameraState] = {}
        self._setup_cameras()

        self._init_bev_grid()

        self.seg_engine: Optional[SegmentationEngine] = None
        self._init_segmentation()

        # ----------- Latest perception outputs (shared across timers) ---
        # Updated by the perception loop, read by the viz loop.
        self._latest_lock = threading.Lock()
        self._latest_bev_rgb:    Optional[np.ndarray] = None
        self._latest_bev_seg_id: Optional[np.ndarray] = None  # uint8 class IDs
        self._latest_drivable:   Optional[np.ndarray] = None  # uint8 0/255
        self._latest_obstacle:   Optional[np.ndarray] = None  # uint8 0/255
        self._latest_debug:      Dict[str, np.ndarray] = {}   # per-camera RGB+seg

        # ----------- Publishers --------------------------------------
        # Machine-consumable (always on, every perception tick)
        self.pub_seg_id    = self.create_publisher(Image, '/bev/segmentation',  10)
        self.pub_drivable  = self.create_publisher(Image, '/bev/drivable_mask', 10)
        self.pub_obstacle  = self.create_publisher(Image, '/bev/obstacle_mask', 10)

        # Viz (slow, optional)
        self.viz_enabled = bool(self.get_parameter('viz.enabled').value)
        self.pub_bev_rgb:   Optional[object] = None
        self.pub_bev_fused: Optional[object] = None
        self.pub_debug: Dict[str, object] = {}
        if self.viz_enabled:
            self.pub_bev_rgb   = self.create_publisher(Image, '/bev/image_raw', 5)
            self.pub_bev_fused = self.create_publisher(Image, '/bev/fused',     5)
            if bool(self.get_parameter('viz.publish_debug').value):
                for cam_name in self.cameras:
                    self.pub_debug[cam_name] = self.create_publisher(
                        Image, f'/bev/debug/{cam_name}', 2
                    )

        # ----------- Timers ------------------------------------------
        self.perception_fps = float(self.get_parameter('perception.fps').value)
        self.downsample = max(1, int(self.get_parameter('perception.downsample_depth').value))
        self.create_timer(1.0 / self.perception_fps, self._perception_callback)

        if self.viz_enabled:
            viz_fps = float(self.get_parameter('viz.fps').value)
            self.create_timer(1.0 / viz_fps, self._viz_callback)

        # ----------- Stats --------------------------------------------
        self._perc_count = 0
        self._viz_count  = 0
        self._perc_time_ema = 0.0

        self.get_logger().info('=== AVL BEV Perception v3 (IGVC AutoNav) ===')
        self.get_logger().info(f'  Cameras       : {list(self.cameras.keys())}')
        self.get_logger().info(f'  BEV grid      : {self.bev_w}x{self.bev_h}px @ '
                               f'{self.bev_res}m/px')
        self.get_logger().info(f'  Perception    : {self.perception_fps:.1f} Hz')
        if self.viz_enabled:
            self.get_logger().info(f'  Viz           : ON @ '
                                   f'{self.get_parameter("viz.fps").value} Hz '
                                   f'(debug per-cam: '
                                   f'{self.get_parameter("viz.publish_debug").value})')
        else:
            self.get_logger().info('  Viz           : OFF')
        self.get_logger().info(f'  Segmentation  : '
                               f'{"ON" if self.seg_engine else "OFF"}')

    # =========================================================================
    #  Parameters
    # =========================================================================

    def _declare_params(self):
        # BEV grid
        self.declare_parameter('bev.x_range', [-10.0, 15.0])
        self.declare_parameter('bev.y_range', [-10.0, 10.0])
        self.declare_parameter('bev.resolution', 0.05)
        self.declare_parameter('bev.height_range', [-0.05, 2.5])  # tighter floor

        # Camera mount poses (base_link frame, REP-103)
        self.declare_parameter('cameras.left.mount_x', -0.10)
        self.declare_parameter('cameras.left.mount_y',  0.35)
        self.declare_parameter('cameras.left.mount_z',  0.60)
        self.declare_parameter('cameras.left.mount_yaw', 1.5708)

        self.declare_parameter('cameras.front.mount_x', 0.35)
        self.declare_parameter('cameras.front.mount_y', 0.0)
        self.declare_parameter('cameras.front.mount_z', 0.75)
        self.declare_parameter('cameras.front.mount_yaw', 0.0)

        self.declare_parameter('cameras.right.mount_x', -0.10)
        self.declare_parameter('cameras.right.mount_y', -0.35)
        self.declare_parameter('cameras.right.mount_z',  0.60)
        self.declare_parameter('cameras.right.mount_yaw', -1.5708)

        # Depth filter
        self.declare_parameter('depth.min', 0.3)
        self.declare_parameter('depth.max', 12.0)  # IGVC course is small

        # Segmentation
        self.declare_parameter('segmentation.enabled', True)
        self.declare_parameter('segmentation.tier2_model_path', '')
        self.declare_parameter('segmentation.device', 'cuda')

        # HSV thresholds (Tier 1 — IGVC defaults; override at venue if needed)
        self.declare_parameter('segmentation.hsv.white.h_min',   0)
        self.declare_parameter('segmentation.hsv.white.h_max', 179)
        self.declare_parameter('segmentation.hsv.white.s_min',   0)
        self.declare_parameter('segmentation.hsv.white.s_max',  60)
        self.declare_parameter('segmentation.hsv.white.v_min', 180)
        self.declare_parameter('segmentation.hsv.white.v_max', 255)
        self.declare_parameter('segmentation.hsv.orange.h_min',   5)
        self.declare_parameter('segmentation.hsv.orange.h_max',  20)
        self.declare_parameter('segmentation.hsv.orange.s_min', 130)
        self.declare_parameter('segmentation.hsv.orange.s_max', 255)
        self.declare_parameter('segmentation.hsv.orange.v_min', 100)
        self.declare_parameter('segmentation.hsv.orange.v_max', 255)
        self.declare_parameter('segmentation.min_line_area_px',   30)
        self.declare_parameter('segmentation.min_barrel_area_px', 200)

        # Performance
        self.declare_parameter('perception.fps', 20.0)
        self.declare_parameter('perception.downsample_depth', 2)

        # Visualization
        self.declare_parameter('viz.enabled', True)
        self.declare_parameter('viz.fps', 2.0)
        self.declare_parameter('viz.publish_debug', True)
        self.declare_parameter('viz.overlay_alpha', 0.5)

        # Output: optionally dilate the obstacle mask so the planner has
        # margin around real obstacles. Value in BEV pixels (5 px @ 0.05
        # m/px = 25 cm).
        self.declare_parameter('output.obstacle_dilate_px', 4)

    # =========================================================================
    #  Camera setup
    # =========================================================================

    def _setup_cameras(self):
        min_d = float(self.get_parameter('depth.min').value)
        max_d = float(self.get_parameter('depth.max').value)

        cam_defs = {
            'left': {
                'rgb_topic':   '/zed_left/zed_node/rgb/image_rect_color',
                'depth_topic': '/zed_left/zed_node/depth/depth_registered',
                'info_topic':  '/zed_left/zed_node/rgb/camera_info',
            },
            'front': {
                'rgb_topic':   '/zed_front/zed_node/rgb/image_rect_color',
                'depth_topic': '/zed_front/zed_node/depth/depth_registered',
                'info_topic':  '/zed_front/zed_node/rgb/camera_info',
            },
            'right': {
                'rgb_topic':   '/zed_right/zed_node/rgb/image_rect_color',
                'depth_topic': '/zed_right/zed_node/depth/depth_registered',
                'info_topic':  '/zed_right/zed_node/rgb/camera_info',
            },
        }

        for cam_name, cfg in cam_defs.items():
            cam = CameraState(
                name=cam_name,
                mount_x=float(self.get_parameter(f'cameras.{cam_name}.mount_x').value),
                mount_y=float(self.get_parameter(f'cameras.{cam_name}.mount_y').value),
                mount_z=float(self.get_parameter(f'cameras.{cam_name}.mount_z').value),
                mount_yaw=float(self.get_parameter(f'cameras.{cam_name}.mount_yaw').value),
                min_depth=min_d, max_depth=max_d,
            )
            self.cameras[cam_name] = cam

            self.create_subscription(
                Image, cfg['rgb_topic'],
                lambda msg, cn=cam_name: self._rgb_callback(cn, msg),
                self.qos_sensor)
            self.create_subscription(
                Image, cfg['depth_topic'],
                lambda msg, cn=cam_name: self._depth_callback(cn, msg),
                self.qos_sensor)
            self.create_subscription(
                CameraInfo, cfg['info_topic'],
                lambda msg, cn=cam_name: self._info_callback(cn, msg),
                self.qos_sensor)

            self.get_logger().info(
                f'  [{cam_name}] yaw={math.degrees(cam.mount_yaw):+.0f}deg  '
                f'pos=({cam.mount_x:+.2f}, {cam.mount_y:+.2f}, {cam.mount_z:+.2f})'
            )

    # =========================================================================
    #  BEV grid
    # =========================================================================

    def _init_bev_grid(self):
        x_range = self.get_parameter('bev.x_range').value
        y_range = self.get_parameter('bev.y_range').value
        self.bev_res = float(self.get_parameter('bev.resolution').value)
        self.height_range = self.get_parameter('bev.height_range').value

        self.bev_x_min, self.bev_x_max = float(x_range[0]), float(x_range[1])
        self.bev_y_min, self.bev_y_max = float(y_range[0]), float(y_range[1])
        self.bev_w = int((self.bev_y_max - self.bev_y_min) / self.bev_res)
        self.bev_h = int((self.bev_x_max - self.bev_x_min) / self.bev_res)

    # =========================================================================
    #  Segmentation init
    # =========================================================================

    def _init_segmentation(self):
        if not self.get_parameter('segmentation.enabled').value:
            self.get_logger().info('  Segmentation: DISABLED by config')
            return
        if not HAS_SEG:
            self.get_logger().warn(
                '  Segmentation: seg_inference module not importable')
            return
        try:
            self.seg_engine = SegmentationEngine(
                white_hsv=self._read_hsv('white'),
                orange_hsv=self._read_hsv('orange'),
                min_line_area_px=int(self.get_parameter(
                    'segmentation.min_line_area_px').value),
                min_barrel_area_px=int(self.get_parameter(
                    'segmentation.min_barrel_area_px').value),
                model_path=self.get_parameter(
                    'segmentation.tier2_model_path').value,
                device=self.get_parameter('segmentation.device').value,
            )
        except Exception as e:
            self.get_logger().error(f'  Segmentation init failed: {e}')
            self.seg_engine = None

    def _read_hsv(self, color: str) -> dict:
        return {
            f'{ch}_{end}': int(self.get_parameter(
                f'segmentation.hsv.{color}.{ch}_{end}').value)
            for ch in ('h', 's', 'v') for end in ('min', 'max')
        }

    # =========================================================================
    #  Subscriber callbacks (light — just stash data)
    # =========================================================================

    def _rgb_callback(self, cam_name: str, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                cam = self.cameras[cam_name]
                cam.rgb = img
                cam.got_rgb = True
        except Exception as e:
            self.get_logger().warn(f'[{cam_name}] RGB convert error: {e}')

    def _depth_callback(self, cam_name: str, msg: Image):
        try:
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) * 0.001
                else:
                    depth = depth.astype(np.float32)
            with self._lock:
                cam = self.cameras[cam_name]
                cam.depth = depth
                cam.got_depth = True
        except Exception as e:
            self.get_logger().warn(f'[{cam_name}] depth convert error: {e}')

    def _info_callback(self, cam_name: str, msg: CameraInfo):
        cam = self.cameras[cam_name]
        if cam.got_info:
            return
        K = np.array(msg.k).reshape(3, 3)
        with self._lock:
            cam.intrinsics = K
            cam.img_w = msg.width
            cam.img_h = msg.height
            cam.got_info = True
        self.get_logger().info(
            f'  [{cam_name}] intrinsics  {msg.width}x{msg.height}  '
            f'fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}'
        )

    # =========================================================================
    #  PERCEPTION LOOP — runs at perception_fps, publishes machine outputs
    # =========================================================================

    def _perception_callback(self):
        t0 = time.monotonic()

        # Fresh canvases each tick.
        bev_seg_id = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        bev_rgb    = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)

        with self._lock:
            snaps = {}
            for name, cam in self.cameras.items():
                if cam.got_rgb and cam.got_depth and cam.got_info:
                    snaps[name] = (
                        cam.rgb.copy(),
                        cam.depth.copy(),
                        cam.intrinsics.copy(),
                        cam.mount_x, cam.mount_y, cam.mount_z, cam.mount_yaw,
                        cam.min_depth, cam.max_depth,
                    )
        if not snaps:
            return

        # ---- Run segmentation per camera ----------------------------
        seg_masks: Dict[str, np.ndarray] = {}
        if self.seg_engine is not None:
            for name, snap in snaps.items():
                try:
                    seg_masks[name] = self.seg_engine.infer(snap[0])
                except Exception as e:
                    self.get_logger().warn(
                        f'[{name}] seg inference error: {e}',
                        throttle_duration_sec=5.0)

        # ---- Project each camera into the shared BEV ----------------
        debug_imgs: Dict[str, np.ndarray] = {}
        for name, snap in snaps.items():
            rgb, depth, K, mx, my, mz, yaw, min_d, max_d = snap
            seg_mask = seg_masks.get(name)
            self._project_camera_to_bev(
                rgb, depth, K, mx, my, mz, yaw, min_d, max_d,
                bev_rgb, bev_seg_id,
                seg_mask=seg_mask,
            )
            # Build per-camera debug image only if anyone might consume it.
            if self.viz_enabled and self.pub_debug:
                debug_imgs[name] = self._build_debug_image(name, rgb, seg_mask)

        # ---- Derive drivable / obstacle masks -----------------------
        drivable_mask, obstacle_mask = self._derive_masks(bev_seg_id)

        # ---- Publish machine outputs --------------------------------
        seg_msg = self.bridge.cv2_to_imgmsg(bev_seg_id, encoding='mono8')
        drv_msg = self.bridge.cv2_to_imgmsg(drivable_mask, encoding='mono8')
        obs_msg = self.bridge.cv2_to_imgmsg(obstacle_mask, encoding='mono8')
        stamp = self.get_clock().now().to_msg()
        for m in (seg_msg, drv_msg, obs_msg):
            m.header.stamp = stamp
            m.header.frame_id = 'base_link'
        self.pub_seg_id.publish(seg_msg)
        self.pub_drivable.publish(drv_msg)
        self.pub_obstacle.publish(obs_msg)

        # ---- Stash latest for the viz timer -------------------------
        if self.viz_enabled:
            with self._latest_lock:
                self._latest_bev_rgb = bev_rgb
                self._latest_bev_seg_id = bev_seg_id
                self._latest_drivable = drivable_mask
                self._latest_obstacle = obstacle_mask
                if debug_imgs:
                    self._latest_debug = debug_imgs

        # ---- Stats ---------------------------------------------------
        self._perc_count += 1
        dt = time.monotonic() - t0
        # Exponential moving average of loop time
        self._perc_time_ema = 0.9 * self._perc_time_ema + 0.1 * dt if self._perc_time_ema else dt
        if self._perc_count % int(self.perception_fps * 5) == 0:
            self.get_logger().info(
                f'Perception: frame {self._perc_count}  |  {len(snaps)} cams  |  '
                f'loop {self._perc_time_ema * 1000:.1f} ms  '
                f'(headroom for {1.0 / max(self._perc_time_ema, 1e-6):.1f} Hz)  |  '
                f'seg={"ON" if seg_masks else "OFF"}'
            )

    # =========================================================================
    #  VIZ LOOP — runs at viz_fps, publishes BGR images for humans
    # =========================================================================

    def _viz_callback(self):
        with self._latest_lock:
            bev_rgb = None if self._latest_bev_rgb is None else self._latest_bev_rgb.copy()
            bev_seg_id = None if self._latest_bev_seg_id is None else self._latest_bev_seg_id.copy()
            debug_snapshot = dict(self._latest_debug)

        if bev_rgb is None or bev_seg_id is None:
            return

        # Build a colorized seg overlay from class IDs.
        seg_color = self._colorize_seg(bev_seg_id)

        # Compose fused.
        fused = self._blend_seg_over_rgb(bev_rgb, seg_color)
        rgb_with_chrome = bev_rgb.copy()
        self._draw_vehicle(rgb_with_chrome)
        self._draw_vehicle(fused)

        # Publish.
        if self.pub_bev_rgb is not None:
            self.pub_bev_rgb.publish(
                self.bridge.cv2_to_imgmsg(rgb_with_chrome, encoding='bgr8'))
        if self.pub_bev_fused is not None:
            self.pub_bev_fused.publish(
                self.bridge.cv2_to_imgmsg(fused, encoding='bgr8'))

        for cam_name, dbg in debug_snapshot.items():
            pub = self.pub_debug.get(cam_name)
            if pub is not None and dbg is not None:
                pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))

        self._viz_count += 1

    # =========================================================================
    #  Depth -> BEV projection
    # =========================================================================

    def _project_camera_to_bev(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mount_x: float, mount_y: float, mount_z: float, mount_yaw: float,
        min_depth: float, max_depth: float,
        bev_rgb: np.ndarray,
        bev_seg_id: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
    ):
        """
        Vectorized depth back-projection to BEV.

        Pothole-friendly: classes in GROUND_PLANE_CLASSES are exempt from
        the lower height filter so painted-on-pavement features survive.
        """
        h, w = depth.shape[:2]
        ds = self.downsample
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cos_y = math.cos(mount_yaw)
        sin_y = math.sin(mount_yaw)
        h_min, h_max = self.height_range

        vs = np.arange(0, h, ds)
        us = np.arange(0, w, ds)
        uu, vv = np.meshgrid(us, vs)
        d_sampled = depth[vv, uu].astype(np.float32)

        valid = (d_sampled > min_depth) & (d_sampled < max_depth) & np.isfinite(d_sampled)
        if not np.any(valid):
            return

        u_flat = uu[valid].astype(np.float32)
        v_flat = vv[valid].astype(np.float32)
        z_flat = d_sampled[valid]

        # ROS optical (Z forward, X right, Y down) -> base_link (X forward,
        # Y left, Z up).
        x_cam = (u_flat - cx) * z_flat / fx
        y_cam = (v_flat - cy) * z_flat / fy
        z_cam = z_flat
        x_veh = cos_y * z_cam - sin_y * x_cam + mount_x
        y_veh = sin_y * z_cam + cos_y * x_cam + mount_y
        z_veh = mount_z - y_cam

        # Pothole-friendly height filter: classes on the ground plane skip
        # the lower bound entirely (so flat painted features survive even
        # when their measured Z dips below h_min from sensor noise).
        u_int = u_flat.astype(np.int32)
        v_int = v_flat.astype(np.int32)
        u_int_clip = np.clip(u_int, 0, rgb.shape[1] - 1)
        v_int_clip = np.clip(v_int, 0, rgb.shape[0] - 1)

        if seg_mask is not None and seg_mask.shape[:2] == rgb.shape[:2]:
            cls = seg_mask[v_int_clip, u_int_clip]
        else:
            cls = np.zeros_like(u_int, dtype=np.uint8)

        is_ground = np.isin(cls, np.array(GROUND_PLANE_CLASSES, dtype=np.uint8))
        h_valid = (z_veh <= h_max) & (
            (z_veh >= h_min) | is_ground
        )
        if not np.any(h_valid):
            return

        x_veh = x_veh[h_valid]
        y_veh = y_veh[h_valid]
        u_src = u_int_clip[h_valid]
        v_src = v_int_clip[h_valid]
        cls   = cls[h_valid]

        bev_row = ((self.bev_x_max - x_veh) / self.bev_res).astype(np.int32)
        bev_col = ((y_veh - self.bev_y_min) / self.bev_res).astype(np.int32)
        in_bounds = (
            (bev_row >= 0) & (bev_row < self.bev_h) &
            (bev_col >= 0) & (bev_col < self.bev_w)
        )
        bev_row = bev_row[in_bounds]
        bev_col = bev_col[in_bounds]
        u_src = u_src[in_bounds]
        v_src = v_src[in_bounds]
        cls   = cls[in_bounds]

        if len(bev_row) == 0:
            return

        # Paint RGB.
        bev_rgb[bev_row, bev_col] = rgb[v_src, u_src]

        # Paint class IDs. Strategy: write where existing cell == 0
        # (background) OR existing cell is also background-class. For
        # collisions between two real classes, the higher class ID wins
        # (obstacles > drivable > line, since CLASS IDs were chosen so
        # that "more important" things have higher numbers... except
        # drivable=5 which is intentionally last). Use np.maximum which
        # is fast and gives stable results.
        if cls.size > 0:
            existing = bev_seg_id[bev_row, bev_col]
            # Override only when (a) existing is background, or (b) the
            # new class is an obstacle (non-zero, non-drivable). We never
            # want a drivable label to overwrite an obstacle.
            new_is_real = cls != CLASS_BACKGROUND
            new_is_obstacle = np.isin(cls, np.array(OBSTACLE_CLASSES, dtype=np.uint8))
            existing_is_obstacle = np.isin(existing, np.array(OBSTACLE_CLASSES, dtype=np.uint8))
            should_write = new_is_real & (
                (existing == CLASS_BACKGROUND) |
                (new_is_obstacle & ~existing_is_obstacle) |
                (new_is_obstacle & existing_is_obstacle)  # last-write-wins for obstacle/obstacle
            )
            sel_rows = bev_row[should_write]
            sel_cols = bev_col[should_write]
            sel_cls  = cls[should_write]
            bev_seg_id[sel_rows, sel_cols] = sel_cls

    # =========================================================================
    #  Mask derivation
    # =========================================================================

    def _derive_masks(self, bev_seg_id: np.ndarray):
        """
        Build the planner-facing binary masks from the class-id BEV.

          drivable_mask: 255 where class == CLASS_DRIVABLE.
                         When Tier 2 is OFF (no drivable predictions ever
                         arrive), fall back to "any non-obstacle cell that
                         the cameras hit" — which is approximated by RGB
                         BEV being non-empty. We approximate this cheaply
                         here by treating any seg-class == 0 cell that was
                         observed (we don't track hits separately to save
                         time) as background, and let the planner decide.

          obstacle_mask: 255 where class IN OBSTACLE_CLASSES, optionally
                         dilated by N pixels.
        """
        # Drivable: only trusted when Tier 2 explicitly labels it.
        drivable_mask = ((bev_seg_id == CLASS_DRIVABLE).astype(np.uint8)) * 255

        # Obstacle.
        obstacle_mask = np.isin(
            bev_seg_id, np.array(OBSTACLE_CLASSES, dtype=np.uint8)
        ).astype(np.uint8) * 255

        # Dilate for planner safety margin.
        dilate_px = int(self.get_parameter('output.obstacle_dilate_px').value)
        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
            )
            obstacle_mask = cv2.dilate(obstacle_mask, k)

        return drivable_mask, obstacle_mask

    # =========================================================================
    #  Compositing helpers (viz only)
    # =========================================================================

    def _colorize_seg(self, bev_seg_id: np.ndarray) -> np.ndarray:
        """class-ID image (HxW) -> colored BGR image (HxWx3)."""
        out = np.zeros((bev_seg_id.shape[0], bev_seg_id.shape[1], 3),
                       dtype=np.uint8)
        for cls_id, color in self._get_seg_colors().items():
            out[bev_seg_id == cls_id] = color
        return out

    def _blend_seg_over_rgb(self, bev_rgb: np.ndarray,
                             seg_color: np.ndarray) -> np.ndarray:
        alpha = float(self.get_parameter('viz.overlay_alpha').value)
        fused = bev_rgb.astype(np.float32)
        seg_present = np.any(seg_color > 0, axis=-1)
        if np.any(seg_present):
            fused[seg_present] = (
                fused[seg_present] * (1.0 - alpha)
                + seg_color[seg_present].astype(np.float32) * alpha
            )
        return np.clip(fused, 0, 255).astype(np.uint8)

    def _build_debug_image(self, cam_name: str, rgb: np.ndarray,
                            seg_mask: Optional[np.ndarray]) -> np.ndarray:
        debug_img = rgb.copy()
        if seg_mask is not None:
            seg_colors = self._get_seg_colors()
            alpha = float(self.get_parameter('viz.overlay_alpha').value)
            overlay = np.zeros_like(rgb)
            for cls_id, color in seg_colors.items():
                overlay[seg_mask == cls_id] = color
            mask = np.any(overlay > 0, axis=-1)
            debug_img[mask] = (
                debug_img[mask].astype(np.float32) * (1.0 - alpha)
                + overlay[mask].astype(np.float32) * alpha
            ).astype(np.uint8)
        cv2.putText(debug_img, cam_name.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return debug_img

    def _draw_vehicle(self, bev: np.ndarray):
        veh_l, veh_w = 1.2, 0.9  # IGVC-class robot footprint, meters
        cx_px = int((self.bev_x_max - 0.0) / self.bev_res)
        cy_px = int((0.0 - self.bev_y_min) / self.bev_res)
        hl = int(veh_l / 2 / self.bev_res)
        hw = int(veh_w / 2 / self.bev_res)
        pts = np.array([
            [cy_px - hw, cx_px - hl],
            [cy_px + hw, cx_px - hl],
            [cy_px + hw, cx_px + hl],
            [cy_px - hw, cx_px + hl],
        ], dtype=np.int32)
        cv2.polylines(bev, [pts], True, (0, 255, 255), 2)
        cv2.arrowedLine(
            bev, (cy_px, cx_px),
            (cy_px, cx_px - int(1.2 / self.bev_res)),
            (0, 255, 255), 2, tipLength=0.3,
        )

    # =========================================================================
    #  Class colors (BGR)  — keep in sync with seg_inference.py CLASS_*
    # =========================================================================

    @staticmethod
    def _get_seg_colors() -> Dict[int, tuple]:
        return {
            CLASS_BACKGROUND: (0, 0, 0),         # transparent-ish
            CLASS_LANE_LINE:  (255, 255, 255),   # white
            CLASS_BARREL:     (0, 140, 255),     # safety orange (BGR)
            CLASS_PERSON:     (0, 255, 255),     # yellow
            CLASS_POTHOLE:    (255, 0, 255),     # magenta
            CLASS_DRIVABLE:   (0, 180, 0),       # green
        }

    # =========================================================================
    #  Shutdown
    # =========================================================================

    def on_shutdown(self):
        self.get_logger().info(
            f'Shutdown:  perception={self._perc_count} frames  '
            f'viz={self._viz_count} frames'
        )


def main(args=None):
    rclpy.init(args=args)
    node = BevPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
