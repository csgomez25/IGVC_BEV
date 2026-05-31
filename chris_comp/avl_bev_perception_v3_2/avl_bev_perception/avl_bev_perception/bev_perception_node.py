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
        CLASS_POTHOLE, CLASS_PERSON, CLASS_DRIVABLE, CLASS_UNKNOWN,
        OBSTACLE_CLASSES,
    )
    HAS_SEG = True
except ImportError:
    HAS_SEG = False
    # Keep these literals in lockstep with seg_inference.py.
    CLASS_BACKGROUND = 0
    CLASS_LANE_LINE  = 1
    CLASS_BARREL     = 2
    CLASS_POTHOLE    = 3
    CLASS_PERSON     = 4
    CLASS_DRIVABLE   = 5
    CLASS_UNKNOWN    = 255
    OBSTACLE_CLASSES = (CLASS_LANE_LINE, CLASS_BARREL, CLASS_POTHOLE, CLASS_PERSON)

try:
    from .auto_calibrate import AutoHsvCalibrator
    HAS_AUTO_CAL = True
except ImportError:
    HAS_AUTO_CAL = False

from concurrent.futures import ThreadPoolExecutor


# Classes that live on the ground plane — projected without the normal
# height-filter floor so painted lines and potholes survive.
GROUND_PLANE_CLASSES = (CLASS_LANE_LINE, CLASS_POTHOLE)

# Pre-computed lookup table: 256-element bool array indexed by class ID,
# True if the class is an obstacle. Used to vectorize obstacle-priority
# painting in the projection loop without np.isin.
_OBSTACLE_LUT = np.zeros(256, dtype=bool)
for _cid in OBSTACLE_CLASSES:
    _OBSTACLE_LUT[_cid] = True

# Pre-computed seg color LUT: 256x3 uint8 BGR colors. Faster than dict
# lookups inside the projection loop.
_SEG_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
_SEG_COLOR_LUT[CLASS_BACKGROUND] = (0, 0, 0)
_SEG_COLOR_LUT[CLASS_LANE_LINE]  = (255, 255, 255)
_SEG_COLOR_LUT[CLASS_BARREL]     = (0, 140, 255)
_SEG_COLOR_LUT[CLASS_POTHOLE]    = (255, 0, 255)
_SEG_COLOR_LUT[CLASS_PERSON]     = (0, 255, 255)
_SEG_COLOR_LUT[CLASS_DRIVABLE]   = (0, 180, 0)
_SEG_COLOR_LUT[CLASS_UNKNOWN]    = (128, 128, 128)


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

    # ---- v3.2: pre-computed projection LUT cache ----
    # Built once when intrinsics arrive, reused every frame.
    # See _build_projection_lut() in BevPerceptionNode.
    lut_built:    bool = False
    lut_u_flat:   Optional[np.ndarray] = None  # downsampled flat u indices  (int32)
    lut_v_flat:   Optional[np.ndarray] = None  # downsampled flat v indices  (int32)
    lut_x_factor: Optional[np.ndarray] = None  # (u-cx)/fx   precomputed
    lut_y_factor: Optional[np.ndarray] = None  # (v-cy)/fy   precomputed
    lut_cos_yaw:  float = 0.0
    lut_sin_yaw:  float = 0.0

    # ---- Kiwicampus adapter state (path A; only used when enabled) ----
    # rgb_stamp / rgb_frame_id are stashed on the latest RGB msg so the
    # per-camera publishers can reuse the input sensor stamp (kiwicampus
    # message-syncs mask + cloud on header.stamp). cloud is the latest
    # organized PointCloud2 — relayed verbatim with stamp rewritten to
    # match the published mask.
    rgb_stamp:     Optional[object] = None   # builtin_interfaces/Time
    rgb_frame_id:  str = ''
    cloud:         Optional[object] = None   # sensor_msgs/PointCloud2 (latest)
    got_cloud:     bool = False


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

        # Read kiwicampus gate before _setup_cameras so it can register the
        # extra cloud subscribers in the same pass.
        self._kiwi_enabled = bool(
            self.get_parameter('kiwicampus.enabled').value)
        # Per-camera kiwicampus publishers, populated in
        # _init_kiwicampus_adapter() when enabled.
        self._kiwi_pubs: Dict[str, Dict[str, object]] = {}

        self.cameras: Dict[str, CameraState] = {}
        self._setup_cameras()
        self._check_zed_serials()

        if self._kiwi_enabled:
            self._init_kiwicampus_adapter()

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

        # v3.2: planner-facing boolean + latency instrumentation.
        from std_msgs.msg import Bool, Float32
        self.pub_lane_detected = self.create_publisher(
            Bool, '/bev/lane_lines_detected', 10)
        self._lane_min_pixels = int(
            self.get_parameter('output.lane_min_pixels').value)

        self.pub_latency: Optional[object] = None
        if bool(self.get_parameter('output.publish_latency').value):
            self.pub_latency = self.create_publisher(
                Float32, '/bev/perception_latency_ms', 10)

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

        # ----------- v3.2: parallel execution + auto-calibration -----
        self._parallel_cameras = bool(
            self.get_parameter('perception.parallel_cameras').value)
        # One worker per camera, capped at 4. Orin has 12 CPU cores so
        # parallelism is essentially free.
        n_workers = min(4, max(1, len(self.cameras)))
        self._executor = ThreadPoolExecutor(max_workers=n_workers)

        self._auto_cal:       Optional[AutoHsvCalibrator] = None
        self._auto_cal_done:  bool = False
        enable_cal = bool(self.get_parameter('segmentation.auto_calibrate').value)
        n_samples  = int(self.get_parameter('segmentation.auto_calibrate_n_samples').value)
        if enable_cal and HAS_AUTO_CAL and self.seg_engine is not None:
            self._auto_cal = AutoHsvCalibrator(n_samples=n_samples)
            self.get_logger().info(
                f'  Auto-calibration ENABLED  (n_samples={n_samples} per camera, '
                f'thresholds will update once collection completes)'
            )
        else:
            self._auto_cal_done = True  # skip phase entirely
            if not enable_cal:
                self.get_logger().info('  Auto-calibration disabled (config)')

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

        self.get_logger().info('=== AVL BEV Perception v3.2 (IGVC AutoNav) ===')
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
        if self._kiwi_enabled:
            _kiwi_prefix = str(
                self.get_parameter('kiwicampus.topic_prefix').value).rstrip('/')
            _kiwi_status = f'ON (per-camera {_kiwi_prefix}/<cam>/* adapter)'
        else:
            _kiwi_status = 'OFF (standalone /bev/* only)'
        self.get_logger().info(f'  Kiwicampus    : {_kiwi_status}')

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

        # Camera topic paths. Defaults assume zed-ros2-wrapper v5.x
        # (`rgb/color/rect/*`). For v4.x bringups override to
        # `/zed_<cam>/zed_node/rgb/image_rect_color` etc via the YAML config
        # or `ros2 param set`.
        self.declare_parameter('cameras.left.rgb_topic',
                               '/zed_left/zed_node/rgb/color/rect/image')
        self.declare_parameter('cameras.left.depth_topic',
                               '/zed_left/zed_node/depth/depth_registered')
        self.declare_parameter('cameras.left.info_topic',
                               '/zed_left/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('cameras.left.cloud_topic',
                               '/zed_left/zed_node/point_cloud/cloud_registered')

        self.declare_parameter('cameras.front.rgb_topic',
                               '/zed_front/zed_node/rgb/color/rect/image')
        self.declare_parameter('cameras.front.depth_topic',
                               '/zed_front/zed_node/depth/depth_registered')
        self.declare_parameter('cameras.front.info_topic',
                               '/zed_front/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('cameras.front.cloud_topic',
                               '/zed_front/zed_node/point_cloud/cloud_registered')

        self.declare_parameter('cameras.right.rgb_topic',
                               '/zed_right/zed_node/rgb/color/rect/image')
        self.declare_parameter('cameras.right.depth_topic',
                               '/zed_right/zed_node/depth/depth_registered')
        self.declare_parameter('cameras.right.info_topic',
                               '/zed_right/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('cameras.right.cloud_topic',
                               '/zed_right/zed_node/point_cloud/cloud_registered')

        # Kiwicampus per-camera adapter (Parsa's Nav2 contract). Off by
        # default — flipping this on adds /perception/<cam>/* publishers,
        # subscribes to each camera's organized cloud, and latches a
        # vision_msgs/LabelInfo so the kiwicampus costmap layer can decode
        # our class IDs. /bev/* outputs are unaffected.
        self.declare_parameter('kiwicampus.enabled', False)
        # Namespace prefix for the per-camera contract topics. Default
        # '/perception' makes the adapter drop straight into Parsa's
        # nav2_params_humble.yaml semantic_layer sources (which already point at
        # /perception/<cam>/semantic_*). Override to e.g. '/bev_perception' to
        # run alongside his front perception_node without a topic collision,
        # then add the prefixed topics as an extra observation_source.
        self.declare_parameter('kiwicampus.topic_prefix', '/perception')

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

        # Tier 1b — Otsu bright-object pass for potholes / unclassified bright stuff.
        self.declare_parameter('segmentation.bright.enabled',     True)
        self.declare_parameter('segmentation.bright.min_area_px',   40)
        self.declare_parameter('segmentation.bright.max_area_px', 8000)
        self.declare_parameter('segmentation.bright.blur_px',        5)

        # Performance
        self.declare_parameter('perception.fps', 20.0)
        self.declare_parameter('perception.downsample_depth', 2)

        # Visualization
        self.declare_parameter('viz.enabled', True)
        self.declare_parameter('viz.fps', 2.0)
        self.declare_parameter('viz.publish_debug', True)
        self.declare_parameter('viz.overlay_alpha', 0.5)

        # Output: optionally dilate the obstacle mask so the planner has
        # margin around real obstacles. Value in BEV pixels.
        # v3.2 default raised from 4 to 8 (40 cm) for differential-drive bots
        # which need more turning room around obstacles than swerve drive.
        self.declare_parameter('output.obstacle_dilate_px', 8)

        # ---- v3.2 additions ----
        # Auto-HSV calibration at startup (Twistopher-inspired)
        self.declare_parameter('segmentation.auto_calibrate',           True)
        self.declare_parameter('segmentation.auto_calibrate_n_samples', 30)

        # Lane-line presence flag — published as /bev/lane_lines_detected.
        # Planner uses this to switch between lane-following and GPS-waypoint
        # modes for the IGVC offroad section.
        self.declare_parameter('output.lane_min_pixels', 200)

        # Latency instrumentation — publish per-frame loop time on
        # /bev/perception_latency_ms for monitoring.
        self.declare_parameter('output.publish_latency', True)

        # Parallel per-camera processing using Orin's 12-core CPU.
        # Set false to debug serial behavior.
        self.declare_parameter('perception.parallel_cameras', True)

    # =========================================================================
    #  Camera setup
    # =========================================================================

    def _setup_cameras(self):
        min_d = float(self.get_parameter('depth.min').value)
        max_d = float(self.get_parameter('depth.max').value)

        for cam_name in ('left', 'front', 'right'):
            rgb_topic   = str(self.get_parameter(f'cameras.{cam_name}.rgb_topic').value)
            depth_topic = str(self.get_parameter(f'cameras.{cam_name}.depth_topic').value)
            info_topic  = str(self.get_parameter(f'cameras.{cam_name}.info_topic').value)

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
                Image, rgb_topic,
                lambda msg, cn=cam_name: self._rgb_callback(cn, msg),
                self.qos_sensor)
            self.create_subscription(
                Image, depth_topic,
                lambda msg, cn=cam_name: self._depth_callback(cn, msg),
                self.qos_sensor)
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, cn=cam_name: self._info_callback(cn, msg),
                self.qos_sensor)

            self.get_logger().info(
                f'  [{cam_name}] yaw={math.degrees(cam.mount_yaw):+.0f}deg  '
                f'pos=({cam.mount_x:+.2f}, {cam.mount_y:+.2f}, {cam.mount_z:+.2f})'
            )
            self.get_logger().info(f'  [{cam_name}]   rgb   {rgb_topic}')
            self.get_logger().info(f'  [{cam_name}]   depth {depth_topic}')
            self.get_logger().info(f'  [{cam_name}]   info  {info_topic}')

            # Kiwicampus adapter: also subscribe to the camera's organized
            # cloud. We relay it verbatim under /perception/<cam>/semantic_points
            # so kiwicampus's TimeSynchronizer sees mask + cloud as a pair.
            if self._kiwi_enabled:
                from sensor_msgs.msg import PointCloud2
                cloud_topic = str(self.get_parameter(
                    f'cameras.{cam_name}.cloud_topic').value)
                self.create_subscription(
                    PointCloud2, cloud_topic,
                    lambda msg, cn=cam_name: self._cloud_callback(cn, msg),
                    self.qos_sensor)
                self.get_logger().info(
                    f'  [{cam_name}]   cloud {cloud_topic} (kiwicampus adapter)')

    # =========================================================================
    #  Kiwicampus adapter setup (path A — per-camera Nav2 contract)
    # =========================================================================

    # IDs 0–3 + 255 use Parsa's class_map.yaml names verbatim so his
    # nav2_params_humble.yaml `class_types: [...]` lists pick them up.
    # IDs 4 and 5 keep our local names — kiwicampus silently ignores classes
    # not listed in class_types, which is the intended behavior.
    _KIWI_CLASS_NAMES = {
        CLASS_BACKGROUND: 'free',
        CLASS_LANE_LINE:  'lane_white',
        CLASS_BARREL:     'barrel_orange',
        CLASS_POTHOLE:    'pothole',
        CLASS_PERSON:     'person',
        CLASS_DRIVABLE:   'drivable',
        CLASS_UNKNOWN:    'unknown',
    }

    def _init_kiwicampus_adapter(self) -> None:
        """
        Create per-camera publishers for the kiwicampus contract:
          /perception/<cam>/semantic_mask        Image mono8       BEST_EFFORT
          /perception/<cam>/semantic_confidence  Image mono8       BEST_EFFORT
          /perception/<cam>/semantic_points      PointCloud2       BEST_EFFORT
          /perception/<cam>/label_info           LabelInfo  latched (TL+REL, d=1)
        LabelInfo is published once here at startup — kiwicampus is a late
        joiner via the transient_local QoS.
        """
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import DurabilityPolicy
        try:
            from vision_msgs.msg import LabelInfo, VisionClass
        except ImportError:
            self.get_logger().error(
                'kiwicampus.enabled=true but vision_msgs is not importable. '
                'Install with `apt install ros-${ROS_DISTRO}-vision-msgs` and '
                'add <exec_depend>vision_msgs</exec_depend> to package.xml. '
                'Disabling the adapter for this run.'
            )
            self._kiwi_enabled = False
            return

        # Match Parsa's `qos_profile_sensor_data` for the streaming triple,
        # and TRANSIENT_LOCAL+RELIABLE depth=1 for the latched label_info.
        from rclpy.qos import qos_profile_sensor_data
        label_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        prefix = str(self.get_parameter('kiwicampus.topic_prefix').value).rstrip('/')
        if not prefix.startswith('/'):
            prefix = '/' + prefix
        for cam_name in self.cameras:
            ns = f'{prefix}/{cam_name}'
            pubs = {
                'mask':  self.create_publisher(
                    Image, f'{ns}/semantic_mask', qos_profile_sensor_data),
                'conf':  self.create_publisher(
                    Image, f'{ns}/semantic_confidence', qos_profile_sensor_data),
                'cloud': self.create_publisher(
                    PointCloud2, f'{ns}/semantic_points', qos_profile_sensor_data),
                'label': self.create_publisher(
                    LabelInfo, f'{ns}/label_info', label_qos),
            }
            self._kiwi_pubs[cam_name] = pubs

            # Build & publish the LabelInfo once. Late joiners pick it up
            # via transient_local. No frame_id (kiwicampus doesn't read it).
            label_msg = LabelInfo()
            label_msg.header.stamp = self.get_clock().now().to_msg()
            for cid, cname in self._KIWI_CLASS_NAMES.items():
                vc = VisionClass()
                vc.class_id = int(cid)
                vc.class_name = cname
                label_msg.class_map.append(vc)
            pubs['label'].publish(label_msg)
            self.get_logger().info(
                f'  [{cam_name}] kiwicampus adapter: latched '
                f'{len(self._KIWI_CLASS_NAMES)} classes on {ns}/label_info')

    # =========================================================================
    #  Serial sanity check  (Edit #4 — v3.2.2)
    # =========================================================================

    # MUST stay in sync with launch/zed_cameras.launch.py CAMERA_BINDINGS.
    # Mapping verified per-port 2026-04-24 (cross-referenced against
    # references/parsa_igvc/src/avros_bringup/launch/sensors.launch.py).
    EXPECTED_SERIALS = {
        'left':  43779087,
        'front': 42569280,
        'right': 49910017,
    }

    def _check_zed_serials(self) -> None:
        """
        Log which ZED serials are connected vs. what the launch expects.
        Tier 1: ZED SDK device probe (real serials, definitive).
        Tier 2: 5-second CameraInfo-presence timer (no SDK dep, informational).
        Never raises — wrong cabling should produce loud logs, not a node crash.
        """
        try:
            import pyzed.sl as sl
            devices = sl.Camera.get_device_list()
            connected = sorted(int(d.serial_number) for d in devices)
            self.get_logger().info(
                f'  ZED SDK reports {len(connected)} camera(s) connected: '
                f'{connected}'
            )
            expected = set(self.EXPECTED_SERIALS.values())
            missing = expected - set(connected)
            extra = set(connected) - expected
            for cam_name, want_sn in self.EXPECTED_SERIALS.items():
                if want_sn in connected:
                    self.get_logger().info(
                        f'  [{cam_name}] serial {want_sn} OK')
                else:
                    self.get_logger().warn(
                        f'  [{cam_name}] serial {want_sn} NOT connected — '
                        f'check cable / power / config'
                    )
            if extra:
                self.get_logger().warn(
                    f'  Connected serials not in config: {sorted(extra)} '
                    f'(extra cameras attached or serial mapping stale)'
                )
            return
        except ImportError:
            self.get_logger().info(
                '  pyzed not installed — falling back to topic-presence '
                'check in 5s (serial values not verified, only connectivity)'
            )
        except Exception as e:
            self.get_logger().warn(
                f'  ZED SDK device probe failed ({e}); '
                f'falling back to topic-presence check')

        # Tier 2 fallback: one-shot timer that fires after 5s and logs
        # which cameras have produced a CameraInfo. Cheap, no SDK dep.
        self._serial_probe_timer = self.create_timer(
            5.0, self._on_serial_probe_fallback)

    def _on_serial_probe_fallback(self) -> None:
        try:
            for cam_name, want_sn in self.EXPECTED_SERIALS.items():
                cam = self.cameras.get(cam_name)
                seen = bool(cam and cam.got_info)
                level = (self.get_logger().info if seen
                         else self.get_logger().warn)
                level(
                    f'  [{cam_name}] expected S/N {want_sn}: '
                    f'CameraInfo {"received" if seen else "MISSING after 5s"}'
                )
        finally:
            # One-shot; tear down so we don't spam the log.
            self._serial_probe_timer.cancel()
            self._serial_probe_timer = None

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
                bright_pass_enabled=bool(self.get_parameter(
                    'segmentation.bright.enabled').value),
                bright_min_area_px=int(self.get_parameter(
                    'segmentation.bright.min_area_px').value),
                bright_max_area_px=int(self.get_parameter(
                    'segmentation.bright.max_area_px').value),
                bright_blur_px=int(self.get_parameter(
                    'segmentation.bright.blur_px').value),
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

    def _finalize_auto_calibration(self) -> None:
        """
        v3.2: Called once when the calibrator has collected enough samples.
        Computes venue-tuned HSV thresholds and pushes them into the
        seg engine. Marks calibration done so it doesn't re-run.
        """
        if self._auto_cal is None or self._auto_cal_done:
            return
        default_white  = self._read_hsv('white')
        default_orange = self._read_hsv('orange')
        white, orange, debug = self._auto_cal.compute_thresholds(
            default_white, default_orange)

        if debug.get('fallback'):
            self.get_logger().warn(
                f'  Auto-calibration FELL BACK to defaults: '
                f'{debug.get("reason", "validation failed")}'
            )
        else:
            self.get_logger().info(
                f'  Auto-calibration SUCCESS:  '
                f'white V_min {default_white["v_min"]}→{white["v_min"]}  '
                f'orange S_min {default_orange["s_min"]}→{orange["s_min"]}  '
                f'(asphalt V={debug["asphalt_median_hsv"][2]:.0f} '
                f'σV={debug["asphalt_std_sv"][1]:.1f})'
            )
        if self.seg_engine is not None:
            self.seg_engine.set_hsv_thresholds(white, orange)
        self._auto_cal_done = True
        # Drop reference to free sample memory.
        self._auto_cal = None

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
                # Stamp/frame are only consumed by the kiwicampus adapter,
                # but stashing them unconditionally keeps the lock-critical
                # section uniform.
                cam.rgb_stamp = msg.header.stamp
                cam.rgb_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'[{cam_name}] RGB convert error: {e}')

    def _cloud_callback(self, cam_name: str, msg) -> None:
        # Latest-stash, like the other sensor callbacks. The publish path in
        # _perception_callback grabs whatever's freshest at tick time. Cheap
        # — we don't deserialize, just hold the message reference.
        with self._lock:
            cam = self.cameras[cam_name]
            cam.cloud = msg
            cam.got_cloud = True

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
        # v3.2: precompute projection LUT once per camera. Saves ~5-10ms
        # per frame by not recomputing meshgrids / trig every loop.
        self._build_projection_lut(cam_name)

    def _build_projection_lut(self, cam_name: str) -> None:
        """
        Pre-compute per-camera projection factors that never change:
          - downsampled (u, v) grid flattened
          - (u - cx)/fx  and  (v - cy)/fy   which depend only on intrinsics
          - cos(yaw), sin(yaw) from mount pose

        In the hot path, BEV projection becomes:
          x_cam = lut_x_factor * z
          y_cam = lut_y_factor * z
          x_veh = cos_yaw * z  - sin_yaw * x_cam + mount_x
          y_veh = sin_yaw * z  + cos_yaw * x_cam + mount_y
          z_veh = mount_z - y_cam
        """
        cam = self.cameras[cam_name]
        if not cam.got_info or cam.intrinsics is None:
            return

        fx = float(cam.intrinsics[0, 0])
        fy = float(cam.intrinsics[1, 1])
        cx = float(cam.intrinsics[0, 2])
        cy = float(cam.intrinsics[1, 2])
        h, w = cam.img_h, cam.img_w
        ds = self.downsample

        vs = np.arange(0, h, ds)
        us = np.arange(0, w, ds)
        uu, vv = np.meshgrid(us, vs)
        u_flat = uu.flatten().astype(np.int32)
        v_flat = vv.flatten().astype(np.int32)

        cam.lut_u_flat   = u_flat
        cam.lut_v_flat   = v_flat
        cam.lut_x_factor = (u_flat.astype(np.float32) - cx) / fx
        cam.lut_y_factor = (v_flat.astype(np.float32) - cy) / fy
        cam.lut_cos_yaw  = float(np.cos(cam.mount_yaw))
        cam.lut_sin_yaw  = float(np.sin(cam.mount_yaw))
        cam.lut_built    = True
        self.get_logger().info(
            f'  [{cam_name}] projection LUT built  '
            f'(n_samples={u_flat.size:,}  ds={ds})'
        )

    # =========================================================================
    #  PERCEPTION LOOP — runs at perception_fps, publishes machine outputs
    # =========================================================================

    def _perception_callback(self):
        t0 = time.monotonic()

        # Fresh canvases each tick.
        bev_seg_id = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        bev_rgb    = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)

        # Snapshot camera state. `snaps[name] = (cam_ref, rgb, depth)`.
        # We keep a reference to the CameraState so the projection function
        # can use its precomputed LUT + mount pose fields directly.
        # `kiwi_snaps[name] = (rgb_stamp, rgb_frame_id, cloud_msg)` — only
        # populated when the kiwicampus adapter is on and the camera has
        # both a fresh RGB stamp and an organized cloud. Cameras without a
        # cloud are skipped from the per-camera publish but still go through
        # the standalone BEV path normally.
        kiwi_snaps: Dict[str, tuple] = {}
        with self._lock:
            snaps = {}
            for name, cam in self.cameras.items():
                if (cam.got_rgb and cam.got_depth and cam.got_info
                        and cam.lut_built):
                    snaps[name] = (cam, cam.rgb.copy(), cam.depth.copy())
                    if (self._kiwi_enabled and cam.got_cloud
                            and cam.cloud is not None
                            and cam.rgb_stamp is not None):
                        kiwi_snaps[name] = (
                            cam.rgb_stamp, cam.rgb_frame_id, cam.cloud)
        if not snaps:
            return

        # ---- v3.2: Auto-HSV calibration phase ----------------------
        # Run for the first N frames after startup, then rebuild seg engine
        # with venue-tuned thresholds.
        if (self._auto_cal is not None
                and not self._auto_cal_done
                and self.seg_engine is not None):
            for name, (_, rgb, _) in snaps.items():
                self._auto_cal.add_sample(name, rgb)
            if self._auto_cal.is_ready(list(snaps.keys())):
                self._finalize_auto_calibration()
            # During calibration, still run perception normally so the
            # robot isn't blind while calibrating.

        # ---- Run segmentation per camera ---------------------------
        seg_masks: Dict[str, np.ndarray] = {}
        if self.seg_engine is not None:
            if self._parallel_cameras and len(snaps) > 1:
                # Parallel seg — leverage Orin's 12-core CPU.
                def _do_seg(item):
                    name, (_, rgb, _) = item
                    try:
                        return name, self.seg_engine.infer(rgb)
                    except Exception as e:
                        self.get_logger().warn(
                            f'[{name}] seg inference error: {e}',
                            throttle_duration_sec=5.0)
                        return name, None
                for name, mask in self._executor.map(_do_seg, snaps.items()):
                    if mask is not None:
                        seg_masks[name] = mask
            else:
                for name, (_, rgb, _) in snaps.items():
                    try:
                        seg_masks[name] = self.seg_engine.infer(rgb)
                    except Exception as e:
                        self.get_logger().warn(
                            f'[{name}] seg inference error: {e}',
                            throttle_duration_sec=5.0)

        # ---- Kiwicampus per-camera publish (path A, optional) ----
        # Hooks the per-camera Tier 1 mask into Parsa's Nav2 stack via the
        # kiwicampus semantic_segmentation_layer contract. Independent of
        # the BEV mosaic — runs even if projection has issues.
        if self._kiwi_enabled and self._kiwi_pubs and kiwi_snaps and seg_masks:
            self._publish_kiwicampus(kiwi_snaps, seg_masks)

        # ---- Project each camera into the shared BEV --------------
        # Projection writes are serial (shared BEV canvas); this is fine
        # because the bulk of per-camera work (HSV+Otsu segmentation and
        # depth validity filtering) already ran in parallel above.
        debug_imgs: Dict[str, np.ndarray] = {}
        paint_rgb = self.viz_enabled
        for name, (cam, rgb, depth) in snaps.items():
            seg_mask = seg_masks.get(name)
            self._project_camera_to_bev(
                cam, rgb, depth,
                bev_rgb, bev_seg_id,
                seg_mask=seg_mask,
                paint_rgb=paint_rgb,
            )
            if self.viz_enabled and self.pub_debug:
                debug_imgs[name] = self._build_debug_image(name, rgb, seg_mask)

        # ---- Derive drivable / obstacle masks ---------------------
        drivable_mask, obstacle_mask = self._derive_masks(bev_seg_id)

        # ---- Publish machine outputs ------------------------------
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

        # ---- v3.2: lane-lines-detected boolean for planner -------
        if self.pub_lane_detected is not None:
            n_lane = int((bev_seg_id == CLASS_LANE_LINE).sum())
            from std_msgs.msg import Bool
            bmsg = Bool()
            bmsg.data = n_lane >= self._lane_min_pixels
            self.pub_lane_detected.publish(bmsg)

        # ---- Stash latest for the viz timer ----------------------
        if self.viz_enabled:
            with self._latest_lock:
                self._latest_bev_rgb = bev_rgb
                self._latest_bev_seg_id = bev_seg_id
                self._latest_drivable = drivable_mask
                self._latest_obstacle = obstacle_mask
                if debug_imgs:
                    self._latest_debug = debug_imgs

        # ---- Stats -----------------------------------------------
        self._perc_count += 1
        dt = time.monotonic() - t0
        self._perc_time_ema = 0.9 * self._perc_time_ema + 0.1 * dt if self._perc_time_ema else dt

        # v3.2: publish latency for external monitoring.
        if self.pub_latency is not None:
            from std_msgs.msg import Float32
            lmsg = Float32()
            lmsg.data = float(dt * 1000.0)
            self.pub_latency.publish(lmsg)

        if self._perc_count % int(self.perception_fps * 5) == 0:
            self.get_logger().info(
                f'Perception: frame {self._perc_count}  |  {len(snaps)} cams  |  '
                f'loop {self._perc_time_ema * 1000:.1f} ms  '
                f'(headroom for {1.0 / max(self._perc_time_ema, 1e-6):.1f} Hz)  |  '
                f'seg={"ON" if seg_masks else "OFF"}  |  '
                f'cal={"DONE" if self._auto_cal_done else "PENDING"}'
            )

    # =========================================================================
    #  Kiwicampus publish path — per-camera contract for Parsa's Nav2 stack
    # =========================================================================

    def _publish_kiwicampus(
        self,
        kiwi_snaps: Dict[str, tuple],
        seg_masks:  Dict[str, np.ndarray],
    ) -> None:
        """
        Publish the kiwicampus per-camera contract for each camera that has
        both a fresh Tier-1 seg mask AND an organized cloud this tick.

        Contract (all three messages share header.stamp = max(rgb, cloud)):
          /perception/<cam>/semantic_mask        Image mono8, H×W = cloud H×W
          /perception/<cam>/semantic_confidence  Image mono8, same H×W
          /perception/<cam>/semantic_points      PointCloud2, original organized cloud

        Mask resize uses INTER_NEAREST — class IDs are categorical so bilinear
        would invent intermediate IDs.
        """
        for name, (rgb_stamp, rgb_frame_id, cloud_msg) in kiwi_snaps.items():
            mask = seg_masks.get(name)
            if mask is None:
                continue
            pubs = self._kiwi_pubs.get(name)
            if pubs is None:
                continue

            cloud_h = int(cloud_msg.height)
            cloud_w = int(cloud_msg.width)
            if cloud_h <= 1 or cloud_w <= 0:
                # Unorganized cloud — kiwicampus needs height>1 (image-shaped).
                self.get_logger().warn(
                    f'[{name}] kiwicampus: cloud is unorganized '
                    f'({cloud_h}x{cloud_w}), skipping publish',
                    throttle_duration_sec=10.0)
                continue

            if mask.shape != (cloud_h, cloud_w):
                mask_pub = cv2.resize(
                    mask, (cloud_w, cloud_h),
                    interpolation=cv2.INTER_NEAREST)
            else:
                mask_pub = mask
            # Tier 1 is binary per class — 255 wherever the mask is not
            # background and not unknown.
            conf_pub = np.where(
                (mask_pub != CLASS_BACKGROUND) & (mask_pub != CLASS_UNKNOWN),
                np.uint8(255), np.uint8(0),
            ).astype(np.uint8)

            # Shared stamp = max(rgb_stamp, cloud_stamp), like Parsa.
            cloud_stamp = cloud_msg.header.stamp
            if ((cloud_stamp.sec, cloud_stamp.nanosec) >
                    (rgb_stamp.sec, rgb_stamp.nanosec)):
                stamp = cloud_stamp
            else:
                stamp = rgb_stamp
            frame_id = rgb_frame_id or cloud_msg.header.frame_id

            mask_msg = self.bridge.cv2_to_imgmsg(mask_pub, encoding='mono8')
            mask_msg.header.stamp = stamp
            mask_msg.header.frame_id = frame_id
            pubs['mask'].publish(mask_msg)

            conf_msg = self.bridge.cv2_to_imgmsg(conf_pub, encoding='mono8')
            conf_msg.header.stamp = stamp
            conf_msg.header.frame_id = frame_id
            pubs['conf'].publish(conf_msg)

            # Relay the organized cloud with stamp rewritten to match the
            # mask so kiwicampus's TimeSynchronizer pairs them. The cloud
            # contents (points, fields) are passed through verbatim.
            cloud_msg.header.stamp = stamp
            pubs['cloud'].publish(cloud_msg)

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
        cam: 'CameraState',
        rgb: np.ndarray,
        depth: np.ndarray,
        bev_rgb: np.ndarray,
        bev_seg_id: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
        paint_rgb: bool = True,
    ):
        """
        Vectorized depth back-projection to BEV, using precomputed
        per-camera LUT for ~5-10ms saved per frame per camera.

        Pothole-friendly: classes in GROUND_PLANE_CLASSES are exempt from
        the lower height filter so painted-on-pavement features survive.

        When `paint_rgb` is False, skip writing into bev_rgb entirely.
        Perception loop sets this to False when viz is disabled to save
        a large memory write.
        """
        if not cam.lut_built:
            # Intrinsics haven't arrived yet; defer.
            return

        u_flat       = cam.lut_u_flat
        v_flat       = cam.lut_v_flat
        x_factor     = cam.lut_x_factor
        y_factor     = cam.lut_y_factor
        cos_y        = cam.lut_cos_yaw
        sin_y        = cam.lut_sin_yaw
        mount_x      = cam.mount_x
        mount_y      = cam.mount_y
        mount_z      = cam.mount_z
        min_depth    = cam.min_depth
        max_depth    = cam.max_depth
        h_min, h_max = self.height_range

        # Read depth at the downsampled grid positions.
        d_sampled = depth[v_flat, u_flat].astype(np.float32)

        valid = (d_sampled > min_depth) & (d_sampled < max_depth) & np.isfinite(d_sampled)
        if not np.any(valid):
            return

        z = d_sampled[valid]
        xf = x_factor[valid]
        yf = y_factor[valid]
        u_src = u_flat[valid]
        v_src = v_flat[valid]

        # Back-project: cam-frame XYZ.
        x_cam = xf * z
        y_cam = yf * z
        # Vehicle frame (X fwd, Y left, Z up).
        x_veh = cos_y * z  - sin_y * x_cam + mount_x
        y_veh = sin_y * z  + cos_y * x_cam + mount_y
        z_veh = mount_z - y_cam

        # Seg class per valid sample (if seg mask provided).
        if seg_mask is not None and seg_mask.shape[:2] == rgb.shape[:2]:
            cls = seg_mask[v_src, u_src]
        else:
            cls = np.zeros(u_src.shape, dtype=np.uint8)

        # Pothole-friendly height filter. Use precomputed ground-plane LUT
        # for vectorized class check (faster than np.isin).
        ground_lut = np.zeros(256, dtype=bool)
        for cid in GROUND_PLANE_CLASSES:
            ground_lut[cid] = True
        is_ground = ground_lut[cls]

        h_valid = (z_veh <= h_max) & ((z_veh >= h_min) | is_ground)
        if not np.any(h_valid):
            return

        x_veh = x_veh[h_valid]
        y_veh = y_veh[h_valid]
        u_src = u_src[h_valid]
        v_src = v_src[h_valid]
        cls   = cls[h_valid]

        bev_row = ((self.bev_x_max - x_veh) / self.bev_res).astype(np.int32)
        bev_col = ((y_veh - self.bev_y_min) / self.bev_res).astype(np.int32)
        in_bounds = (
            (bev_row >= 0) & (bev_row < self.bev_h) &
            (bev_col >= 0) & (bev_col < self.bev_w)
        )
        if not np.any(in_bounds):
            return

        bev_row = bev_row[in_bounds]
        bev_col = bev_col[in_bounds]
        u_src = u_src[in_bounds]
        v_src = v_src[in_bounds]
        cls   = cls[in_bounds]

        # Paint RGB (skipped when viz is off — saves a large write).
        if paint_rgb:
            bev_rgb[bev_row, bev_col] = rgb[v_src, u_src]

        # Paint class IDs with obstacle-precedence rule.
        if cls.size > 0:
            existing             = bev_seg_id[bev_row, bev_col]
            new_is_real          = cls != CLASS_BACKGROUND
            new_is_obstacle      = _OBSTACLE_LUT[cls]
            existing_is_obstacle = _OBSTACLE_LUT[existing]
            should_write = new_is_real & (
                (existing == CLASS_BACKGROUND) |
                (new_is_obstacle & ~existing_is_obstacle) |
                (new_is_obstacle & existing_is_obstacle)  # last-write-wins obstacle/obstacle
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
            CLASS_POTHOLE:    (255, 0, 255),     # magenta
            CLASS_PERSON:     (0, 255, 255),     # yellow
            CLASS_DRIVABLE:   (0, 180, 0),       # green
            CLASS_UNKNOWN:    (128, 128, 128),   # gray
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
