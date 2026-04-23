# avl_bev_perception (v3.2 — IGVC AutoNav)

3-camera ZED X Bird's Eye View perception tuned for the IGVC AutoNav course.
Hybrid segmentation (HSV + Otsu + optional ONNX), decoupled perception/viz
timers, optimized for **Jetson AGX Orin** + tank/differential drive.

## What's new in v3.2

- **Auto-HSV calibration at startup.** Inspired by Oklahoma's Twistopher
  (IGVC 2025 Auto-Nav 1st place, 2:20). Captures asphalt samples from each
  camera at boot, derives venue-tuned HSV thresholds, and pushes them into
  the seg engine before the first perception tick. Survives lighting drift
  between runs without manual recalibration.
- **Pre-computed projection LUTs.** The per-camera `(u-cx)/fx`, `(v-cy)/fy`,
  meshgrid, and `cos/sin(yaw)` factors are computed once when intrinsics
  arrive and cached on the `CameraState`. Saves 5-10ms per camera per
  frame in the projection loop. Numerically equivalent to the previous
  per-frame computation (verified to 1e-6).
- **Parallel per-camera segmentation.** ThreadPoolExecutor with one worker
  per camera. On the Orin's 12-core CPU this is essentially free and cuts
  per-loop seg latency by ~2-3x. Toggle with
  `perception.parallel_cameras`.
- **Lane-detected boolean for planner mode switch.**
  `/bev/lane_lines_detected` (`std_msgs/Bool`) — true when ≥N lane-line
  pixels are visible. Lets the planner cleanly switch between
  lane-following and GPS-waypoint modes without GPS-zone gating.
- **Latency instrumentation.** `/bev/perception_latency_ms`
  (`std_msgs/Float32`) — per-frame loop time for monitoring.
- **Wider obstacle dilation default** (4→8 px, 40 cm). Tank/differential
  drive needs more clearance to turn around obstacles than swerve.
- **Skip writing into `bev_rgb` when viz is disabled.** Saves a large
  memory write per frame in race mode.

## Cameras

| Position | Model | Serial   |
|----------|-------|----------|
| Left     | ZED X | 43779087 |
| Front    | ZED X | 42569280 |
| Right    | ZED X | 49910017 |

## Build

```bash
cd ~/your_ros2_ws
colcon build --packages-select avl_bev_perception --symlink-install
source install/setup.bash
```

Python deps:

```bash
pip install opencv-python numpy --break-system-packages
# Only needed if you have a Tier 2 ONNX model:
pip install onnxruntime-gpu --break-system-packages
```

## Run

```bash
# Default: full perception + viz
ros2 launch avl_bev_perception bev_perception.launch.py

# Race mode: perception only, no BGR viz topics
ros2 launch avl_bev_perception bev_perception.launch.py viz_enabled:=false

# Slow viz to 1 Hz, perception to 30 Hz
ros2 launch avl_bev_perception bev_perception.launch.py perc_fps:=30 viz_fps:=1.0

# Disable auto-calibration (use config defaults)
ros2 param set /bev_perception_node segmentation.auto_calibrate false

# View output
ros2 run rqt_image_view rqt_image_view /bev/fused
```

## Topics

### Subscribed (per camera, `<cam>` = `left` | `front` | `right`)

- `/zed_<cam>/zed_node/rgb/image_rect_color`   sensor_msgs/Image
- `/zed_<cam>/zed_node/depth/depth_registered` sensor_msgs/Image
- `/zed_<cam>/zed_node/rgb/camera_info`        sensor_msgs/CameraInfo

### Published — perception loop (default 20 Hz, machine consumable)

- `/bev/segmentation`             mono8   — class IDs (see table below)
- `/bev/drivable_mask`            mono8   — 255 = drivable
- `/bev/obstacle_mask`            mono8   — 255 = obstacle (dilated for safety margin)
- `/bev/lane_lines_detected`      Bool    — true when lanes visible (planner mode flag)
- `/bev/perception_latency_ms`    Float32 — per-frame loop time

### Published — viz loop (default 2 Hz, optional)

- `/bev/image_raw`      bgr8 — RGB BEV with vehicle footprint
- `/bev/fused`          bgr8 — RGB + colorized seg overlay
- `/bev/debug/<cam>`    bgr8 — input camera with seg overlay

## Class set (IGVC)

| ID | Class            | Source                         | Color (BGR)     |
|----|------------------|--------------------------------|-----------------|
| 0  | background       | —                              | (0, 0, 0)       |
| 1  | lane line        | Tier 1a (HSV white)            | (255, 255, 255) |
| 2  | barrel           | Tier 1b (HSV orange)           | (0, 140, 255)   |
| 3  | person           | Tier 2 (ONNX, optional)        | (0, 255, 255)   |
| 4  | pothole          | Tier 1c (Otsu) or Tier 2       | (255, 0, 255)   |
| 5  | drivable area    | Tier 2 (ONNX, optional)        | (0, 180, 0)     |

Tier 1 (HSV + Otsu) runs every frame and handles lanes + barrels + potholes
on its own — fast (≈3 ms total) and reliable on IGVC's color palette.
Auto-calibration adapts white V_min and orange S_min at startup based on
asphalt color. Tier 2 is an optional ONNX hook for harder classes; set
`segmentation.tier2_model_path` in `config/bev_config.yaml` to enable it.

## Integration with the rest of the AVL stack

This package handles **camera-derived BEV perception only**. Other concerns
are owned by separate packages:

### Xsens MTi-680G (IMU + GPS)

The Xsens publishes a fused position/orientation estimate via the
`xsens_mti_ros2_driver` (`/filter/positionlla`, `/filter/orientation`).
**Do not run `robot_localization` to re-fuse this data** — the EKF is
already inside the Xsens. Just consume the filter topics directly.

### Velodyne VLP-16

LiDAR is a separate package (`avl_lidar`). Recommended division of labor:

- **Cameras** (this package) handle: lane lines, barrels, potholes, near-field obstacles
- **LiDAR** handles: ramp detection (height-stratified PCL filter), rear blind-spot obstacles, all-weather backup obstacle layer

### Nav2 wiring

Wire `/bev/obstacle_mask` and `/bev/drivable_mask` into Nav2's local
costmap as separate layers. Example minimal `nav2_params.yaml` snippet:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["bev_obstacle_layer", "inflation_layer"]
      bev_obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: bev_obstacles
        bev_obstacles:
          topic: /bev/obstacle_mask    # convert via your bev_to_pointcloud node
          data_type: "PointCloud2"
          marking: true
          clearing: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.55         # tank drive needs more inflation
```

The planner subscribes to `/bev/lane_lines_detected` to switch behavior
trees between lane-keeping and GPS-waypoint modes for the IGVC offroad
section.

## Configuration

All tunables live in `config/bev_config.yaml` — BEV grid bounds, mount
poses, depth limits, HSV thresholds, auto-cal toggle, perception/viz
rates, obstacle dilation margin, parallel cameras.

If lighting at the venue is unusual, you can:
1. Let auto-cal do its thing (default — no action required), OR
2. Disable auto-cal and tune HSV manually with `tools/calibrate_hsv.py`,
   OR
3. Override individual values via `ros2 param set` without rebuilding.

## What's intentionally NOT in this package

- **LiDAR fusion** (Velodyne VLP-16) — separate `avl_lidar` package
- **IMU/GPS integration** (Xsens MTi-680G) — use the `xsens_mti_ros2_driver` directly; no need for `robot_localization`
- **Path planning / waypoint following** — separate planner package
- **Nav2 costmap inflation** — separate package; consume `/bev/obstacle_mask`

## Performance budget on Jetson AGX Orin (64 GB)

Measured target with v3.2 + parallel cameras + LUTs + viz disabled:

| Stage                       | Latency        |
|-----------------------------|----------------|
| Per-camera HSV+Otsu seg     | ~2–4 ms        |
| Per-camera projection (LUT) | ~1–2 ms        |
| BEV mask derivation         | <1 ms          |
| Total perception loop       | **~10–15 ms**  |

Headroom for >50 Hz, comfortable margin for adding LiDAR fusion later
without giving up reaction time.

See `docs/CHANGELOG_AND_DESIGN_v3.2.docx` for the full design breakdown,
v3→v3.2 changes, and the IGVC 2025 competitive analysis that drove the
v3.2 design choices.
