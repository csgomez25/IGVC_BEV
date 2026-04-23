# avl_bev_perception (v3 — IGVC AutoNav)

3-camera ZED X Bird's Eye View perception tuned for the IGVC AutoNav course.
Hybrid segmentation (HSV + optional ONNX), decoupled perception/viz timers.

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

# View output
ros2 run rqt_image_view rqt_image_view /bev/fused
```

## Topics

### Subscribed (per camera, `<cam>` = `left` | `front` | `right`)

- `/zed_<cam>/zed_node/rgb/image_rect_color`   sensor_msgs/Image
- `/zed_<cam>/zed_node/depth/depth_registered` sensor_msgs/Image
- `/zed_<cam>/zed_node/rgb/camera_info`        sensor_msgs/CameraInfo

### Published — perception loop (default 20 Hz, machine consumable)

- `/bev/segmentation`   mono8 — class IDs (0=bg, 1=lane, 2=barrel, 3=person, 4=pothole, 5=drivable)
- `/bev/drivable_mask`  mono8 — 255 = drivable
- `/bev/obstacle_mask`  mono8 — 255 = obstacle (dilated for safety margin)

### Published — viz loop (default 2 Hz, optional)

- `/bev/image_raw`      bgr8 — RGB BEV with vehicle footprint
- `/bev/fused`          bgr8 — RGB + colorized seg overlay
- `/bev/debug/<cam>`    bgr8 — input camera with seg overlay

## Class set (IGVC)

| ID | Class            | Source             | Color (BGR)     |
|----|------------------|--------------------|-----------------|
| 0  | background       | —                  | (0, 0, 0)       |
| 1  | lane line        | Tier 1 (HSV white) | (255, 255, 255) |
| 2  | barrel           | Tier 1 (HSV orange)| (0, 140, 255)   |
| 3  | person           | Tier 2 (ONNX)      | (0, 255, 255)   |
| 4  | pothole          | Tier 2 (ONNX)      | (255, 0, 255)   |
| 5  | drivable area    | Tier 2 (ONNX)      | (0, 180, 0)     |

Tier 1 (HSV thresholds) runs every frame and handles lane lines + barrels —
fast and reliable on IGVC's color palette. Tier 2 is an optional ONNX hook
for the harder classes; set `segmentation.tier2_model_path` in
`config/bev_config.yaml` to enable it.

## Offroad / waypoint section

The IGVC course has a no-lane-line GPS-waypoint section. The BEV node
publishes the same topics regardless of where on the course you are. The
mode switch lives in the planner: it should ignore `/bev/drivable_mask`
during the waypoint segment and plan straight toward the next GPS goal
using only `/bev/obstacle_mask`. When the lane lines re-appear, the
planner switches back.

## Configuration

All tunables live in `config/bev_config.yaml` — BEV grid bounds, mount
poses, depth limits, HSV thresholds, perception/viz rates, obstacle
dilation margin.

If lighting at the venue changes the white/orange thresholds, override
just the affected HSV values from a launch file or via `ros2 param set`
without rebuilding.

## What's intentionally NOT in this package

- LiDAR fusion (Velodyne VLP-16) — separate package
- IMU integration (Xsens MTi) — separate package
- Path planning / waypoint following — separate package
- Costmap → nav2 inflation — separate package (consume `/bev/obstacle_mask`)

See `docs/CHANGELOG_AND_DESIGN_v3.docx` for the full design breakdown,
v2→v3 changes, and tuning reference.
