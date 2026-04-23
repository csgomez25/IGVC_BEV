# AVL BEV Perception

ROS 2 package for **3-camera ZED X Bird's Eye View perception** with hybrid semantic segmentation, built for an autonomous food-delivery robot competing in **IGVC AutoNav**.

The pipeline takes synchronized RGB + depth from three ZED X cameras, projects everything into a shared top-down grid around the robot, runs IGVC-tuned segmentation, and publishes obstacle / drivable masks for the planner plus visualization images for RViz.

---

## Hardware

| Position | Model | Serial |
|----------|-------|--------|
| Left  | ZED X | 43779087 |
| Front | ZED X | 42569280 |
| Right | ZED X | 49910017 |

Other onboard sensors (Velodyne VLP-16 LiDAR, Xsens MTi-680G IMU) are handled by separate packages and are **not** consumed here.

---

## How it works (the short version)

The node runs **two independent timers** so visualization can never bottleneck the planner:

```
                   ┌────────── ZED X Left ──────────┐
                   ├────────── ZED X Front ─────────┤
                   ├────────── ZED X Right ─────────┤
                   │                                │
                   ▼                                ▼
         ┌───────────────────┐          ┌─────────────────────┐
         │ PERCEPTION TIMER  │          │     VIZ TIMER       │
         │   (20 Hz default) │          │   (2 Hz default,    │
         │                   │          │     optional)       │
         │  • snapshot data  │          │                     │
         │  • run seg        │          │ • colorize seg      │
         │  • project to BEV │ ────►    │ • blend overlay     │
         │  • derive masks   │  shared  │ • draw vehicle      │
         │  • publish mono8  │  state   │ • publish bgr8      │
         └───────────────────┘          └─────────────────────┘
                   │                                │
                   ▼                                ▼
         /bev/segmentation                /bev/image_raw
         /bev/drivable_mask               /bev/fused
         /bev/obstacle_mask               /bev/debug/<cam>
                   │                                │
                   ▼                                ▼
              [ Planner ]                       [ RViz ]
```

**Segmentation is two-tier:**

- **Tier 1 — HSV thresholds (always on, ~1–3 ms/frame).** Detects white lane lines and orange barrels using OpenCV color thresholds + morphology. Bulletproof for IGVC's color palette, no model required.
- **Tier 2 — ONNX model (optional, off by default).** Hook for a learned model that handles people, potholes, and drivable-area classification. Enable by setting `segmentation.tier2_model_path` in the config.

**Class set (output of `/bev/segmentation` as mono8 class IDs):**

| ID | Class | Source |
|----|-------|--------|
| 0 | background | — |
| 1 | lane line | Tier 1 (HSV white) |
| 2 | barrel | Tier 1 (HSV orange) |
| 3 | person | Tier 2 (ONNX) |
| 4 | pothole | Tier 2 (ONNX) |
| 5 | drivable area | Tier 2 (ONNX) |

**Pothole-friendly height filter.** IGVC potholes are flat painted circles, not real holes. Classes that live on the ground plane (lane lines, potholes) are exempt from the lower height filter so they survive the projection.

---

## Topics

### Subscribed (per camera, `<cam>` ∈ `left | front | right`)

```
/zed_<cam>/zed_node/rgb/image_rect_color    sensor_msgs/Image
/zed_<cam>/zed_node/depth/depth_registered  sensor_msgs/Image
/zed_<cam>/zed_node/rgb/camera_info         sensor_msgs/CameraInfo
```

If your ZED launch publishes under different names, edit `cam_defs` at the top of `_setup_cameras()` in `bev_perception_node.py`.

### Published — perception loop (default 20 Hz, machine-consumable)

```
/bev/segmentation     mono8   class IDs 0–5 per BEV cell
/bev/drivable_mask    mono8   255 = drivable (Tier 2 only)
/bev/obstacle_mask    mono8   255 = obstacle, dilated for safety
```

### Published — viz loop (default 2 Hz, optional)

```
/bev/image_raw        bgr8    RGB BEV with vehicle footprint
/bev/fused            bgr8    RGB + colorized seg overlay
/bev/debug/<cam>      bgr8    Per-camera input + seg overlay
```

---

## Build & run

```bash
# 1. Drop the package into your workspace
cd ~/your_ros2_ws/src
# (place the avl_bev_perception/ folder here)

# 2. Build
cd ..
colcon build --packages-select avl_bev_perception --symlink-install
source install/setup.bash

# 3. Python deps
pip install opencv-python numpy --break-system-packages
# Only needed if you have a Tier 2 ONNX model:
pip install onnxruntime-gpu --break-system-packages

# 4. Run (start your ZED launch in another terminal first)
ros2 launch avl_bev_perception bev_perception.launch.py
```

### Launch arguments

| Argument | Default | Effect |
|----------|---------|--------|
| `seg_enabled` | `true`  | Toggle segmentation entirely. |
| `viz_enabled` | `true`  | False = no BGR images on the network (race mode). |
| `perc_fps`    | `20.0`  | Perception loop rate in Hz. |
| `viz_fps`     | `2.0`   | Viz loop rate in Hz. |
| `use_rviz`    | `false` | Auto-open RViz with the bundled layout. |

```bash
# Race mode — perception runs full speed, no viz traffic
ros2 launch avl_bev_perception bev_perception.launch.py viz_enabled:=false

# Debug mode — RViz on, slow viz, full perception
ros2 launch avl_bev_perception bev_perception.launch.py use_rviz:=true viz_fps:=1.0

# Inspect outputs
ros2 run rqt_image_view rqt_image_view /bev/fused
ros2 topic hz /bev/obstacle_mask
```

---

## Configuration

All tunables live in [`avl_bev_perception/config/bev_config.yaml`](avl_bev_perception/config/bev_config.yaml). The most-changed knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `bev.resolution` | `0.05` | Meters per BEV pixel. |
| `bev.x_range`, `y_range` | `[-10, 15]`, `[-10, 10]` | BEV extent in meters. |
| `depth.max` | `12.0` | Drop far depth noise. |
| `perception.fps` | `20.0` | Planner update rate. |
| `viz.fps` | `2.0` | RViz update rate. |
| `output.obstacle_dilate_px` | `4` | Safety margin around obstacles (≈20 cm at 5 cm/px). |
| `segmentation.hsv.white.v_min` | `180` | Raise if shadows are misdetected as lane lines. |
| `segmentation.hsv.orange.s_min` | `130` | Raise if asphalt is being false-positived; lower if barrels look faded. |

### Tuning HSV at the venue

Lighting changes (overcast, wet pavement, sunrise/sunset) will shift the white and orange thresholds. There's an interactive calibration tool in [`avl_bev_perception/tools/calibrate_hsv.py`](avl_bev_perception/tools/calibrate_hsv.py):

```bash
# Against a saved still image
python3 avl_bev_perception/tools/calibrate_hsv.py path/to/igvc_scene.jpg

# Against a live camera feed
python3 avl_bev_perception/tools/calibrate_hsv.py --topic /zed_front/zed_node/rgb/image_rect_color
```

Drag the trackbars until lines / barrels are cleanly highlighted, press `p` to print the YAML snippet, paste into `bev_config.yaml`, restart the node.

---

## IGVC course behavior

- **Lane-line sections.** Planner uses `/bev/obstacle_mask` (which includes lane lines) as soft walls and stays in the corridor.
- **GPS-waypoint sections.** No lane lines visible. The mode switch lives **in the planner**, not here — this node publishes the same topics regardless. Planner ignores `/bev/drivable_mask`, plans straight toward the next GPS waypoint, uses `/bev/obstacle_mask` for collision avoidance only.
- **Ramp.** The depth-based BEV will likely flag the upslope as elevated terrain. Either widen `bev.height_range` for that GPS zone, or rely on LiDAR + IMU to handle ramp navigation. Known limitation; flagged in `docs/CHANGELOG_AND_DESIGN_v3.docx`.
- **Colored navigation dots (if used as gates).** Tier 1 only handles white and orange. If the course uses red/blue/green/yellow gates the robot must hit in order, additional HSV bands need to be added to `_infer_tier1` — the architecture supports it cleanly.

For the full design breakdown, timer architecture, projection math, and tuning reference, see [`avl_bev_perception/docs/CHANGELOG_AND_DESIGN_v3.docx`](avl_bev_perception/docs/CHANGELOG_AND_DESIGN_v3.docx).

---

## Repo layout

```
.
├── README.md                          ← you are here
├── LICENSE
├── .gitignore
└── avl_bev_perception/                ← the ROS 2 package
    ├── README.md                      ← in-package quick reference
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── avl_bev_perception/
    │   ├── __init__.py
    │   ├── bev_perception_node.py     ← main node (two timers, projection, masks)
    │   └── seg_inference.py           ← Tier 1 HSV + Tier 2 ONNX engine
    ├── config/
    │   └── bev_config.yaml            ← all tunables
    ├── launch/
    │   └── bev_perception.launch.py
    ├── rviz/
    │   └── bev_perception.rviz        ← RViz layout
    ├── tools/
    │   └── calibrate_hsv.py           ← interactive HSV tuner
    ├── docs/
    │   └── CHANGELOG_AND_DESIGN_v3.docx
    └── resource/
        └── avl_bev_perception
```

---

## Pushing this to GitHub

If you don't already have a repo:

```bash
cd avl_bev_perception_repo

# Create the repo on github.com first (empty, no README/license/gitignore — we have ours).

git init
git add .
git commit -m "Initial commit: avl_bev_perception v3 (IGVC AutoNav)"
git branch -M main
git remote add origin git@github.com:<your-username>/<your-repo-name>.git
git push -u origin main
```

If the repo already exists and you're updating it:

```bash
cd avl_bev_perception_repo
git add .
git commit -m "Add v3 IGVC perception build"
git push
```

> **Note:** the package itself lives under `avl_bev_perception/` inside this repo. If you'd rather keep your colcon workspace structure flat, you can move just the inner `avl_bev_perception/` folder into your `src/` directory — it's a self-contained ROS 2 package.

---

## What this package does **not** do

These belong in separate packages and are intentionally out of scope:

- **LiDAR fusion** (Velodyne VLP-16)
- **IMU integration** (Xsens MTi-680G)
- **Path planning** / waypoint following / mode switching between lane and GPS sections
- **Costmap inflation** for nav2 — though `/bev/obstacle_mask` is in the right format to feed one

---

## License

MIT. See `LICENSE`.
