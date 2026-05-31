# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This repo contains a single ROS 2 (ament_python) package, **`avl_bev_perception` v3.2**, nested one folder deep under [avl_bev_perception_v3_2/avl_bev_perception/](avl_bev_perception_v3_2/avl_bev_perception/). When the user references "the package" or "the node," they mean this one. Everything below paths from there.

It is the **camera-derived BEV perception** layer for an IGVC AutoNav robot: 3× ZED X cameras → shared top-down grid → hybrid HSV/Otsu/(optional) ONNX segmentation → `/bev/*` topics for the planner. LiDAR fusion, IMU/GPS, and path planning live in separate, out-of-tree packages and are explicitly out of scope here. See [Integration with Parsa's Stack](#integration-with-parsas-stack-referencesparsa_igvc) below for the planner/controller/sensor stack this is meant to feed into.

## Build / run

The package is colcon-built, ament_python. There is no test suite, no linter wired up, no CI in this repo — don't fabricate commands for those.

```bash
# From a ROS 2 workspace that contains this package under src/
colcon build --packages-select avl_bev_perception --symlink-install
source install/setup.bash

# Bring up cameras (serial-pinned) and the perception node in separate terminals:
ros2 launch avl_bev_perception zed_cameras.launch.py
ros2 launch avl_bev_perception bev_perception.launch.py

# Useful flags:
#   viz_enabled:=false   race mode — drops all BGR viz topics
#   perc_fps:=30 viz_fps:=1.0
#   use_rviz:=true       opens the bundled rviz/bev_perception.rviz layout
```

Python deps (install with `--break-system-packages` on Jetson): `opencv-python`, `numpy`, and only if a Tier 2 model is configured, `onnxruntime-gpu`.

Target hardware is **Jetson AGX Orin**. Parallel per-camera segmentation (`perception.parallel_cameras: true`) assumes Orin's 12-core CPU and is a load-bearing assumption for the latency budget.

## Architecture — the parts you can't see from one file

**Two timers, one shared state.** [`bev_perception_node.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) runs a perception timer (default 20 Hz) that produces the machine-consumable masks (`/bev/segmentation`, `/bev/drivable_mask`, `/bev/obstacle_mask`, `/bev/lane_lines_detected`, `/bev/perception_latency_ms`), and a separate, optional viz timer (default 2 Hz) that produces BGR overlays. The viz loop reads from `self._latest_*` snapshots under `_latest_lock` so it can never block perception. **Don't make the viz path do work the perception path depends on.**

**Per-camera projection LUTs.** When `CameraInfo` arrives, [`_build_projection_lut()`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) precomputes `(u-cx)/fx`, `(v-cy)/fy`, the downsampled meshgrid, and `cos/sin(mount_yaw)` once and caches them on `CameraState`. The projection hot path **must** use those cached factors — recomputing per-frame regresses ~5–10 ms/camera and was a v3.2 explicit fix.

**Ground-plane class exemption.** `GROUND_PLANE_CLASSES = (CLASS_LANE_LINE, CLASS_POTHOLE)` is exempted from the lower height filter in the projection so painted-flat features survive. Any new "painted on asphalt" class belongs in that tuple.

**Tier 1 / Tier 2 segmentation.** [`seg_inference.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/seg_inference.py): Tier 1 is HSV (white lanes, orange barrels) + Otsu bright pass (potholes), always on, ~3 ms. Tier 2 is an optional ONNX hook gated by `segmentation.tier2_model_path`. Tier 1 wins on overlap because it's more reliable for the colors it covers. Class IDs (`CLASS_BACKGROUND..CLASS_DRIVABLE`) are duplicated as fallbacks at the top of `bev_perception_node.py` so the node still loads if `seg_inference` fails to import — keep both lists in sync if you renumber.

**Auto-HSV calibration at startup.** [`auto_calibrate.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/auto_calibrate.py) samples the lower 30% of each camera frame for ~30 frames, derives venue-tuned white V_min and orange S_min, validates the result covers <40% of the image (else falls back to config defaults), then pushes thresholds into the seg engine via `set_hsv_thresholds()`. Runs once per node lifetime; perception keeps ticking during the warm-up.

**Strict serial-to-namespace binding.** [`zed_cameras.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/zed_cameras.launch.py) hard-pins ZED X serials to namespaces (`zed_left`, `zed_front`, `zed_right`). This is intentional: the wrapper's default enumeration is order-of-USB-plug, which silently mis-maps cameras and produces a stitched-but-geometrically-wrong BEV. If a camera is missing, fix the cable, not the binding.

## Authoritative source of truth for camera serials and mount poses

As of v3.2.2 (2026-05-13), the serial mapping is aligned with Parsa's `IGVC_ROS2` stack ("Verified 2026-04-24 via per-port enumeration"). All four sources of truth now agree — [`config/bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml), [`launch/zed_cameras.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/zed_cameras.launch.py), [`launch/tf_static.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/tf_static.launch.py), and both READMEs:

| Position | Serial   |
|----------|----------|
| Left     | 43779087 |
| Front    | 42569280 |
| Right    | 49910017 |

The node also runs a serial sanity check at startup ([bev_perception_node.py:_check_zed_serials](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py)): if `pyzed` is importable, it logs the real connected serials and warns on any that don't match the expected mapping; otherwise it falls back to a 5-second `CameraInfo`-presence timer. Warns, never errors — a miswired camera produces loud logs but still lets the node come up.

Mount poses in [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) are derived from a team measurement sketch (inches, sketch axes X=right Y=forward Z=up) and converted to REP-103 (X=forward, Y=left, Z=up, meters). They're keyed by position, not serial, so the v3.2.2 left/right serial swap did **not** change them — but if anyone physically swapped cameras between mounts before to "fix" the old wrong mapping, the robot may now have left/right cameras pointing the wrong way. Diagnose by viewing each `/zed_<cam>/...` feed.

## Tuning vs. coding

Most user requests on this codebase are **parameter tuning**, not code changes. Before editing a `.py` file, check whether the knob already exists in [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) — every threshold, rate, ROI, and mount offset is exposed as a ROS param and can be overridden at runtime with `ros2 param set`. Add new parameters via `_declare_params()` and the YAML; don't hard-code.

The interactive HSV tuner [`tools/calibrate_hsv.py`](avl_bev_perception_v3_2/avl_bev_perception/tools/calibrate_hsv.py) is the manual override path when auto-calibration is disabled.

## Files Claude should not invent

- No `tests/` directory exists. Don't claim there are tests; don't run `pytest` and report results. If asked to add tests, ask which framework.
- No CI workflows exist in this repo. Don't reference `.github/workflows/...`.
- The design docs are `.docx` files under [`avl_bev_perception/docs/`](avl_bev_perception_v3_2/avl_bev_perception/docs/) and are not readable by your text tools — refer the user to them, don't try to summarize their contents.

---

## Integration with Parsa's Stack ([references/parsa_igvc](../references/parsa_igvc/))

[`references/parsa_igvc/`](../references/parsa_igvc/) is a **read-only checkout** of Parsa Ghasemi's `IGVC_ROS2` — the team's full autonomous-vehicle stack that this BEV package is intended to plug into. **Do not modify any file under `references/parsa_igvc/`.** Treat it as documentation of the deployment target.

> For known kiwicampus / costmap failure modes and the bring-up validation checks, see [KIWICAMPUS_DEBUG.md](KIWICAMPUS_DEBUG.md) at the repo root.

### What Parsa's stack does end-to-end

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              Sensors (Jetson Orin)                             │
│  Velodyne VLP-16  •  ZED X cameras  •  Xsens MTi-680G (IMU+GNSS+RTK via NTRIP) │
└────────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼─────────────────────────────────┐
        ▼                              ▼                                 ▼
  /velodyne_points              /zed_<cam>/zed_node/...           /imu/data, /gnss
        │                              │                                 │
        │                       avros_perception                         │
        │                  (single-camera, no fusion):                   │
        │                  RGB+cloud → stub/HSV/sooner25 →               │
        │                  /perception/<cam>/semantic_*                  │
        │                              │                                 ▼
        │                              │                  robot_localization
        │                              │                  (dual EKF: local
        │                              │                  ekf_odom publishes
        │                              │                  odom→base_link; global
        │                              │                  ekf_map publishes
        │                              │                  map→odom, fed by
        │                              │                  navsat_transform)
        │                              │                                 │
        ▼                              ▼                                 ▼
  Nav2 local + global           Nav2 local_costmap:              /odometry/filtered
  costmaps: STVL                kiwicampus                       (used by Nav2 +
  (spatio_temporal_              semantic_segmentation_           controller)
   voxel_layer) — global         _layer (camera-cost)
   migrated off ObstacleLayer
   2026-05-19 (decay-based,
   bounded accumulation)
        │                              │                                 │
        └──────────────┬───────────────┘                                 │
                       ▼                                                 │
                Nav2 planner: NavfnPlanner (Dijkstra, holonomic)         │
                  (was SmacPlannerHybrid/DUBIN; reverted because         │
                   2.31 m turn radius can't fit IGVC's 2-3 m lanes       │
                   and tracked diff-drive has 0 m radius anyway)         │
                Nav2 controller: nav2_mppi_controller (Humble prod —     │
                  batch 500 × 56 steps × 50 ms; vx_max=1.5 (0.7→1.5      │
                  2026-05-30), vx_min=-0.6, wz_max=1.9. vx_max read at    │
                  configure-time — live param set is a no-op, edit YAML)  │
                  (RPP only in Jazzy fallback config)                    │
                Nav2 BT (Humble prod): navigate_igvc_autonav_humble      │
                  (recovery BT, default since 2026-05-21) — pt-to-pt     │
                Nav2 route_server (Jazzy fallback only): GeoJSON campus  │
                  graph, cpp_campus_graph.geojson, 52 nodes / 113 edges  │
                       │                                                 │
                       ▼                                                 │
                  /cmd_vel  ──→  avros_control/actuator_node  ◄──────────┘
                                  (slew-rate + IMU heading-hold +
                                   diff-drive inverse → Teensy serial)
                                                │
                                                ▼
                                   Teensy 4.1 → CAN → SparkMAX
                                   velocity PID → NEO brushless
                                   → 12.75:1 gearbox → track drive
```

GPS goal in (`/navigate_to_pose` action) → Humble: NavfnPlanner builds a direct point-to-point path, no route graph (Jazzy fallback uses route_server with the campus graph) → MPPI controller (Humble) / RPP (Jazzy) follows path → STVL costmaps (LiDAR voxels + camera-classified obstacles via kiwicampus, both local *and* global since the 2026-05-19 migration) do collision avoidance → cmd_vel drives the diff-drive tracks. WebUI is bench-test only; e-stop is a topic.

### Does he already have perception/segmentation/BEV?

**Yes on segmentation, no on BEV.** [`avros_perception`](../references/parsa_igvc/src/avros_perception/) has a `perception_node` with three pipelines selectable via `pipeline:` param:

- **`stub`** — paints a vertical class stripe for plumbing tests
- **`hsv`** — per-class HSV thresholds for `lane_white` / `barrel_orange` / `pothole` with Sooner-2023 box-blur, iscumd adaptive V-floor, and a sky ROI polygon. Conceptually the same as my Tier 1a/1b.
- **`sooner25`** — Sooner Robotics 2025 winning approach: single inverted threshold that matches asphalt and inverts so paint + saturated objects become obstacles. Single-class output. I don't have an equivalent. **As of 2026-05-28 this is his DEFAULT production pipeline** (`pipeline: 'sooner25'` in `perception.yaml`), switched off `hsv` because per-class HSV kept misclassifying bright concrete/asphalt at the IGVC practice course. His pothole HSV is neutered (V-floor 250). **Consequence: his production mask is single-class — it cannot tell a barrel from a lane line.** That matters for the new soft-lane plan (see § Lane-cost change below).
- `onnx` — planned (Phase 6), not implemented.

**What he does NOT have:**
- No top-down BEV mosaic across multiple cameras. Currently only `front` runs in production; `left.yaml` / `right.yaml` are staged for Phase 5.
- No Otsu pothole pass (Tier 1c in my package).
- No auto-HSV calibration at startup (Twistopher-style).
- No 3-camera projection / stitching.
- No `vehicle frame → BEV pixel` projection in his node — he hands the camera frame's organized cloud + 2D mask to the `kiwicampus/semantic_segmentation_layer` Nav2 plugin, which does the projection by raytracing cloud points and writing into the local costmap.

**The gap my package fills:** multi-camera fusion + BEV mosaic + auto-cal + the wider obstacle dilation and ground-plane class exemption tuned for IGVC. Net: he has *what to detect per camera*, I have *the unified top-down picture across all three cameras*.

### Topic / message contract — what my package would need to publish for his planner

His Nav2 stack does **not** consume my current `/bev/*` mono8 grids directly. The hook into Nav2 is the **kiwicampus `semantic_segmentation_layer` contract**, which is defined per-camera and is what [`perception_node.py:79-235`](../references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py) currently produces:

| Topic (per camera `<cam>`)              | Type                       | QoS              | Notes |
|------------------------------------------|----------------------------|------------------|-------|
| `/perception/<cam>/semantic_mask`        | `sensor_msgs/Image` mono8  | sensor_data      | H×W = organized-cloud H×W, class IDs from `class_map.yaml` |
| `/perception/<cam>/semantic_confidence`  | `sensor_msgs/Image` mono8  | sensor_data      | same H×W, 0–255 |
| `/perception/<cam>/semantic_points`      | `sensor_msgs/PointCloud2`  | sensor_data      | **organized** (`height > 1`), same H×W as mask, relayed from ZED's `point_cloud/cloud_registered` with stamp rewritten |
| `/perception/<cam>/label_info`           | `vision_msgs/LabelInfo`    | **transient_local + reliable, depth=1** | Latched once at startup. Late-joining costmap layer reads it for ID↔name mapping. |

**Hard constraints** (all gates inside kiwicampus that silently drop your data if violated):
1. `mask`, `confidence`, and `cloud` MUST share `header.stamp` exactly. kiwicampus message-filter-syncs them. (`perception_node` uses `max(image_stamp, cloud_stamp)` — see [`perception_node.py:343-348`](../references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py).)
2. Mask H×W MUST equal cloud H×W. He resizes BGR to cloud shape *before* running the pipeline; if you keep your full-res BEV approach, you'd need a separate per-camera path emitting a downsampled mask matched to cloud H×W (the cloud is `point_cloud_res: COMPACT` → 256×448 on ZED X HD1080).
3. `LabelInfo` MUST be published with `transient_local + reliable` QoS so a late-joining layer gets the latched message.
4. The class IDs in your `LabelInfo` must match exactly what your mask paints, and every class name must be listed under one of the `class_types: [...]` blocks in [`nav2_params_humble.yaml`](../references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml) (`danger` for marking, `ignored` for `free`/`unknown`). Names not declared are silently ignored.

So integration paths (in order of effort):
- **A. Adapter** ✅ **Implemented as the `kiwicampus.enabled` gate (see § Path A — Kiwicampus adapter below).** No separate node — the per-camera publishers are inline in `bev_perception_node.py` and reuse the existing Tier 1 segmentation pass. kiwicampus + Nav2 stay untouched.
- **B. Costmap shim** — write a `bev_to_pointcloud` node that converts `/bev/obstacle_mask` to a PointCloud2 and feed it into an STVL `observation_sources` entry in [`nav2_params_humble.yaml`](../references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml) (both costmaps use STVL as of the 2026-05-19 migration; plain ObstacleLayer is gone). Loses semantic info but keeps the BEV mosaic.
- **C. Replace** — emit OccupancyGrid and swap in a `StaticLayer`-style costmap plugin. Highest effort, biggest deviation from his current stack.

### Path A — Kiwicampus adapter (implemented)

Single-param gate in [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) (`kiwicampus.enabled`, default `false`). When `true`, the same node *also* publishes the per-camera kiwicampus contract alongside the existing `/bev/*` outputs — no extra process, no second inference pass, no perturbation of the standalone path.

**Usage**
```bash
# Standalone (default — unchanged behavior):
ros2 launch avl_bev_perception bev_perception.launch.py

# Joint with Parsa's Nav2 stack:
ros2 launch avl_bev_perception bev_perception.launch.py \
    --ros-args -p kiwicampus.enabled:=true
# (or set kiwicampus.enabled: true in bev_config.yaml)
```

**What was added**

- **Param** `kiwicampus.enabled` (bool, default `false`) — the master gate.
- **Param** `cameras.<cam>.cloud_topic` per camera (defaults to ZED v5's `/zed_<cam>/zed_node/point_cloud/cloud_registered`).
- **Subscriber** (per camera, only when enabled): the organized `PointCloud2` from the ZED, latest-stashed on `CameraState.cloud`.
- **Publishers** (per camera, only when enabled): `/perception/<cam>/semantic_mask` (mono8), `/perception/<cam>/semantic_confidence` (mono8), `/perception/<cam>/semantic_points` (PointCloud2 relay), `/perception/<cam>/label_info` (`vision_msgs/LabelInfo`, **latched** with `transient_local + reliable, depth=1`).
- **Helper** `_publish_kiwicampus()` in [`bev_perception_node.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py): resizes the per-camera Tier 1 mask to cloud H×W with `INTER_NEAREST`, derives confidence as `(mask != background & != unknown) * 255`, and publishes all three streaming messages with a shared `header.stamp = max(rgb_stamp, cloud_stamp)` (the same trick Parsa uses to keep his `ApproximateTimeSynchronizer` happy).
- **Class-name table** `_KIWI_CLASS_NAMES` mapping our IDs to the names kiwicampus's `class_types: [...]` lists declare: `0=free, 1=lane_white, 2=barrel_orange, 3=pothole, 4=person, 5=drivable, 255=unknown`. IDs 0–3 + 255 match Parsa's [`class_map.yaml`](../references/parsa_igvc/src/avros_perception/config/class_map.yaml) verbatim; IDs 4 and 5 are local extensions that kiwicampus silently ignores (intended).
- **Dependency** `<exec_depend>vision_msgs</exec_depend>` added to [`package.xml`](avl_bev_perception_v3_2/avl_bev_perception/package.xml). The `vision_msgs` import is lazy (inside `_init_kiwicampus_adapter`), so standalone builds without the apt package still load.
- **CameraState fields**: `rgb_stamp`, `rgb_frame_id`, `cloud`, `got_cloud` — populated by the existing RGB callback (stamp/frame stashed unconditionally; cheap) and the new cloud callback.
- **Startup banner line** `Kiwicampus : ON / OFF` so it's obvious from logs which mode you booted in.

**What changed (non-additive)**

- `_rgb_callback` now also stashes `msg.header.stamp` and `msg.header.frame_id` on `CameraState`. Unconditional — overhead is one field assignment under a lock we were already holding.
- `_perception_callback`'s lock-protected snapshot block also captures `(rgb_stamp, rgb_frame_id, cloud)` when the gate is on and a cloud has arrived. Cameras without an organized cloud skip the per-camera publish but still feed the standalone BEV mosaic normally — degraded operation, not a hard error.

**What was removed**

- Nothing. The standalone path (`/bev/segmentation`, `/bev/drivable_mask`, `/bev/obstacle_mask`, `/bev/lane_lines_detected`, `/bev/perception_latency_ms`, viz topics, auto-cal) is byte-for-byte unchanged when `kiwicampus.enabled: false`.

**How it connects**

```
                  RGB ──► seg_engine.infer() ──► seg_masks[<cam>] ──┬──► /bev/* mosaic (existing)
                                                                    │
                                                                    └──► (if kiwicampus.enabled)
                                                                         resize → INTER_NEAREST → cloud H×W
                                                                         + confidence = (mask>0 & !=255) * 255
                                                                         + relay cloud_msg (restamp)
                                                                         + stamp = max(rgb, cloud)
                                                                         ▼
                            /perception/<cam>/semantic_mask        ─┐
                            /perception/<cam>/semantic_confidence  ─┤── kiwicampus
                            /perception/<cam>/semantic_points      ─┤   ApproximateTimeSync
                            /perception/<cam>/label_info (latched) ─┘   → STVL local_costmap
                                                                          (and global, post 2026-05-19)
```

**Validation steps when bringing this up jointly**

1. Launch with `kiwicampus.enabled:=false` → confirm `ros2 topic list | grep perception` returns nothing (only `/bev/*`).
2. Flip to `true`, relaunch → confirm all four `/perception/<cam>/*` topics appear and `ros2 topic echo --once /perception/front/label_info` returns 7 entries.
3. `ros2 topic hz /perception/front/semantic_mask` should match `perception.fps` (default 20 Hz). Cameras without cloud only log a throttled warning; they do not crash the node.
4. `ros2 topic echo --no-arr /perception/front/semantic_mask | head -10` and `/perception/front/semantic_points`: stamps must agree exactly.
5. Bring up Parsa's Nav2 — the kiwicampus layer prints `received N classes` on its first LabelInfo. Local costmap should show painted lethal cells where lanes/barrels were detected.

### Topics I'd need to subscribe to from Parsa's stack

His ZED wrapper is **v5.x**, which **uses different topic names than my code currently subscribes to**:

| What I currently subscribe to (v4 paths)           | What Parsa publishes (v5 paths)                          |
|----------------------------------------------------|----------------------------------------------------------|
| `/zed_<cam>/zed_node/rgb/image_rect_color`         | `/zed_<cam>/zed_node/rgb/color/rect/image`               |
| `/zed_<cam>/zed_node/rgb/camera_info`              | `/zed_<cam>/zed_node/rgb/color/rect/camera_info`         |
| `/zed_<cam>/zed_node/depth/depth_registered`       | same — `/zed_<cam>/zed_node/depth/depth_registered` (still v5-valid) |
| —                                                  | `/zed_<cam>/zed_node/point_cloud/cloud_registered` (organized, what kiwicampus uses) |

My package's `cam_defs` in [`bev_perception_node.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) will need topic-path updates (or expose them as ROS params) before deploying alongside his launch. See `Known Issues & Fixes` entry in [Parsa's CLAUDE.md](../references/parsa_igvc/CLAUDE.md) under "ZED v5 topic names differ from v4."

Other topics worth knowing:

| Topic                  | Type                       | Use case for my package |
|------------------------|----------------------------|--------------------------|
| `/odometry/filtered`   | `nav_msgs/Odometry`        | EKF fused pose (odom frame) — only needed if I want to project history / accumulate map |
| `/imu/data`            | `sensor_msgs/Imu` @ 100 Hz | not currently consumed; available if I add yaw-rate-based de-roll |
| `/tf`, `/tf_static`    | `tf2_msgs/TFMessage`       | If I switch from YAML mount-pose params to TF lookups (`base_link → zed_<cam>_left_camera_frame_optical`) the URDF in [`avros_bringup`](../references/parsa_igvc/src/avros_bringup/) is the source of truth |

### Custom message packages I'd need to add to package.xml

Only one new dep is required to participate in the kiwicampus contract:

- **`vision_msgs`** — provides `vision_msgs/LabelInfo`. Available via apt as `ros-humble-vision-msgs`. Add `<exec_depend>vision_msgs</exec_depend>` to my [`package.xml`](avl_bev_perception_v3_2/avl_bev_perception/package.xml).

Everything else I'd need (`sensor_msgs`, `std_msgs`, `cv_bridge`, `message_filters`) is either already declared or implicit. I do **not** need any of Parsa's custom packages — `avros_msgs` (`ActuatorCommand`, `ActuatorState`, `PlanRoute`) is for the actuator/teleop/route paths, not perception.

### TF tree / coordinate frame setup

```
map                                    ← ekf_filter_node_map (global EKF;
                                         GPS-fed via navsat_transform_node,
                                         which has broadcast_cartesian_transform:
                                         false to avoid a TF loop)
 └── odom                              ← ekf_filter_node_odom (local EKF)
      └── base_link                    ← robot_state_publisher (URDF)
           ├── imu_link                ← static, IMU mount on chassis
           ├── velodyne                ← static, LiDAR mount
           ├── zed_<cam>_camera_link   ← static (via zed_macro.urdf.xacro)
           │    ├── zed_<cam>_camera_center
           │    ├── zed_<cam>_left_camera_frame
           │    │    └── zed_<cam>_left_camera_frame_optical   ← image / cloud frame_id
           │    ├── zed_<cam>_right_camera_frame
           │    │    └── zed_<cam>_right_camera_frame_optical
           │    └── zed_<cam>_imu_link
           └── base_footprint          ← static
```

REP-103 (`X` forward, `Y` left, `Z` up, meters) — matches the convention my [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) already uses for mount poses. The image / cloud `frame_id` is `zed_<cam>_left_camera_frame_optical` (optical-frame convention: Z forward, X right, Y down). My current code projects via mount-pose params and ignores TF; that works as long as the YAML poses agree with the URDF.

**TF source of truth:** [`src/avros_bringup/urdf/avros.urdf.xacro`](../references/parsa_igvc/src/avros_bringup/urdf/) plus `zed_macro.urdf.xacro` from the ZED wrapper. Sensor mount positions in his URDF are flagged as "TODO measure" in his [`TODO.md`](../references/parsa_igvc/TODO.md) — they may not yet agree with my [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) measurements.

### His perception vs. mine — concrete differences

| Aspect                         | My `avl_bev_perception`                            | Parsa's `avros_perception`                              |
|--------------------------------|-----------------------------------------------------|----------------------------------------------------------|
| Camera count in production     | 3 (left + front + right)                            | 1 (front) in `perception_node`; **left + right `semantic_layer` sources now configured in `nav2_params_humble.yaml`** (2026-05-29) but not yet produced — the costmap is staged for 3-cam, the node isn't |
| Output topology                | Shared top-down BEV grid, 3 cameras fused           | Per-camera mask + cloud + LabelInfo; no fusion           |
| Output topic shape             | `/bev/*` mono8 grids (segmentation, drivable, obstacle) | `/perception/<cam>/semantic_{mask,confidence,points}` + `label_info` |
| Downstream consumer            | Custom — assumes planner reads occupancy grids      | Nav2 `kiwicampus/semantic_segmentation_layer` plugin     |
| HSV pipeline                   | Tier 1a (white) + Tier 1b (orange) + Tier 1c (Otsu pothole) | Per-class (lane_white, barrel_orange, pothole) + adaptive V-floor + sky ROI poly |
| Alternate pipeline             | Tier 2 ONNX hook (planned, off by default)          | `sooner25` inverted-threshold (asphalt match → invert) — **now his DEFAULT, single-class** |
| Auto-HSV calibration           | Yes, Twistopher-style at startup                    | No                                                       |
| Projection method              | Depth image + per-camera LUT cache, vehicle-frame mosaic | Hands ZED's organized cloud to kiwicampus; layer does the projection via raytracing |
| Sync                           | None — perception timer snapshots latest RGB/depth/info | `message_filters.ApproximateTimeSynchronizer` on RGB + cloud (slop 0.02 s, queue 2) |
| Resolution                     | Full image, downsampled by `perception.downsample_depth` (default 2) | Cloud-resolution (resizes BGR to cloud H×W) — full-res mode is opt-in |
| Class set                      | 7 classes: 0=bg, 1=lane, 2=barrel, **3=pothole, 4=person**, 5=drivable, 255=unknown (v3.2.2 — IDs 0–3+255 mirror Parsa's) | 4 classes: 0=free, 1=lane_white, 2=barrel_orange, 3=pothole, 255=unknown |
| Live param updates             | Not implemented for thresholds                      | Yes — `on_set_parameters_callback` allowlists pipeline params; `ros2 param set` works at runtime |
| Latency target                 | ~10–15 ms perception loop on Orin                   | not measured in his docs                                 |

### Conflicts that need reconciliation before joint deployment

**Status as of v3.2.2 (2026-05-13):** items 1–3 below were the original conflicts; the resolved ones stay listed for institutional memory.

1. ~~**Camera serial mapping disagrees.**~~ ✅ **Resolved in v3.2.2** — adopted Parsa's per-port-verified mapping verbatim (Left=43779087, Front=42569280, Right=49910017). Startup serial sanity check added in [`_check_zed_serials()`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py). Hardware caveat: if cameras were physically rearranged before to compensate for the old wrong mapping, the robot may now have left/right pointing the wrong way — confirm with a live feed before competition.

2. ~~**ZED topic paths v4 vs v5.**~~ ✅ **Resolved in v3.2.2** — topic paths are ROS params now (`cameras.<cam>.{rgb,depth,info}_topic` in [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml)) defaulting to v5.x. v4.x bringups override per camera in YAML or with `ros2 param set` before the node starts.

3. ~~**Class ID mismatch.**~~ ✅ **Resolved in v3.2.2** for IDs 0–3 + 255, which now mirror Parsa's [`class_map.yaml`](../references/parsa_igvc/src/avros_perception/config/class_map.yaml). The mask can publish to his kiwicampus contract directly. Local extensions (4=person, 5=drivable from Tier 2 ONNX) are not in his class_map; his costmap will silently ignore them, which is fine since Tier 2 is off.

4. **`publish_tf: false`** — Parsa's launch sets this on every ZED so `robot_localization` owns `odom→base_link` exclusively. My package doesn't assume ZED publishes TF, but my standalone [`tf_static.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/tf_static.launch.py) publishes `base_link → zed_<cam>_camera_center` directly. Parsa's URDF (via `zed_macro.urdf.xacro`) publishes those same frames as children of `zed_<cam>_camera_link`. **If both stacks come up together, `tf_static` will collide on the `_camera_center` parent.** Fine for vision-only standalone runs; gate `tf_static.launch.py` behind a launch arg before joint bringup.

### Parsa-stack changes since 2026-05-21 (field testing @ IGVC practice, Oakland U)

His checkout has ~30 commits dated 2026-05-28→05-30 that postdate the notes above. Net deltas relevant to joint bring-up:

- **Production pipeline is now `sooner25` (single-class), not `hsv`.** His mask no longer distinguishes barrel from lane. See § "Does he already have perception" above.
- **Lane-cost pivot (the big one).** [`docs/yaw_diag_session_2026_05_28/lane_following_strategy.md`](../references/parsa_igvc/docs/yaw_diag_session_2026_05_28/lane_following_strategy.md): lethal lane walls (cost 254 + inflation) were trapping the robot in 2–3 m lanes. Plan is to split `lane_white` out of the `danger` class into a **`soft_lane` class_type (base_cost 180, max 220 — below the inscribed-inflated 253)** so lanes become a centerline gradient, not a wall; barrels/potholes stay lethal. Config-only on his side, but it changes the *downstream cost meaning* of the `lane_white` ID my adapter emits.
- **MPPI:** `batch_size: 500`, `time_steps: 56`, `model_dt: 0.05`, **`vx_max: 1.5`** (was 0.7), `vx_min: -0.6`. `vx_max` is read at configure-time — live `ros2 param set` does nothing.
- **Footprint:** circular `robot_radius` → **measured chassis rectangle** `[[0.8128,0.415],[0.8128,-0.415],[-0.2794,-0.415],[-0.2794,0.415]]`; `robot_radius` now ignored.
- **Inflation:** local **1.0→0.3**, global **0.65→0.3** (narrow-lane fit). Goal tol **2.0→0.2** (RTK precision).
- **EKF heading:** `wheel_odom` `vyaw` **removed from both EKFs** — heading is now **Xsens-gyro-only**. Wheel `vx` still fused.
- **Map EKF GPS:** now fuses **RTK GPS in ABSOLUTE mode** (`odom0_differential: false`) via MDOT CORS NTRIP; GNSS lever arm set to `[0.74,0,0]` (was `[0,0,0]` TODO).
- **PR3 raytrace-clear** is treated as applied & stable on his side (`clearing: true` + `raytrace_max_range: 8.0` live on all sources, `tile_map_decay_time: 0.3`). The patch file is **still not vendored** in this checkout (no `src/avros_bringup/patches/` dir).
- **Caveat:** his own [`CLAUDE.md`](../references/parsa_igvc/CLAUDE.md) and `ekf.yaml` header are stale vs. his own code (CLAUDE still says VoxelLayer/ObstacleLayer pre-STVL; ekf header still says wheel vyaw fused). Trust the YAML/source, not his prose, on costmaps and EKF.

### Standalone vision testing (v3.2.2)

Vision-only bringup that needs neither the rest of the robot nor Parsa's stack — see `## Standalone vision testing` in the in-package README. TL;DR:

```bash
# Live cameras only (skip Xsens / Velodyne / Nav2 / actuator)
ros2 launch avl_bev_perception vision_test.launch.py

# Bag replay — develop off-Jetson, no cameras attached
ros2 launch avl_bev_perception vision_test.launch.py \
    use_bag:=true bag_path:=/data/bev_session_...

# Record a replay-able bag from the live robot
./avl_bev_perception/tools/record_session.sh /data/run_42
```

The bag captures only what BEV needs (per-camera rgb rect + camera_info + depth_registered, plus `/tf` + `/tf_static`). Vision-only — IMU/LiDAR are intentionally not recorded.
