# Architecture — a pluggable CV → BEV → costmap stack

This is the design blueprint for growing `cv_lane_core` from a single-camera lane
detector into a **drop-in perception stack**: give it the correct information
(intrinsics, extrinsics/TF, ground model, pose) and it produces what a path
planner consumes — on most robots, with no code edits.

It describes the **target design** — the destination, not the current build.

> **When to build this — track split:** the near-term project is **pure CV**, and
> that now includes the **standalone multi-camera BEV**: segmentation [1],
> projection [2], and fusion [3] are all **Track A** (numpy + opencv, geometry
> from a YAML homography / flat-ground, no ROS). Only the **costmap adapter [4]**,
> the **robot-sourced geometry** (TF extrinsics, depth projection, pose
> accumulation), and the **optional model** are **Track B** — built only **once
> the CV + BEV are mature and pass the quality gate** (TODO.md). Read this as the
> north star you aim the Track-A work toward; don't start Track B yet.

What exists today is the segmentation layer (`lane_cv/`) plus live-capture
adapters (`adapters/`). Everything else here is specified and tracked in
[TODO.md](TODO.md); sections marked _(planned)_ are not yet implemented — do not
assume the file/class exists.

---

## 1. The one principle

> **A stable input contract and a stable output contract, with everything
> between them swappable.**

If a robot can supply the inputs (§3) and consume the output (§4), the stack
works. Each internal stage (§5) hides behind an interface, so the classical
detector, an ML model, a flat-ground projector, or a depth projector are all
interchangeable. Portability is a property of the *contracts*, not of any one
implementation.

Two rules keep it true:

- **The core stays dependency-light.** `lane_cv/` (and the planned `bev/`)
  import only numpy + opencv. ROS, TF, and hardware live **only** in `adapters/`.
- **Degrade, never hard-fail.** Missing depth → flat-plane projection. Missing a
  camera → fuse the rest. Missing TF → fall back to YAML mount poses. A stack
  that crashes on incomplete information is not pluggable.

---

## 2. The pipeline

```
  per camera                                          robot frame
 ┌──────────┐   ┌───────────────────┐   ┌───────────┐   ┌─────────┐   ┌──────────────┐
 │ RGB(+dep)│─► │ [1] Segmentation  │─► │ [2]Project│─► │[3] Fuse │─► │[4] Costmap   │─► planner
 │  source  │   │     Provider      │   │  to ground│   │  N cams │   │   adapter    │
 └──────────┘   └───────────────────┘   └───────────┘   └─────────┘   └──────────────┘
                   classical OR onnx       intrinsics +     class-       semantic layer /
                   (same interface)        extrinsics +     priority     occupancy grid /
                                           ground model     composite    obstacle cloud
```

Static configuration (intrinsics, extrinsics, grid spec) is loaded once;
per-frame data (images, optional depth, optional dynamic pose) streams through.

---

## 3. Input contract — "the correct information"

Everything the stack needs from the host robot. Each item is accepted from
**either** a ROS source **or** a YAML value, behind one loader, so the same core
runs on a full robot (TF/URDF) or a bare bench rig (measured YAML).

| Input | What it is | ROS source | Standalone source | Needed for | If missing |
|---|---|---|---|---|---|
| **Intrinsics** | `fx, fy, cx, cy`, distortion | `CameraInfo` topic | `camera.yaml` | undistort + per-pixel ray | cannot project — required |
| **Extrinsics** | `base_link → <cam>_optical` (static) | TF lookup (URDF) | mount pose in YAML (REP-103) | placing pixels in the robot frame | fall back YAML→TF→error |
| **Ground model** | flat-plane height **or** per-pixel depth | depth image / organized cloud | `ground_height_m` (flat) | pixel → ground (x, y) | flat-plane assumption |
| **Static yaw** | each camera's mount yaw | TF (in extrinsics) | YAML | multi-camera alignment | required for fusion |
| **Dynamic pose** | vehicle pose/heading over time | `/odometry/filtered`, `/tf` | — | *accumulating* a map across frames | instantaneous BEV only |
| **Sync + units** | stamps, `frame_id`, REP-103, meters | message headers | implicit | geometric correctness | per-frame snapshot |

**Static vs dynamic** is the distinction people miss: mount extrinsics (static)
are needed *always*; vehicle pose (dynamic) is needed *only* if you want a
persistent accumulated occupancy map rather than a fresh BEV each frame. Start
with instantaneous BEV; add pose-based accumulation later.

**Frame conventions:** robot frame is REP-103 (X forward, Y left, Z up, meters);
image/cloud `frame_id` is the optical frame (Z forward, X right, Y down). The
loader converts; downstream is always REP-103.

---

## 4. Output contract — what the planner consumes

A planner does not take "a mask"; it takes one of three Nav2-shaped products.
The stack should support the first two, selected by config.

1. **Semantic costmap layer** _(recommended default for IGVC)_ — the kiwicampus
   `semantic_segmentation_layer` contract: per-camera `mask + confidence +
   organized cloud + LabelInfo`. **This is Parsa's actual integration hook**, and
   it keeps class semantics all the way to the costmap, so:
   - `lane_white` → **soft cost** (base ~180, max ≤220 — a centerline gradient)
   - `barrel_orange`, `pothole` → **lethal** (254)

   That soft-lane / lethal-barrel split is precisely the fix his field notes
   converged on (lethal lanes trapped the robot in 2–3 m corridors). Preserving
   it is the difference between driving a corridor and getting walled in.
2. **OccupancyGrid / costmap_2d layer** — fused BEV flattened to occupied/free.
   Most generic; works with any Nav2 stack; loses semantics.
3. **PointCloud2 of obstacle cells** → STVL `observation_source`. Simplest; least
   information.

**Keeping semantics to the costmap is what separates an optimal stack from a
binary blob.** Options 2–3 are fallbacks for stacks that can't take option 1.

---

## 5. The internal interfaces (the swappable seams)

Sketches, not final signatures — they define the *seams*, all _(planned)_ except
where noted.

### `SegmentationProvider` — RGB(+depth) → per-camera class mask
```python
class SegmentationProvider(Protocol):
    def infer(self, bgr, depth=None) -> SegResult: ...
    # SegResult: class_mask (uint8 IDs), confidence (uint8), segments(optional)
```
- `ClassicalProvider` — wraps today's `LaneDetector` (+ future barrel/pothole
  passes). Zero data, fully portable. **The always-on baseline.** _(adapts existing code)_
- `OnnxProvider` — an optional trained model behind the *same* interface, off by
  default. See §6.

Class IDs come from config and default to Parsa's `class_map.yaml`
(`0=free, 1=lane_white, 2=barrel_orange, 3=pothole, 255=unknown`).

### `Projector` — pixel mask → ground-frame cells
```python
class Projector(Protocol):
    def project(self, mask, intrinsics, extrinsics, ground) -> GroundPoints: ...
```
- `FlatGroundProjector` — inverse-perspective mapping (homography / flat-plane
  assumption, RGB-only). **Track A** — the standalone, YAML-configured path.
- `DepthProjector` — uses depth/organized cloud (accurate on slopes, curbs).
  **Track B** — needs the robot's depth stream.

Precompute and cache the per-camera projection LUT (the v3.2 node already proves
this pattern saves ~5–10 ms/camera); the hot path must reuse the cache.

### `Fuser` — N projected masks → one BEV grid
```python
class Fuser(Protocol):
    def fuse(self, projected: dict[str, GroundPoints], grid: GridSpec) -> BevGrid: ...
```
- Class-priority compositing (lane vs barrel vs pothole).
- **Ground-plane class exemption** — painted-flat classes (lane, pothole) survive
  a height/footprint filter that would otherwise drop them.

### `CostmapAdapter` — BEV grid / per-cam masks → planner product
```python
class CostmapAdapter(Protocol):
    def publish(self, ...): ...   # semantic layer | OccupancyGrid | PointCloud2
```
Lives in `adapters/` (it is ROS-bound by nature).

---

## 6. Classical vs. model — the seam, and why classical stays the foundation

A model is the **least portable** component: it generalizes only as far as its
training data, and needs a GPU + an export/eval pipeline. Making it the
foundation *destroys* plug-and-play — every new venue would need data + retrain.

> **Classical is the always-on, zero-data, fully-portable baseline. A model is an
> optional Tier-2 behind `SegmentationProvider`, config-selected, off by default,
> fused so classical wins on the colors it is reliable for.**

This is the Tier-1/Tier-2 split already in `seg_inference.py`, formalized as a
clean interface.

**When a model earns its keep:** low-contrast lines (the `test.mp4` case that
needed a hand-tuned profile), dashed lines, shadows, surface variety, real
semantic classes (person/pothole), and cross-venue generalization *without*
per-venue HSV tuning.

**What it costs:** labeled data, ~5–15 ms GPU latency, ONNX export, a
calibration/eval loop. Worth it once you have data; never worth *blocking* a
working baseline on.

**Endgame:** once the geometry layers exist, train a small model **in the BEV
frame** on accumulated grids — segmentation and the planner then share one
top-down coordinate system.

---

## 7. Declarative bring-up — one file per robot

A new robot is one `robot.yaml` _(planned)_; no code:

```yaml
frame: base_link
bev:                      # the shared top-down grid
  extent_m: [12, 12]
  resolution_m_per_px: 0.05
cameras:
  - name: front
    rgb_topic:   /zed_front/zed_node/rgb/color/rect/image
    info_source: topic        # or: yaml: configs/front_intrinsics.yaml
    extrinsics:  tf           # or: yaml: {x,y,z,roll,pitch,yaw}
    ground:      depth        # or: flat
    segmentation: configs/bright_concrete.yaml   # a lane_cv profile, or an onnx model ref
  # - name: left  ...
  # - name: right ...
output:
  mode: semantic_layer        # or: occupancy_grid | obstacle_cloud
  class_map: configs/class_map.yaml
```

This file *is* the input contract made concrete: intrinsics source, extrinsics
source, ground model, segmentation backend, and output mode — all per-camera,
all swappable.

---

## 8. What makes it optimal (the non-negotiables)

1. **Dual-source loaders** — every input accepts ROS *or* YAML (§3). This is the
   real portability unlock, more than any algorithm.
2. **Adapter boundary held** — core import-clean; ROS/TF/hardware only in `adapters/`.
3. **Calibration ingestion + self-check** — read `CameraInfo`, look up TF, then
   *validate*: reproject a known ground target, print error, warn on mismatch
   (extend the existing serial-sanity-check pattern to geometry).
4. **Graceful degradation** — depth→flat, N→N-1 cameras, TF→YAML; warn, continue.
5. **Observability** — publish every stage (per-cam mask, projected cloud, fused
   grid, costmap) so bring-up is visual, like the review panels.
6. **The `SegmentationProvider` seam** — classical ↔ model is a config switch.
7. **Semantics to the costmap** — soft lanes, lethal barrels (§4).

---

## 9. Mapping to the code and the roadmap

| Layer | Track | Status | Where | Phase |
|---|---|---|---|---|
| Segmentation (classical) | A | **exists** | `lane_cv/` | 0 ✅ |
| `SegmentationProvider` seam + classes | A | planned | `lane_cv/` refactor | 1 |
| Multi-camera | A | planned | `MultiCamDetector` | 2 |
| Projection + Fusion (standalone BEV) | **A** | planned | `bev/`, `tools/calibrate_bev.py` | 3 |
| BEV geometry upgrades (TF / depth / pose) | B | planned | `bev/`, `adapters/` | 3b |
| Costmap adapter (semantic / occ / cloud) | B | planned | `adapters/` | 4 |
| Calibration / self-check tooling | A/B | planned | `tools/`, `adapters/` | 5 |
| Optional ONNX Tier-2 | B | planned | `OnnxProvider` | 6 |

Track A = pure CV, runs to a standalone top-down fused grid (YAML/homography
geometry, no ROS). Track B = robot/planner coupling, built after the quality gate.

Live-capture adapters (`adapters/usb_cam.py`, `adapters/ros2_zed.py`) already
exist as the §1 boundary and the per-camera test path.

See [TODO.md](TODO.md) for the detailed, checkable breakdown, and
[CODE_GUIDE.md](CODE_GUIDE.md) for a walkthrough of what is built today.
