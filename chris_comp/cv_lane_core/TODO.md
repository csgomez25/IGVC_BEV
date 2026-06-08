# TODO — `cv_lane_core` roadmap

> **Final goal:** a **fully transferable** computer-vision layer — multi-camera +
> BEV fusion — that you move to a different vehicle by **editing YAML only**, no
> code changes. One detector library, N camera profiles, one BEV/grid profile.

> **Build order (important):** **stay pure-CV first.** Track A is the near-term
> scope and now runs all the way to a **standalone multi-camera BEV fused grid** —
> detection *and* the top-down picture, pure numpy + opencv, geometry from YAML.
> What waits for Track B is only the **robot/planner coupling**: the costmap
> hand-off, TF/depth-sourced geometry, and the optional ML model.
> [ARCHITECTURE.md](ARCHITECTURE.md) is the north star we grow into **once the CV
> + BEV are mature and accurate** — don't start Track B until Track A clears its
> quality gate (below).

Context and rationale: [ISSUES.md](ISSUES.md). Design blueprint (contracts +
swappable interfaces): [ARCHITECTURE.md](ARCHITECTURE.md). Library entry doc:
[README.md](README.md).

> **Status — 2026-06-06 (post-competition foundation pivot).** IGVC is over;
> this folder is now being built as a **multi-year, vehicle-agnostic BEV
> foundation** — a future team drag-and-drops it, calibrates, and focuses on
> path planning. Decision: the BEV layer ships **both** projection modes —
> **homography (default, any mono camera, no depth)** + an **optional depth
> projector** — so the depth path (previously parked in Track B) is pulled into
> Track A as a co-equal standalone mode, since the current ZED-X robot has depth.
>
> **Landed this session (vertical slice, tested via `tests/check.py` Phases 2+3):**
> `lane_cv/bev.py` (`GroundGrid`, `HomographyProjector`, `DepthProjector`,
> `fuse_grids`), `lane_cv/rig_config.py` (`RigConfig` — the single per-vehicle
> file), `lane_cv/rig.py` (`Rig` = the `MultiCamDetector`+`MultiCamBev` of the
> blueprint), and `configs/vehicle_example.yaml`. Naming note: the code uses
> `Rig`/`RigConfig`/`vehicle.yaml`; the blueprint's `MultiCamBev`/`robot.yaml`
> are the same concepts. Structure note: BEV lives in `lane_cv/bev.py` (still
> 100% ROS-free) rather than a separate `bev/` package — can be split later.

Guiding rules (so the goal stays true):
- **No code edits to port.** Anything robot/venue-specific is a YAML field. If a
  port needs a `.py` edit, that's a bug in the config schema → fix the schema.
- **Stable contracts, swappable middle.** One input contract (intrinsics,
  extrinsics/TF, ground model, pose) + one output contract (planner product);
  every stage hides behind an interface (`SegmentationProvider`, `Projector`,
  `Fuser`, `CostmapAdapter`). See [ARCHITECTURE.md](ARCHITECTURE.md) §3–§5.
- **Dual-source loaders.** Every input accepts ROS *or* YAML — this is the real
  portability unlock, more than any algorithm.
- **ROS stays out of `lane_cv/` (and `bev/`).** Adapters live in a separate layer.
- **Degrade, never hard-fail.** Depth→flat-plane, N→N-1 cameras, TF→YAML.
- **Per-frame, per-camera detection is the unit of reuse** — BEV fuses results,
  it does not re-implement detection.
- **Classical is the portable baseline; a model is an optional Tier-2** behind the
  same `SegmentationProvider` seam, off by default (§6).

---

# ░ Track A — pure CV (NOW) ░

Detection **and** the standalone multi-camera BEV fused grid. Get it accurate and
robust; keep it ROS-free, numpy + opencv, geometry from YAML. This is the whole
near-term scope — it ends with a trustworthy top-down picture, no costmap, no ROS.

## Phase 0 — Single-camera core ✅ DONE

- [x] ROS-free `LaneDetector` (`pipeline.py`) — numpy + opencv only.
- [x] Color cues: white gate, asphalt-invert, near-field adaptive floor (`segmentation.py`).
- [x] **Geometric speck filter** — elongation / fill / orientation / Hough (`line_filter.py`).
- [x] Temporal N-of-window voting.
- [x] `LaneConfig` YAML schema + `default.yaml` + `example_other_car.yaml`.
- [x] `tools/run.py` (image / mixed folder / video, `--video-out`, `--view`,
      `--recursive`) + `tools/tune.py` (live sliders).
- [x] Verified on synthetic: 2 lines + 80 specks + patch → only the 2 lines survive.
- [x] Verified on **real footage** (`test.mp4`, bright concrete + low-contrast
      white lines): default profile under-detected → fitted
      `configs/bright_concrete.yaml` (drop not-asphalt gate, lower floor to the
      measured line tail) → white lines detected, barrels/bumper rejected.
      Confirms the "new surface = new YAML, not new code" claim.

## Phase 1 — YAML-only single-camera completeness

Goal: a new camera is a new YAML, full stop.

**Harness-surfaced recall gaps** (from `tests/check.py`, 2026-06-05 — speck
rejection is already solid at `false_lines=0` across all clutter scenes; these
are *recall* misses):
- [ ] **Dashed lines (`scene:dashed`, recall 0.00)** — each dash is a short stub
      that fails the elongation gate, so the whole line is dropped. Add an
      *orientation-aware* close (bridge collinear gaps along the lane direction)
      or a cross-component collinearity merge — without welding isotropic specks.
- [ ] **Shadowed lane (`scene:shadow`, recall 0.23) + low-contrast far lines
      (real `test.mp4`)** — the absolute `white.v_min` clamp drops lane pixels in
      shadow, and the far end of a low-contrast line fades into the surface. Move
      to a *local/contrast* gate (lane brightness relative to nearby surface)
      instead of a global brightness floor; this would also reduce the need for a
      hand-fitted `bright_concrete.yaml`.
- [ ] **Combined hard scene (`scene:hard_all`, recall 0.65)** — should recover
      once dashed + shadow are handled; keep as the integration check.

- [x] **`SegmentationProvider` seam** ✅ (2026-06-06) — `providers.py`:
      `infer(bgr, depth) -> SegResult(class_mask, confidence)`; `LaneDetector`
      adapted as `ClassicalProvider`; `YoloProvider` + `RoadSegProvider` (stub)
      behind the same contract; `build_provider()` factory selected by
      `segmentation.backend` in `vehicle.yaml`. `Rig` now runs through providers,
      so a YOLO/road-seg backend drops in by config, no rewrite. Driving-class
      registry in `classes.py` (`collapse_to_bev` maps rich classes → grid set,
      so a detected person/cone becomes `CLASS_OBSTACLE`). Lazy imports keep the
      core ultralytics-free. See ARCHITECTURE §5.
- [ ] **Schema validation** — `LaneConfig.validate()`: range-check every field,
      clear errors on bad ROI / HSV / band, fail fast at load.
- [ ] **Profile inheritance** — `extends: default.yaml` key so a per-car profile
      overrides only what differs (avoid copy-paste drift).
- [~] **Class-ID map** — driving-class registry landed in `classes.py`
      (id↔name + `is_obstacle`, COCO→our-id map). Providers emit class IDs, not
      binary. Still TODO: make the registry/mapping YAML-overridable per profile
      and reconcile IDs with Parsa's `class_map.yaml` for the kiwicampus contract.
- [ ] **Barrel + pothole passes** (closes ISSUE #5) — add color/shape gates as
      configurable "classes" so it's not lane-only, still YAML-driven.
- [ ] **Confidence output** — per-pixel confidence plane alongside the mask.
- [ ] Golden-frame regression set + a tiny runner (pick a framework first — no
      test harness exists in this repo yet).

## Phase 2 — Multi-camera (N cameras, one config tree) ✅ MOSTLY DONE (2026-06-06)

Goal: 3+ cameras with zero per-camera code.
- [x] **Per-vehicle config tree** — `RigConfig` (`rig_config.py`) +
      `configs/vehicle_example.yaml`: a `cameras` list, each with `name`,
      a `detect_profile: <file>` (a `lane_cv` profile), and a `projection` block.
      Unified with the grid spec in one file (realizes the Phase-5 `robot.yaml`
      goal early), superseding the separate `cameras.yaml`/`bev.yaml` plan.
- [x] **`Rig`** (`rig.py`) — holds one `LaneDetector` per camera (each keeps its
      own adaptive + temporal state); `process({name: bgr}, depths={...})` →
      fused `BevResult`. This is the blueprint's `MultiCamDetector`.
- [x] **Per-camera health** — a camera absent from a tick's `frames` (or a depth
      camera with no depth) is skipped, never crashes the set. Tested
      (`tests/check.py` Phase 2: `rig:dropped-camera`).
- [ ] **Parallelism knob** — per-camera detection across cores (Orin has 12);
      `parallel: true` in YAML. Mirror the existing node's `parallel_cameras`.
- [ ] `tools/run.py` multi-input mode (folder-of-folders / multi-video).

## Phase 3 — Multi-camera BEV fusion (standalone CV) ✅ CORE DONE (2026-06-06)

Goal: N per-camera masks → one shared **top-down fused grid**, pure numpy +
opencv, geometry from YAML. This is the **end state of Track A** — the whole
transferable top-down picture, with **NO ROS and NO costmap**. Implements the
`Projector` + `Fuser` interfaces (ARCHITECTURE §5) in their standalone form.
- [x] **Grid + per-camera projection config** — `GroundGrid` (extent_m,
      resolution_m_per_px) + per-camera `projection` block in `vehicle.yaml`
      (homography matrix, or depth intrinsics + mount extrinsics). All YAML —
      the only thing that changes geometrically between vehicles.
- [x] **`lane_cv/bev.py` projectors** — `HomographyProjector` (flat-ground IPM
      warp, any mono camera) **and** `DepthProjector` (per-pixel back-projection,
      yaw-only mount model reimplemented from `bev_perception_node.py`, ROS-free).
- [x] **Fuser** — `fuse_grids`: class-priority compositing (obstacle > lane > bg),
      shared grid shape. Depth path has ground-plane class exemption (lanes
      survive the lower height gate). Tested (`tests/check.py` Phase 3).
- [x] **`Rig`** — ties per-camera `LaneDetector` → projector → fuser:
      `process({cam: bgr}, depths={...}) -> BevResult` (`class_grid` +
      `lane_mask`/`obstacle_mask` views). `meters_per_px` honored via grid res.
- [x] **Geometry verified** — `tests/check.py` Phase 3: grid `world_to_px`
      matches the projection matrix; depth back-projection + homography warp land
      a pixel at the predicted vehicle coordinate; fusion priority correct.
- [ ] **Homography calibration tool** — `tools/calibrate_bev.py`: click a few
      known ground points per camera, `cv2.findHomography`, write the YAML.
      **← highest-leverage next step (make-or-break for "drag-and-drop").**
- [ ] **Multi-class projection** — `rig.py` currently maps the detector's single
      lane class into the grid; the marked hook takes the Phase-1 multi-class mask
      (lane/barrel/pothole) so cones become `CLASS_OBSTACLE` automatically.
- [ ] **(Optional) detect-in-BEV mode** — warp first, then detect on the top-down
      view, where lanes are parallel / constant-width and the elongation +
      dashed-bridging gates work better. An accuracy lever, not required.
- [ ] **BEV view in `tools/run.py`** + a fused-grid scene in `sim/` (real-footage
      fused-grid scoring; the current Phase-3 checks are synthetic geometry only).

---

# ░ Track B — integration (LATER, once the CV + BEV are accurate) ░

**Do not start until Track A clears the quality gate.** Track A ends with a good
*standalone* top-down fused grid; Track B is everything that couples it to the
**robot and the planner** — the costmap hand-off, robot-sourced geometry, and the
optional model. The *live-capture* adapters already exist early for testing; only
these integration parts wait on Track A.

> **Track A → B quality gate.** Move to Track B only when, on a representative set
> of real clips per surface:
> - `tests/check.py` is fully green (dashed + shadow fixed, plus a multi-cam fusion scene), and
> - real-footage lane recall is consistently high with `false_lines ≈ 0`, and
> - the fused BEV grid is geometrically sane (a calibration target lands where it should), and
> - a new surface/venue needs only a YAML retune / recalibration — no code — confirmed on ≥2 real datasets.
> Until then, every hour goes into Track A.

## Phase 3b — BEV geometry upgrades (need the robot)

Accuracy upgrades to the Phase-3 BEV that require live robot data. Optional — the
standalone YAML/homography path already works without them.
- [ ] **Extrinsics auto-loaded from URDF/TF** — allow `source: tf`/`source: urdf`
      (`base_link → <cam>_optical`) instead of hand-entered YAML. NOTE: the
      example's mount values are now copied from Parsa's `avros.urdf.xacro`
      (resurveyed 2026-05-06) by hand; this item is to parse them automatically
      so they can't drift from the URDF.
- [x] **`DepthProjector`** — standalone (`lane_cv/bev.py`), takes a plain depth
      array (no ROS/cloud), fits Track A.
      - [x] **Full roll/pitch/yaw mount model** (2026-06-06) — replaced the
        yaw-only model (which silently dropped pitch — wrong for the real robot,
        whose FRONT ZED is tilted **15° down**). Principled REP-103
        optical→body convention (`P_veh = R_mount @ R_o2b @ P_opt + xyz`); image
        -right now maps to vehicle-right (the old lifted math was mirrored).
        `configs/vehicle_example.yaml` carries the real measured front/left/right
        mounts from the URDF. Verified: `check.py` Phase 3 `depth:handedness` +
        `depth:pitch-applied`.
      - [ ] **On-robot validation** against live ZED depth (handedness/pitch
        confirmed in sim only; confirm a calibration target lands correctly).
- [ ] **Pose-based accumulation** — use `/odometry/filtered` (or `/tf`) to build a
      persistent map vs. an instantaneous BEV. Static extrinsics always required;
      dynamic pose only for accumulation.
- [ ] **Calibration self-check vs TF** — compare the YAML homography against TF and
      warn on mismatch (extend the serial-sanity-check pattern to geometry).

## Phase 4 — Adapters (optional, kept thin and outside `lane_cv/`)

Goal: integrate without contaminating the portable core. **Live-capture
adapters started early (for hardware testing) — see `adapters/`.**
- [x] **`adapters/usb_cam.py`** — plain OpenCV USB/UVC webcam (or video) runner.
      Verified end-to-end on a video stream.
- [x] **`adapters/ros2_zed.py`** — rclpy node, 1 or 3 ZED cameras, v5 topic paths
      from the references, per-camera detector, publishes `/lane/<cam>/{mask,overlay}`.
      Compiles + core stays ROS-free; **needs on-Jetson validation against live
      ZED topics** (can't run ROS in this env).
- [x] **`adapters/__init__` guarded imports** so `lane_cv` still imports without ROS
      (verified: importing the core pulls in no rclpy).
- [ ] **`CostmapAdapter` with selectable output modes** (the output contract,
      ARCHITECTURE §4) — config picks one of:
      - `semantic_layer` *(recommended)* — kiwicampus per-camera contract
        (mask + confidence + organized cloud relay + LabelInfo), **keeping class
        semantics: `lane_white` → soft cost (≤220), `barrel_orange`/`pothole` →
        lethal (254)**. This is the soft-lane / lethal-barrel split his field
        notes converged on. Needs Phase 1 class-IDs.
      - `occupancy_grid` — fused BEV → `nav_msgs/OccupancyGrid` (generic, no semantics).
      - `obstacle_cloud` — obstacle cells → `PointCloud2` for an STVL source.
- [ ] Wire in `MultiCamDetector` + BEV output once Phases 2–3 exist.
- [ ] Document the one-line hook to replace `avl_bev_perception` Tier-1 white pass.

## Phase 5 — Calibration & portability tooling

Goal: bring-up on a new car is "run two scripts, edit two YAMLs."
- [ ] **Auto-HSV calibration** — port `auto_calibrate.py` as a ROS-free helper that
      samples frames and writes a profile YAML (white V/S, asphalt range).
- [ ] **BEV extrinsics check** — project a known ground target, print reprojection
      error so mount YAML can be sanity-checked without the full robot.
- [ ] **`new_vehicle.md`** — a checklist: which YAMLs to copy, what to measure,
      how to validate, in order.
- [ ] CI/lint — only if/when a test framework is chosen for this repo.
- [ ] **Unify bring-up config** — reconcile `cameras.yaml` + `bev.yaml` into a
      single declarative `robot.yaml` (ARCHITECTURE §7): per-camera intrinsics
      source, extrinsics source, ground model, segmentation backend, plus the BEV
      grid and output mode. One file per robot.

## Phase 6 — Optional ML model (Tier-2, behind the same seam)

Goal: better accuracy where classical struggles, **without** breaking
plug-and-play. The model is opt-in; classical stays the zero-data baseline.
- [x] **`YoloProvider`** ✅ (2026-06-06) — ultralytics YOLO (nano default),
      driving-relevant COCO classes (person/car/bus/truck/bicycle/motorcycle/
      dog/cat/traffic_light/stop_sign) → our IDs, boxes rasterized into the
      class+confidence masks. Lazy import; off unless a camera selects it.
- [ ] **Road/sidewalk segmentation** — `RoadSegProvider` is a **stub** awaiting a
      model choice (research): a Cityscapes-pretrained net (SegFormer-B0 /
      BiSeNet / DeepLab-mobile), ONNX-exported for the Jetson, emitting the
      `road`/`sidewalk` IDs from `classes.py`. The seam is ready; only `infer()`
      is unimplemented.
- [ ] **`OnnxProvider`** (generic) implementing `SegmentationProvider` — for any
      ONNX seg model beyond the road segmenter. Same contract; fuse so classical
      wins on colors it is reliable for.
- [ ] **Targets the known gaps** — low-contrast lines (the `test.mp4` case),
      dashed/shadowed lines, surface variety, real semantic classes
      (person/pothole), cross-venue generalization without per-venue HSV tuning.
- [ ] **Cost to accept knowingly** — labeled data, GPU (~5–15 ms), ONNX export,
      an eval loop. Never block the classical baseline on it.
- [ ] **Endgame** — once Phase 3 exists, train a small model **in the BEV frame**
      on accumulated grids, so segmentation and the planner share one frame.

---

## Definition of done (final goal)

A new vehicle is brought up by:
1. `cp -r configs/<known_car> configs/<new_car>`
2. edit one `robot.yaml` — per-camera topics/serials, intrinsics + extrinsics
   (TF or YAML) + ground source, each camera's `lane_cv` profile (or model ref),
   the BEV grid, and the output mode
3. run the adapter

…and **no `.py` file under `lane_cv/` or `bev/` is touched.** That is the test.
