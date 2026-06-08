# `cv_lane_core` — portable lane detection + multi-camera BEV

A **self-contained, ROS-free** computer-vision foundation: per-camera lane/line
detection **plus** a multi-camera Bird's-Eye-View (BEV) fusion layer that turns
those detections into one metric top-down grid for a path planner. Split out of
the IGVC BEV stack so the CV is one transferable thing you drop into another
vehicle with **config + calibration, not code edits**.

> **End goal.** A future team brings up a new robot by copying this folder,
> editing one `vehicle.yaml`, and calibrating each camera — then consumes the
> fused BEV grid and focuses on path planning. No `.py` edits to port. The full
> roadmap + definition-of-done live in [TODO.md](TODO.md); the design contracts
> in [ARCHITECTURE.md](ARCHITECTURE.md).

Depends only on `numpy` + `opencv-python` (`pyyaml` optional, for config I/O).
No ROS, no message types, no robot-specific paths — copy it onto any machine.

```python
from lane_cv import LaneDetector, LaneConfig

det = LaneDetector(LaneConfig.from_yaml("configs/default.yaml"))
result = det.process(bgr_frame)      # any HxWx3 BGR image
result.lane_mask                     # uint8 0/255 — confirmed painted lines
result.candidate                     # uint8 0/255 — color stage, pre-shape-filter
result.segments                      # list[LineSegment] (endpoints, angle, length)
```

Try it:

```bash
# a single image, a folder (images and/or videos), or a video file
python tools/run.py --config configs/default.yaml --input some_frame.jpg
python tools/run.py --config configs/default.yaml --input my_testset/ --recursive

# write the result out as a playable video (great for reviewing a clip)
python tools/run.py --config configs/default.yaml \
    --input clip.mp4 --video-out clip_lanes.mp4 --view panel --no-display

# live trackbar tuner against one representative frame
python tools/tune.py --config configs/default.yaml --input some_frame.jpg
```

`--view panel` writes the 3-up `overlay | candidate | lane_mask` (best for
*seeing the CV work*); `--view overlay` writes just the lanes drawn on the input.

---

## Multi-camera BEV (the foundation)

One step up from a single camera: a `Rig` runs a detector per camera, projects
each camera's detections onto a shared metric ground grid, and fuses them into
one top-down picture — the product a path planner consumes.

```python
from lane_cv import Rig

rig = Rig.from_yaml("configs/vehicle_example.yaml")     # the ONE per-vehicle file
out = rig.process(
    {"front": front_bgr, "usb_test": usb_bgr},          # {camera_name: BGR}
    depths={"front": front_depth},                      # depth cams only
)
out.class_grid       # uint8 HxW vehicle-frame BEV: 0=bg, 1=lane, 2=obstacle
out.lane_mask        # uint8 0/255 convenience view
out.obstacle_mask    # uint8 0/255 convenience view
out.grid             # the GroundGrid (px <-> meters conversions)
```

**Two projection modes, chosen per camera in `vehicle.yaml`:**

- **`homography`** — flat-ground image→ground warp. Works with **any** camera,
  **no depth sensor**. Calibrate once from ≥4 ground-point correspondences. This
  is the portable default — it's what lets a laptop + USB cam exercise the whole
  pipeline. Correct for things on the ground (paint, sidewalk edges).
- **`depth`** — per-pixel back-projection using intrinsics + the camera's mount
  pose. Needs a depth/stereo camera (e.g. ZED X), but recovers true 3D so it
  handles object height. Intrinsics come from the camera; only the mount pose
  (extrinsics) is calibrated/measured. The math is reimplemented here from the
  field-validated `avl_bev_perception` node — **this folder imports nothing from
  that ROS package.**

`configs/vehicle_example.yaml` shows both: a ZED-X `depth` camera and a USB
`homography` camera in one rig. A dropped camera (missing frame, or a depth
camera with no depth this tick) is skipped, never crashes the rig.

> Calibrating a new camera's `H` / extrinsics is the make-or-break for
> "drag-and-drop." The guided tool (`tools/calibrate_bev.py`) is the next piece —
> see [TODO.md](TODO.md) Phase 3 / Phase 5.

---

## Why this exists — what was wrong with the line detection

Both perception stacks in this repo (`avl_bev_perception`'s Tier-1 HSV and
Parsa's `avros_perception` `hsv`/`sooner25` pipelines) detect lanes by
**thresholding color/brightness per pixel and then filtering only by blob
area**. The field notes in `references/parsa_igvc/docs/` show that approach
failing repeatedly. The root causes:

1. **Color can't separate paint from bright not-paint.** "White lane" = low
   saturation + high V. Sun-lit concrete, light asphalt, painted curbs, glints,
   and especially **tar-filled cracks, expansion joints, and bright road specks**
   all pass the same gate. Parsa dropped per-class `hsv` as his production
   pipeline on 2026-05-28 because *"per-class HSV kept misclassifying bright
   concrete/asphalt at the IGVC practice course."*

2. **The adaptive brightness floor is scene-dependent.** The `mean + k·σ`
   V-threshold drifts with whatever is in frame (sky, shadow, hood). On some
   frames it sits below the specks and they pass. `hsv.py`'s own comments record
   this *("bit us in /tmp/hsv_iter all session")* and the band-restriction hack
   added to fight it.

3. **No shape discrimination — the direct cause of "specks → fake lines."**
   Components were filtered only by **area**. A bright crack-cluster easily
   exceeds the min-area and gets painted as "lane." Nothing checked that a lane
   blob is actually *line-shaped*.

4. **Morphology manufactures lines.** `MORPH_CLOSE` / the horizontal
   `lane_close_w` bridge welds nearby specks into a continuous streak that then
   *looks* like a dashed lane.

5. **Single-class collapse loses information.** `sooner25` (now Parsa's default)
   thresholds asphalt and inverts → one "obstacle" class. Robust to the V-floor
   problem, but it *"cannot tell a barrel from a lane line,"* and every
   non-asphalt speck still becomes an obstacle blob.

6. **CV false positives were amplified downstream.** Lanes were marked
   lethal + inflated, so one speck-induced lane pixel became a wall that trapped
   the robot in 2–3 m lanes (`lane_following_strategy.md`). Garbage in, wall out.

7. **No temporal confirmation.** Each frame is thresholded independently; a
   one-frame glint off a wet speck becomes a one-frame fake line.

8. **Tangled with one robot.** Thresholds, ROI, class IDs, ZED topic paths, and
   downsample factors were interwoven with the ROS node and tuned to one camera
   mount — not portable.

## How this module fixes it

| Problem | Fix in `lane_cv` |
|---|---|
| 1, 5 color ambiguity | `combine: white_gated` AND-s the white gate with a Sooner-25 *not-asphalt* gate (`segmentation.py`) |
| 2 floor drift | near-field-band adaptive floor, clamped to never drop below the static `white.v_min` (`AdaptiveCfg`) |
| **3 specks → lines** | **`line_filter.py`** — keeps a blob only if it is *line-shaped*: `min_elongation` (long/thin), `max_fill` (not a solid patch), optional orientation + Hough-straightness gates. **This is the core of the speck fix.** |
| 4 manufactured lines | shape filter runs on a **gently opened** mask, *before* any aggressive close, so specks are never welded first |
| 7 flicker | `temporal.window` / `min_hits` N-of-window voting (`pipeline.py`) |
| 8 portability | all the above live in a `LaneConfig` YAML; the package imports no ROS |

The `tools/run.py` panel shows `candidate` (color stage) next to `lane_mask`
(after the shape filter) so you can literally watch the specks get dropped.

---

## Testing / simulation (phase by phase)

A synthetic scene generator + scoring harness lets you check each [TODO](TODO.md)
phase objectively — no real footage needed. Dependency-free (`python`, no pytest).

```bash
python tests/check.py            # run all implemented phases, save panels
python tests/check.py --phase 0  # just Phase 0
python tests/check.py --quiet    # table only
```

`sim/scene.py` renders labelled asphalt scenes with controllable adversaries
(specks, tar cracks, shadow, grass, barrels, curves, dashes) and ground-truth
lane masks. `sim/metrics.py` scores each as **`lane_recall`** (did we keep the
real lines?) and **`false_line_count`** (did clutter become phantom lanes?).
Panels (`input | detected | ground-truth | lane_mask`) land in `_artifacts/`.
Unbuilt phases report `PENDING`; the harness exits nonzero only on a real `FAIL`,
so it doubles as a CI gate as phases land.

## Running on your own images / videos

```bash
# one folder of mixed images + videos, write a review video per clip
python tools/run.py --config configs/default.yaml --input my_testset/ \
    --video-out out_videos/ --no-display

# one clip -> one output video
python tools/run.py --config configs/default.yaml \
    --input drive.mp4 --video-out drive_lanes.mp4 --no-display
```

Behaviour notes: still images are each judged independently (temporal state is
reset between them); each video resets at its start and then accumulates temporal
votes within that clip. `--out <dir>` additionally dumps per-frame PNGs;
`--dump-masks` also writes the raw `*_lane.png` / `*_candidate.png`.

**Expect to tune the profile for your footage.** `configs/default.yaml` is tuned
for dark asphalt with bright paint. On *bright concrete with low-contrast white
lines* the defaults under-detect — see `configs/bright_concrete.yaml`, a profile
fitted to real footage (measured concrete V≈150 vs line V≈205): it drops the
not-asphalt gate (`combine: white`) and lowers the brightness floor to the
measured line brightness. This is the intended workflow — a new surface/venue is
a YAML, not a code change.

## Porting to another car

Copy `configs/default.yaml` → `configs/<your_car>.yaml` and adjust, in rough
order of importance (worked examples: `configs/example_other_car.yaml`, and
`configs/bright_concrete.yaml` which is fitted to real low-contrast footage):

1. **`roi_poly`** — where the sky / hood / horizon sit in *your* camera.
2. **`line.min_area` / `line.min_elongation`** — how big and how thin a real
   lane line is at *your* resolution and mount. Raising `min_elongation` is the
   main dial for rejecting more specks.
3. **`white` + `asphalt` HSV** — your venue's lighting and surface color. Use
   `tools/tune.py` on a representative frame.
4. **`adaptive.band` / `adaptive.k`** — the near-field strip your lanes live in.
5. **`temporal`** — raise `window`/`min_hits` on a noisy sensor.
6. **`meters_per_px`** — optional, for downstream BEV scaling.

## Optional: wiring back into the ROS package

This folder is intentionally standalone. To use it from
`avl_bev_perception`'s Tier 1, construct a `LaneDetector` once in the node and,
inside the per-camera segmentation, replace the white-lane `inRange` block with:

```python
res = self._lane_det.process(bgr)
mask[res.lane_mask > 0] = CLASS_LANE_LINE
```

leaving the barrel/pothole passes as they are. Keep the class-ID mapping in the
node; `lane_cv` deliberately emits a plain binary mask, not class IDs, so it
stays car-agnostic.

## Layout

```
cv_lane_core/
  lane_cv/            the portable core (numpy + opencv only, no ROS)
    config.py         LaneConfig + sub-configs (per-camera detection tuning)
    segmentation.py   color cues: white gate, asphalt-invert, adaptive floor
    line_filter.py    geometric speck rejection  <-- the core detection fix
    pipeline.py       LaneDetector orchestrator + temporal voting + overlay
    bev.py            GroundGrid + Homography/Depth projectors + fuse_grids
    rig_config.py     RigConfig — the one per-vehicle config (grid + cameras)
    rig.py            Rig — N detectors -> project -> fuse -> BevResult
  configs/
    default.yaml            starting profile (dark asphalt + bright paint)
    example_other_car.yaml  worked "port to a second vehicle" example
    bright_concrete.yaml    profile fitted to real low-contrast concrete footage
    vehicle_example.yaml    multi-camera rig: a depth cam + a homography cam
  tools/
    run.py            run on image / folder / video; --video-out, --view, --recursive
    tune.py           live trackbar tuner
  sim/                synthetic labelled scenes + scoring (scene.py, metrics.py)
  tests/
    check.py          phase-by-phase verification harness (no pytest needed)
  adapters/           live-capture wrappers (ONLY place ROS/hardware is allowed)
    usb_cam.py        plain USB / UVC webcam or video file
    ros2_zed.py       ROS 2 node for the ZED X cameras (1 or 3)
  ARCHITECTURE.md     design blueprint: input/output contracts + swappable layers
  CODE_GUIDE.md       plain-language walkthrough of every source file
  ISSUES.md           the problem + root causes      TODO.md  the roadmap
```
