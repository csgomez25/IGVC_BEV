# Code guide — `cv_lane_core`

A plain-language walkthrough of every source file, section by section. The aim
is that a new reader can open any file and understand what each part does and
why it exists, without re-deriving it from scratch. For *why the project exists*
see [ISSUES.md](ISSUES.md); for *where it is going* see [TODO.md](TODO.md).

## The one-paragraph mental model

A camera frame comes in. We first ask, by colour, "which pixels *could* be
paint?" — this is permissive and deliberately also lights up bright road specks
and cracks. We then ask, by **shape**, "which of those blobs actually look like
a line?" — long and thin survives, stubby specks are discarded. Finally we
require a detection to persist across a few frames before trusting it. The
colour stage is venue-tunable; the shape stage is the part that fixes the
"specks become fake lanes" problem.

```
frame ─► preprocess ─► colour candidate ─► geometric line filter ─► temporal vote ─► lane mask
         (blur+HSV)    (segmentation.py)    (line_filter.py)         (pipeline.py)
```

---

# Package: `lane_cv/` — the portable core

This package has no ROS or hardware dependencies. It needs only numpy and
opencv (and optionally PyYAML for reading config files).

## `lane_cv/config.py` — all the tunable knobs in one place

Everything that changes between robots or venues lives here as plain data, so
porting to a new vehicle is a config edit, never a code edit.

- **`WhiteCfg`** — the colour range that defines "white paint": low saturation
  (paint is not colourful) and high brightness. `v_min` is a hard floor the
  brightness gate is never allowed to drop below.
- **`AsphaltCfg`** — the colour range that defines drivable asphalt (dim,
  greyish). It is used inverted — "not asphalt" — as an optional second gate
  that helps suppress bright grass or sky. It does **not** reject road specks on
  its own; specks are also "not asphalt". That job belongs to the shape filter.
- **`AdaptiveCfg`** — settings for a brightness threshold that adjusts itself to
  the scene. It measures the average brightness in a near-field strip of the
  image and sets the floor a few standard deviations above it, refreshing every
  few frames. Sampling only the near strip stops the sky or the robot's hood
  from dragging the threshold around.
- **`LineFilterCfg`** — the shape gates, i.e. the speck killer: minimum and
  maximum blob area, minimum *elongation* (length divided by width — real lines
  are long and thin), maximum *fill* (a solid square is not a line), and two
  optional extras (an orientation gate and a straight-segment check).
- **`TemporalCfg`** — how many of the last few frames a pixel must appear in
  before it is accepted, so a one-frame glint never becomes a line.
- **`LaneConfig`** — the top-level object that bundles all of the above plus
  preprocessing settings and a `combine` switch that selects which colour cues
  to use. Its `from_dict`, `from_yaml`, and `to_yaml` methods load and save
  profiles; `from_dict` knows how to rebuild the nested sub-config objects.

## `lane_cv/segmentation.py` — the colour stage ("what could be paint?")

Turns a colour image into a rough binary candidate mask. Intentionally generous:
it would rather include a speck than miss a line, because the next stage removes
the specks.

- **`_AdaptiveFloor`** — a small stateful helper that remembers the current
  self-adjusting brightness floor and only recomputes it every few frames. It
  samples the near-field band, computes `mean + k·sigma`, and never returns a
  value below the static floor from `WhiteCfg`.
- **`preprocess`** — lightly blurs the frame to remove single-pixel noise, then
  converts it to HSV (hue/saturation/value), the colour space the gates work in.
- **`roi_mask`** — builds a "region we keep" mask from the configured polygon,
  used to discard the top of the image (sky, trees, far pavement).
- **`white_gate`** — keeps pixels that are inside the white colour range, using
  the adaptive brightness floor as the lower brightness bound.
- **`not_asphalt_gate`** — marks the asphalt with its colour range, then inverts,
  giving "everything that is not road surface".
- **`candidate_mask`** — the conductor: it computes the brightness floor, applies
  the colour cues according to the `combine` setting (white only, white AND
  not-asphalt, or not-asphalt only), and finally clips the result to the ROI.

## `lane_cv/line_filter.py` — the shape stage (the actual speck fix)

Takes the candidate mask and keeps only blobs that are shaped like a painted
line. This is the section that solves the reported problem.

- **`LineSegment`** — a small record describing one accepted line: its two
  endpoints, length, angle, area, and the elongation/fill measurements that let
  it pass.
- **`_angle_in_gate`** — optional check that a blob's long axis points roughly in
  an expected direction. Off by default, because lane orientation differs
  between vehicles and scenes.
- **`_has_straight_run`** — optional check, using a Hough-line detector, that a
  blob actually contains a straight segment of a minimum length. Off by default.
- **`_segment_from_rect`** — converts the tightest rotated rectangle around a
  blob into a clean centre-line segment (two endpoints plus an angle) for output.
- **`filter_lines`** — the core loop. It first does a *gentle* open to peel off
  isolated noise without merging neighbouring blobs (merging first would weld
  specks into a fake line). It then examines each connected component and keeps
  it only if: its area is in range, it is sufficiently elongated, it is not a
  solid filled patch, and it passes the optional orientation and straightness
  checks. It returns the cleaned mask plus the list of accepted segments.

## `lane_cv/pipeline.py` — the orchestrator

Ties the stages together and holds the only changing state.

- **`LaneResult`** — the output bundle: the final confirmed lane mask, the
  pre-shape-filter candidate mask (useful for debugging), and the line segments.
- **`LaneDetector.__init__` / `reset`** — stores the config and creates the
  adaptive-floor helper and a small ring buffer of recent frames. `reset` clears
  that state, for example when the camera or scene changes.
- **`process`** — the public entry point. It validates the frame, then runs
  preprocess → candidate mask → line filter → temporal confirmation and returns
  a `LaneResult`. This is the single function callers use.
- **`_temporal_confirm`** — keeps a pixel only if it was lit in at least
  `min_hits` of the last `window` frames. While the buffer is still filling it
  does not suppress anything, to avoid a blind first second.
- **`draw_overlay`** — a convenience that paints detected lanes green and draws
  the fitted segments in red on a copy of the input, for human inspection.

## `lane_cv/__init__.py` — the public interface

Re-exports the classes a user actually needs (`LaneDetector`, `LaneConfig`, the
sub-configs, `LineSegment`) so they can be imported directly from `lane_cv`, and
declares the package version. Importing this pulls in no ROS or hardware code.

---

# Package: `sim/` — the synthetic test world

Generates fake-but-labelled camera scenes and scores results against them, so
each development phase can be checked objectively without real footage.

## `sim/scene.py` — the scene generator (the "simulation")

Renders asphalt courses with a known lane corridor and controllable clutter, and
returns the ground-truth lane mask alongside each image.

- **`Scene`** — the output record: the rendered image, the true lane mask, any
  barrel positions, a name, and metadata.
- **`_asphalt`** — paints a grey background with mild random texture to imitate
  road surface.
- **`_lane_points`** — computes the points of a lane line from the horizon to the
  bottom of the image, with an optional sideways bow for curved lanes.
- **`_draw_lane`** — draws a lane line that thins with distance (as a real line
  appears) and writes the same pixels into the ground-truth mask. Supports
  dashes.
- **`_add_specks`** — scatters small bright blobs: pebbles, paint chips, glints —
  the main adversary the shape filter must reject.
- **`_add_cracks`** — draws short jagged bright or dark seams, the classic
  "looks a bit like a line" trap.
- **`_add_shadow`** — darkens a horizontal band, to stress the adaptive
  brightness floor.
- **`_add_grass` / `_add_barrels`** — add green borders and orange drums, i.e.
  off-road clutter and future non-lane targets.
- **`make_scene`** — assembles one labelled scene from the chosen ingredients;
  the difficulty is simply how much clutter is switched on.
- **`default_suite`** — a fixed list of scenes spanning easy to hard (straight,
  dashed, curved, heavy specks, cracks, shadow, grass, and an everything-at-once
  scene), reused by the test harness.

## `sim/metrics.py` — scoring

Turns a detection plus its ground truth into the two numbers that matter.

- **`_kernel`** — builds a small round shape used to allow a few pixels of
  tolerance when comparing masks.
- **`Score`** — the result record (recall, false-line count, component counts)
  with an `ok` helper that checks both numbers against pass thresholds.
- **`score`** — computes **lane recall** (of the true lane pixels, how many we
  found, allowing slight tolerance) and **false-line count** (how many detected
  blobs do *not* line up with a real lane — i.e. phantom lines made from
  clutter). A good detector has high recall and a false-line count of zero.

## `sim/__init__.py` — the public interface

Re-exports `make_scene`, `default_suite`, `Scene`, `score`, and `Score` for easy
import from `sim`.

---

# `tests/` — the phase-by-phase harness

## `tests/check.py` — the verification runner

Runs the scene suite through whatever capability each roadmap phase has, prints
a per-phase pass/fail table, and saves side-by-side panels. It needs only plain
Python (no test framework) and exits non-zero if any *implemented* check fails,
so it can also serve as a continuous-integration gate later.

- **`Check`** — a small record for one result row: phase number, name, status
  (pass/fail/pending), and a detail string.
- **`_panel`** — assembles a four-up comparison image (input, detection overlay,
  ground truth, final lane mask) for visual inspection.
- **`phase0`** — the only fully built phase today: for each scene it runs the
  detector for a few frames (to let temporal voting settle), scores the result,
  records pass/fail against the thresholds, and saves a panel.
- **`PHASE_BUILDERS`** — the registry mapping each roadmap phase to its check
  function. Phases not yet built are listed with `None` and reported as
  "pending", so the table doubles as a progress dashboard.
- **`main`** — parses arguments (run all phases or just one, choose the output
  folder, optionally skip panels), runs the selected phases, prints the grouped
  table with a pass/fail/pending summary, and sets the exit code.

---

# `tools/` — developer utilities

## `tools/run.py` — run on images, videos, or a whole folder

A command-line viewer/exporter for real or saved frames; the quickest way to see
the detector's output on a file. Accepts a single image, a single video, or a
folder containing images and/or videos (optionally recursing into subfolders),
and can save the result as PNG panels or as a playable video.

- **`load_config`** — loads a profile YAML, or the built-in defaults if none is
  given.
- **`make_view`** — renders one frame for display/saving: either the `overlay`
  (lanes drawn on the input) or the three-up `panel` (overlay, candidate mask,
  final lane mask), with on-image labels and a segment count.
- **`process_frame`** — runs one frame through the detector, optionally writes
  the chosen view and raw masks to disk, and optionally shows it in a window.
- **`process_image`** — handles one still image; it resets the detector first so
  temporal state never carries across unrelated photos.
- **`process_video`** — handles one video; it resets at the start (so the
  previous clip can't vote here), then streams frames, and — if `--video-out` is
  set — opens a video writer once the frame size is known and writes each
  rendered frame to a playable output video.
- **`_video_out_path`** — works out where a clip's output video should go: a
  `--video-out` that is a file path is used directly for a single clip; otherwise
  it is treated as a directory and one `<name>_lanes.mp4` is written per input.
- **`gather`** — given a path, returns the lists of images and videos to process
  (a single file yields one entry; a directory is scanned, with `--recursive`
  descending into subfolders).
- **`main`** — parses options (config, input, `--out` PNGs, `--video-out`,
  `--view`, `--recursive`, headless, step-through, dump-masks), gathers the work,
  and dispatches each image and video to its handler.

## `tools/tune.py` — interactive threshold tuner

Opens a window with sliders for the settings that matter most, so a profile can
be dialled in against a representative frame.

- **`main`** — loads an image and a starting profile, creates trackbars for the
  white brightness/saturation, the adaptive factor, and the key shape gates
  (minimum area, minimum elongation, maximum fill), then continuously re-runs the
  detector as the sliders move and shows the overlay with a live segment count.
  Pressing `s` prints the tuned profile as YAML to copy into a new config; `q`
  quits. The two sliders most relevant to the speck problem are minimum area and
  minimum elongation.

---

# `adapters/` — live-capture wrappers (the only place hardware/ROS is allowed)

The core stays import-clean; anything tied to a specific input source lives here
and is loaded only by its own entry-point script, never by the core.

## `adapters/usb_cam.py` — plain USB / UVC webcam (or video file), no ROS

- **`_open`** — opens a camera by index, a device path, or a video file, and
  optionally requests a capture resolution.
- **`_panel`** — builds the live three-up view (overlay with FPS and segment
  count, candidate mask, final lane mask).
- **`main`** — opens the source, then loops: read a frame, run the detector,
  measure a smoothed frame rate, show and/or save the panel. Keyboard controls
  allow quitting, resetting the detector's state, pausing, and saving a frame.
  It fails with a clear message if the device cannot be opened.

## `adapters/ros2_zed.py` — ZED X cameras over ROS 2

A thin ROS node that runs the detector on one or all three ZED cameras and
republishes the results. All vision logic stays in `lane_cv`; this file is glue.

- **`LaneZedNode.__init__`** — declares overridable parameters (the config path,
  the topic-name templates, and an overlay on/off switch), loads the base
  profile, and for each requested camera creates its own detector plus an image
  subscription and the mask/overlay publishers. A separate detector per camera
  means one camera's exposure changes cannot affect another.
- **`_on_image`** — the per-frame callback: convert the incoming ROS image to an
  OpenCV frame, run the detector, publish the mono8 lane mask (and the overlay if
  enabled) with the source timestamp preserved, and log a heartbeat periodically.
  A conversion error is logged and skipped rather than allowed to crash the node.
- **`main`** — reads which cameras to run from the command line, passes any
  remaining ROS arguments through untouched, starts the node, and spins it until
  interrupted.

## `adapters/__init__.py` — boundary note

Contains no executable code, only the documentation that this folder is the sole
place ROS and hardware dependencies may appear, and that nothing here is imported
at package load — so `import lane_cv` never drags in ROS.

---

# How the pieces connect

```
            configs/*.yaml ──► LaneConfig ──┐
                                            ▼
 frame ─► LaneDetector.process ─► segmentation ─► line_filter ─► temporal ─► LaneResult
   ▲                                                                            │
   │                                                                            ▼
 sources:                                                            consumers / tests:
   tools/run.py      (files)                                          tests/check.py  (scored vs sim/)
   adapters/usb_cam  (webcam)                                         tools/tune.py   (interactive)
   adapters/ros2_zed (ZED topics)                                     overlay / mask  (downstream)
```

`sim/` provides labelled inputs and `metrics.py` provides the score, so
`tests/check.py` can certify each phase. The adapters provide real inputs for the
same detector. Everything points at the same `LaneDetector`, which is the single
reusable unit.
