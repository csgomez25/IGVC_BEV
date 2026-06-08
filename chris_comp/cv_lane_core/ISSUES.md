# Issue — lane / line detection is unreliable, and not transferable

**Status:** root-caused. Core single-camera fix landed in `lane_cv/` (see
[README](README.md)). Multi-camera + BEV + YAML-only configurability still open
(see [TODO.md](TODO.md)).

**Scope:** the camera computer-vision layer only — `avl_bev_perception`'s Tier-1
HSV pass and Parsa's `avros_perception` (`hsv` / `sooner25`) pipelines. LiDAR,
localization, planning are out of scope.

---

## Symptom

Painted lane lines are detected inconsistently, and **bright clutter on the road
surface — tar-filled cracks, expansion joints, pebbles, sun-glints, "specks" —
gets reported as lane lines.** Downstream this becomes phantom walls the planner
can't cross. Reported directly by the user; corroborated across the field notes
in `references/parsa_igvc/docs/`.

## Reproduction / evidence

- Parsa switched his **production pipeline off `hsv`** on 2026-05-28 because
  *"per-class HSV kept misclassifying bright concrete/asphalt at the IGVC
  practice course"* (`CLAUDE.md`, § "Does he already have perception").
- `avros_perception/pipelines/hsv.py` carries scars of the same fight — the
  near-field-band hack and comment *"bit us in /tmp/hsv_iter all session."*
- `docs/yaw_diag_session_2026_05_28/lane_following_strategy.md`: lethal-inflated
  lanes (cost 254) trapped the chassis in 2–3 m corridors — i.e. CV false
  positives had outsized downstream cost.

## Root causes

| # | Root cause | Where |
|---|------------|-------|
| 1 | **Color can't separate paint from bright not-paint.** "White lane" = low-S + high-V also matches sunlit concrete, light asphalt, glints, cracks, specks. | `seg_inference.py` `_infer_tier1`; `hsv.py` lane gate |
| 2 | **Adaptive brightness floor is scene-dependent.** `mean + k·σ` drifts with sky/shadow/hood; some frames it sits below the specks. | `hsv.py` adaptive-V block |
| 3 | **No shape discrimination — the direct cause of "specks → lines."** Components filtered by **area only**; a crack-cluster passes min-area and is painted as lane. | both pipelines' `_filter_by_area` |
| 4 | **Morphology manufactures lines.** `MORPH_CLOSE` / `lane_close_w` welds nearby specks into a continuous streak that looks like a dashed line. | `hsv.py` close ops; `seg_inference.py` close |
| 5 | **Single-class collapse loses the discriminator.** `sooner25` (Parsa's default) thresholds asphalt + inverts → one obstacle class; *"cannot tell a barrel from a lane line,"* and every non-asphalt speck becomes a blob. | `sooner25.py` |
| 6 | **Downstream amplification.** Lanes marked lethal+inflated → one speck pixel = a wall. | `nav2_params_humble.yaml` semantic_layer |
| 7 | **No temporal confirmation.** Each frame thresholded independently; a one-frame glint = a one-frame line. | both pipelines |
| 8 | **Not transferable.** Thresholds, ROI, class IDs, ZED topic paths, downsample, and mount poses are interwoven with one ROS node / one robot. | `bev_perception_node.py`, launch files |

## What's been fixed (single camera)

`lane_cv/` addresses 1–4, 7, and the portability half of 8:

- **#3 (the speck fix):** `line_filter.py` keeps a blob only if it is *line-shaped*
  — `min_elongation` (long/thin), `max_fill` (not a solid patch), optional
  orientation + Hough-straightness gates. Specks are stubby → rejected; lines
  pass. Verified: 2 lines + 80 specks + a patch → only the 2 lines survive.
- **#4:** shape filter runs on a *gently-opened* mask **before** any close, so
  specks are never welded first.
- **#1 / #5:** `combine: white_gated` AND-s the white gate with a Sooner-25
  not-asphalt gate.
- **#2:** near-field-band adaptive floor clamped to never fall below the static
  `white.v_min`.
- **#7:** N-of-window temporal voting.
- **#8 (partial):** the whole pipeline is ROS-free and config-driven.

Confirmed on **real footage** (`test.mp4`, bright concrete + low-contrast white
lines): the default profile under-detected (it is tuned for dark asphalt), but a
YAML-only retune (`configs/bright_concrete.yaml`) recovered the lines while still
rejecting barrels/bumper — i.e. a new surface is a config change, not a code
change. The far, lowest-contrast end of a line still fades out; that motivates
the local/contrast-gate item in [TODO.md](TODO.md) Phase 1.

## What's still open

- **#5 fully** — barrel/pothole classes are still the ROS node's job; `lane_cv`
  emits a binary lane mask only.
- **#6** — a costmap-cost concern, not CV; tracked on Parsa's side.
- **#8 fully** — the remaining, larger goal: **multi-camera + BEV fusion driven
  entirely by YAML, with zero code edits to move to another vehicle.** That is
  the roadmap in [TODO.md](TODO.md).
