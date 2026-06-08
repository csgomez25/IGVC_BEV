# cv_lane_core — Critical review of the base CV layer

**Scope:** the *base computer-vision* path only — `lane_cv/segmentation.py`,
`lane_cv/line_filter.py`, `lane_cv/pipeline.py`, `lane_cv/config.py`, and the
`ClassicalProvider` in `lane_cv/providers.py`. The BEV projection / fusion / grid
(`bev.py`, `rig.py`) is **out of scope here** by request and only touched where a
base-CV decision leaks into it. Reviewed against the current code and the
project's own synthetic harness (`tests/check.py`).

**Method:** read the pipeline end-to-end, then reproduced behaviour with the
in-repo synthetic suite and targeted probes (numbers below are from
`python3 tests/check.py` and one-off scripts on the same scenes).

---

## Status

**C1 and H1 are FIXED** (see the ✅ notes in their sections). The Phase-0 harness
now passes **8/8** scenes (`dashed` 0.00→1.00, `shadow` 0.23→1.00, `hard_all`
0.65→1.00); `tests/check.py` reports 16/16 checks. The other findings (C2, H2,
H3, M*, L*, D*) remain open.

## TL;DR — the headline (pre-fix)

The harness the repo shipped with **failed 3 of 8 Phase-0 scenes**, while
`ISSUES.md`/`README.md` described the single-camera fix as "landed / verified."
The three failures were not flakiness — they were three distinct, reproducible
design flaws in the base CV (the first two are now fixed):

| Scene | Recall (before → after) | Root flaw |
|---|---|---|
| `dashed` | **0.00 → 1.00** ✅ | the speck-rejecting elongation/fill gates also rejected every dash (C1) |
| `shadow` | **0.23 → 1.00** ✅ | global brightness gates (floor + asphalt AND) erased a dimmed line (H1) |
| `hard_all` | **0.65 → 1.00** ✅ | combination of the above |

```
=== Phase 0 — single-camera lane vs specks ===
  ✓ easy_straight  recall=1.00   ✗ dashed   recall=0.00
  ✓ curve          recall=1.00   ✗ shadow   recall=0.23
  ✓ specks_80      recall=1.00   ✓ grass_glint recall=1.00
  ✓ cracks_30      recall=0.84   ✗ hard_all recall=0.65
```

Below, issues are ranked by how much they threaten real IGVC footage, each with
the evidence, the root cause with file/line, and a concrete fix.

---

## Critical

### C1 — Dashed lane lines are completely erased (recall 0.00) — ✅ FIXED

> **Resolved.** `dashed` now scores recall 1.00. Two changes in
> `line_filter.py` + `config.py`:
> 1. **Collinear dash linking** (`_link_dashes`): components that are line-ish
>    but too short (`dash_min_elongation ≤ elong < min_elongation`) are held and
>    linked into one line when ≥ `dash_link_min_segments` are *collinear*
>    (matching heading AND joined end-to-end, not side by side). Specks/cracks
>    don't form such chains, so clutter stays out — verified `false_lines` did
>    not rise from the linker (identical with `dash_link` on/off).
> 2. **`max_fill` scoped to near-round blobs only** — it no longer rejects
>    *solid elongated* strokes (a merged run of close dashes, or a solid line),
>    which it previously did (the elong-5.2, fill-0.82 blob was being dropped).
>    New knobs in `LineFilterCfg`: `dash_link`, `dash_min_elongation`,
>    `dash_link_max_gap_px`, `dash_link_angle_tol_deg`, `dash_link_min_segments`.

*(original analysis below)*


**Evidence.** `tests/check.py` scene `dashed`: `recall=0.00, components=0`. Probe:
the color stage produces healthy candidate components (areas 192–323 px, all
above `min_area=80`), but **all are dropped by the shape filter**. Measured
per-dash elongation `long/short` = `[4.15, 3.91, 3.54, 3.38, 3.33, …]` against a
gate of `min_elongation: 4.0` → essentially everything fails.

**Root cause.** `line_filter.py:126` rejects any component with
`elongation < min_elongation`. A *dash* is short by construction, so a single
dash is not 4:1 elongated — the very property that rejects specks
(`config.py:98` `min_elongation: 4.0`) also rejects dashes. Worse, the design
**deliberately forbids the obvious workaround**: `line_filter.py:20-22` and the
module docstring refuse to `MORPH_CLOSE` before filtering ("closing first would
weld neighboring specks into a line"). So there is no stage that ever reconnects
collinear dashes, and there is no dash-aware path.

**Why it matters.** IGVC AutoNav uses dashed lane markings. A detector that
scores 0.00 on dashed lines is not deployable, and the gap is structural, not a
threshold tweak — lowering `min_elongation` to keep dashes re-admits the specks
the gate exists to kill.

**Fix.** Add a **collinear-segment linking pass *after* per-component shape
filtering** (not a morphological close before it). The accepted `LineSegment`s
already carry centroid + angle + endpoints; group segments that are (a) close to
collinear in angle and (b) lie along a common axis within a gap tolerance, then
accept the *group* as a dashed line even though no single member is 4:1. This
keeps the speck rejection (specks don't form collinear chains) while recovering
dashes. Expose `dash_link_max_gap_px` / `dash_link_angle_tol_deg` in
`LineFilterCfg`. Add a `dashed`-passing assertion to the harness.

---

### C2 — Temporal confirmation is image-space and not motion-compensated

**Evidence.** `pipeline.py:65-82` votes pixel-by-pixel: a lane pixel is emitted
only if lit in `min_hits` of the last `window` frames *at the same (row,col)*.
The harness "passes" temporal only because `tests/check.py:67` feeds **the same
static frame 4×** (`for _ in range(4): det.process(scene.bgr)`), so every pixel
trivially agrees with itself.

**Root cause.** On a moving robot the camera translates/rotates every frame, so a
genuine lane line sweeps across image pixels between frames. Pixelwise N-of-window
voting therefore **erodes and thins the true, moving lane** and structurally
*favors* whatever is pixel-stable — which on a forward-driving robot is more
likely fixed glare/lens artifacts than the receding road paint. The fix for
ISSUES.md #7 ("one-frame glint") is implemented in the one reference frame where
it cannot be validated.

**Why it matters.** This is the single largest gap between "passes the synthetic
suite" and "works on `test.mp4` / the real robot." It silently degrades the
primary signal exactly when the vehicle is moving, i.e. always.

**Fix (one of):**
- **Preferred:** do temporal confirmation in the **ground/BEV frame after
  projection**, where static world features *are* spatially stable frame-to-frame
  (compensated by ego-motion/pose). This is the natural home once the BEV layer
  lands; until then, gate temporal voting behind a `motion_compensated` flag and
  default `window: 1` for moving cameras.
- **Interim, image-space:** warp the history ring buffer by the inter-frame
  homography from sparse optical flow (or known ego-motion) before voting.
- **Minimum:** document that `temporal.window > 1` is only valid for a static
  camera, and stop feeding the harness a frozen frame — render a translating
  sequence so the test exercises the real regime.

---

## High

### H1 — Global brightness gates erase shadowed lines (recall 0.23) — ✅ FIXED

> **Resolved.** `shadow` now scores recall 1.00. Added a **contrast-relative
> (`combine: tophat`) gate** as the new default: a white top-hat
> (`V − opening(V)`) detects pixels brighter than their *local* neighborhood, so
> a painted line is found by local contrast rather than an absolute V floor —
> robust to both shadow (line's absolute V is low but still locally bright) and
> bright concrete (flat bright background → top-hat ≈ 0). Saturation
> (`white.s_max`) and a soft absolute floor (`contrast.v_min`) still gate out
> colored / very-dark clutter. New `ContrastCfg` (`tophat_ksize`, `min_contrast`,
> `v_min`); `default.yaml` switched `combine: white_gated → tophat`. The absolute
> white/asphalt gates are untouched and still available for the other modes; the
> shadow-erasing asphalt AND-gate is simply no longer on the default path.

*(original analysis below)*


**Evidence.** Scene `shadow`: `recall=0.23`. Probes on the same scene:
- adaptive `V_floor` clamps to **200** (`segmentation.py:50`, "never drop below
  static `white.v_min`"); the shadowed line dips to `V=131` (median 227), so
  **22% of lane pixels sit below the floor** and are cut by `white_gate`.
- the **asphalt AND-gate is the bigger culprit**: in `white_gated` mode the line
  is also ANDed with `not_asphalt`. A dim shadowed white line (low-S, V≈131)
  falls *inside* the asphalt range `v∈[0,210], s∈[0,95]` (`config.py:48-62`), so
  it is classified asphalt and erased. Measured keep-rates on lane pixels:
  `white_gate=78%`, `not_asphalt=59%`, `white_gated(AND)=59%`. The asphalt gate
  alone throws away ~19% of the line; shape-filtering the resulting fragments
  then drops recall to 0.23.

**Root cause.** Two *global, absolute* brightness thresholds (the clamped
`mean+kσ` floor and the fixed asphalt `v_max`) decide "paint vs surface" by an
absolute V value, but a shadowed line is darker than sunlit asphalt — the
absolute ordering inverts. The `bright_concrete.yaml` profile already had to
**disable the asphalt gate** (`configs/bright_concrete.yaml:32`, "the gate
wrongly erased it") for the same reason; that's a symptom, not a coincidence.

**Fix.** Replace the absolute V floor with a **contrast-relative / local** gate:
a morphological top-hat (`MORPH_TOPHAT`) or local-mean adaptive threshold
responds to "brighter than its *neighborhood*," which survives both shadow and
bright concrete without per-venue retuning. This is exactly the
"contrast-relative gate is the real fix" the code comment at
`segmentation.py:48-49` and TODO Phase 1 already point at — it should be
promoted from "TODO" to "the failing-test fix." Keep the asphalt AND-gate
**off** by default (or make it a soft de-weight, not a hard AND) until it is
contrast-relative too.

### H2 — The shape filter cannot distinguish a long crack/joint from a line

**Evidence.** `cracks_30` passes (recall 0.84) only because the synthetic cracks
are *short* (2–4 short segments, `sim/scene.py:67-79`). Real asphalt expansion
joints and sealed cracks are **long, straight, and thin** — they pass *every*
shape gate (`min_elongation`, `max_fill`, even the optional Hough straight-run)
because they are geometrically indistinguishable from paint.

**Root cause.** `line_filter.py` separates paint from clutter purely by *shape*,
which is correct for blobs/specks but provides **no discriminator** against
clutter that is itself line-shaped. Color is the only remaining separator, and a
sun-bleached sealed crack is bright + desaturated like paint.

**Why it matters.** This is the residual of ISSUES.md #1/#3 that the current
"specks → lines" fix does **not** cover; it's the most likely source of phantom
walls on a real course, and the harness doesn't test it.

**Fix.** (a) Add a *long*-crack adversary to `sim/scene.py` so the gap is
visible; (b) lean on the temporal/BEV layer (cracks don't persist as lanes do
across a moving view) and on width consistency (paint has a stable stroke width;
cracks vary); (c) ultimately this is the canonical case for the optional model
(ARCHITECTURE §6) — note it as a known classical-pipeline limit rather than
pretending the shape filter solves it.

### H3 — Curved lanes are a latent failure of the bounding-rect shape test

**Evidence.** `curve` passes at the *mild* curvature in the suite (`curve=0.12`),
but the shape test uses `cv2.minAreaRect` of the whole component
(`line_filter.py:118`). A strongly curved lane is one long component whose
minimum-area rectangle is wide and near-square → **low elongation, high fill** →
rejected. The optional Hough straight-run gate (`line_filter.py:56-65`) would
reject curves outright if enabled.

**Root cause.** Elongation and fill are computed from a straight bounding box; a
curve fills its box. The metric assumes straight strokes.

**Fix.** Measure elongation as **arc-length / mean-width along the contour
skeleton** (or fit a polyline/spline and use its length-to-width ratio), not from
`minAreaRect`. Document that `hough_min_len_px > 0` is incompatible with curved
lanes, or replace it with a piecewise-straight check.

---

## Medium

### M1 — The white gate can go silently, fully blind

`segmentation.py:50` may clamp the adaptive floor to **255**; `white_gate` then
builds `inRange` with a lower V bound of 255, matching only pure-white pixels —
a near-empty mask. The comment acknowledges this ("degrades to a near-empty
mask, not a crash") but there is **no warning, counter, or coverage telemetry**.
The ROS package's `auto_calibrate.py` already has the right pattern (validate
that the mask covers a sane fraction, else fall back / warn). Port that: if
candidate coverage is ~0% or implausibly high for N consecutive frames, log a
throttled warning. Silent blindness is worse than a loud failure.

### M2 — Adaptive-floor statistics are contaminated by non-road content

`_AdaptiveFloor.value` (`segmentation.py:31-50`) samples the near-field band
`[0.55, 0.95]` across the **full image width** — it is **not** restricted to the
ROI polygon, nor to asphalt. Grass borders, orange barrels, and a red bumper in
that band inflate both `mean` and `σ`, pushing the floor up and biasing the
"paint vs surface" estimate. Sample the band **inside `roi_mask`** (and ideally
only over not-asphalt-excluded surface), so the contrast estimate reflects road,
not scenery.

### M3 — `asphalt_inv` combine mode is inconsistent with the others

In `candidate_mask` (`segmentation.py:98-103`): the `asphalt_inv` branch ignores
the computed `v_floor` entirely (fine), but it also **does not check
`cfg.asphalt.enabled`**, while the `white_gated` branch does. So `enabled: false`
silently means different things depending on `combine`. Make `enabled` authoritative
across all modes, or drop it in favor of `combine` being the single switch.

### M4 — ClassicalProvider emits binary confidence, defeating soft-lane cost

`providers.py:71-76`: `confidence = (lit) * 255` — a hard 0/255. ARCHITECTURE §4
and Parsa's field-tested plan both depend on **graded** lane confidence (the
"soft_lane" centerline gradient, base 180 → max 220). The classical path already
has three natural confidence sources it throws away: distance of V above the
floor (contrast margin), the temporal vote count (`votes` in
`pipeline.py:80`), and proximity to a segment centerline. Carry one of these
through as a real 0–255 confidence so the downstream soft-lane semantics have
something to grade on. (Flagged here because it's a base-CV output decision,
even though the consumer is downstream.)

---

## Low / polish

- **L1 — minAreaRect angle convention.** `_segment_from_rect`
  (`line_filter.py:68-84`) relies on OpenCV's `minAreaRect` angle, whose
  convention changed at OpenCV 4.5. The code normalizes via `arctan2(dy,dx)` so
  the emitted `angle_deg` is probably robust, but the `orientation_gate` result
  depends on it; add a unit test pinning expected angles for a known
  vertical/horizontal/diagonal blob across the OpenCV version in use.
- **L2 — No lens undistortion in the CV path.** Wide ZED FOV bows straight paint
  near frame edges, which both weakens the straightness/elongation gates and
  later projection. Intrinsics are available in the rig config; consider
  undistorting (or at least documenting the assumption) before the shape stage.
- **L3 — Per-frame allocations on the hot path.** `not_asphalt_gate` is recomputed
  even when `white_gate` already excludes the region; `collapse_to_bev` rebuilds a
  256-entry LUT every call (`classes.py:78-81`). Minor, but the package advertises
  a ~10–15 ms budget — precompute the obstacle LUT once and short-circuit the
  asphalt gate when `combine == "white"`.
- **L4 — `max_area: 40000` default may clip near lines at ZED resolution.** Fine
  at 640×480 (the sim size); a 1080p near-field lane stroke can exceed it. It's
  configurable, but the default should be documented as resolution-relative or
  expressed as a fraction of frame area.

---

## Process / documentation issues

- **D1 — Docs overclaim vs. the repo's own harness.** `ISSUES.md` ("Verified: 2
  lines + 80 specks + a patch → only the 2 lines survive") and `README.md` present
  the single-camera fix as done, but `tests/check.py` fails `dashed`, `shadow`,
  and `hard_all` out of the box. Either the failing scenes should be acknowledged
  as known-open (with these issue numbers) or the claims softened. Right now a
  reader trusts a green status that isn't green.
- **D2 — The harness can't see the two worst real-world flaws.** It feeds a frozen
  frame (hides C2/temporal) and uses only short cracks (hides H2). Add: a
  translating-sequence scene (for temporal), a long-straight-crack adversary (for
  false positives), and a strong-curve scene (for H3). Wire `dashed`/`shadow`
  back to PASS once C1/H1 are fixed so the suite is a real gate.

---

## Suggested fix order

1. **H1** (contrast-relative gate) — unblocks `shadow`, and is the same fix that
   makes `bright_concrete.yaml`'s asphalt-gate workaround unnecessary. Highest
   value-to-effort.
2. **C1** (collinear dash linking) — unblocks `dashed`; structural but
   self-contained in `line_filter.py`.
3. **C2** (motion-aware temporal) — biggest real-deployment risk; start by
   defaulting `window: 1` for moving cameras and fixing the harness to expose it,
   then move temporal into the BEV frame when that layer lands.
4. **M1/M2** (blindness telemetry + ROI-restricted floor stats) — cheap
   robustness.
5. **H2/H3** (cracks, curves) — add adversaries first so progress is measurable;
   H2 likely waits on the optional model.

Items C1, H1, M1, M2, M3, L1, L3 are all contained within the base-CV files and
need no BEV/ROS work.
