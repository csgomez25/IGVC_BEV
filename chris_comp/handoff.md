# Handoff — Costmap clearing + sensor drift freeze on loops

*Last updated: 2026-05-21. Status note added 2026-05-30 after re-diffing Parsa's checkout — see banner below.*

> **2026-05-30 reconciliation:** Parsa's field-test commits (05-28→05-30) effectively closed this handoff's "Outcome A": PR3 is applied on his Jetson, `tile_map_decay_time` settled at the `0.3` default (no `5.0` workaround), and the dual-clock race was left as cosmetic per the plan below. The PR3 patch file is still NOT in our `references/` checkout. Separately, his lane handling pivoted — `lane_white` is moving from lethal `danger` (254) to a soft `soft_lane` class (~200); see `KIWICAMPUS_DEBUG.md` §3.

---

## Goal

Make the robot survive **multi-lap loop testing** without the Nav2 local costmap freezing the planner. Right now, when we drive loops with perception on, painted lane cells accumulate forever, the robot icon doesn't translate in the costmap even while it rotates, and the controller eventually deadlocks against the cell pile-up. Ship a fix that:

1. Lets the kiwicampus semantic layer **actively clear** cells the camera "sees through," not just rely on the rolling window scrolling them out.
2. Makes `odom → base_link` translate properly so the rolling window actually moves with the robot.
3. Keeps decay behavior sane (no `tile_map_decay_time: 5.0` band-aid that just makes the frozen pool bigger).

Success criterion: drive 3+ laps of the test loop with perception live, costmap stays bounded (≤ ~50 LETHAL cells in steady-state), no planner freeze, no manual `/clear_*_costmap` calls needed mid-run.

---

## Current state

The user-visible symptom — "lines don't clear, sensor drift, freeze on loops" — is **three stacked bugs**, all already diagnosed by Parsa, two already fixed in his workspace, one fix pending application on ours.

| Bug | Status | Owner of fix |
|---|---|---|
| 1. kiwicampus `semantic_segmentation_layer` is write-only — no clearing path exists | **Applied & stable on Parsa's side** (as of his 2026-05-29/30 commits: `clearing: true` + `raytrace_max_range: 8.0` live on all sources, `tile_map_decay_time: 0.3`). Still **not vendored** in our `references/` checkout — no `patches/` dir. | Parsa (PR3 patch on his Jetson) |
| 2. Dual-clock decay race (`bufferSegmentation` uses sensor stamp, `updateBounds` uses `node->now()`) | Worked around via PR3; proper fix not written | TODO `kiwicampus_align_purge_clocks.patch` |
| 3. EKF #1 had no translation source — robot rotated but didn't translate; rolling window never scrolled | **Fixed in Parsa's stack (2026-04-28)** | Parsa — verified with 14m drive + 360° spin |

What's **confirmed working** right now in Parsa's stack:
- `actuator_node.py` publishes `/wheel_odom` with proper diagonal covariance (`pose_cov[0]=0.001`, `[7]=0.001`).
- `ekf.yaml` subscribes `odom0: /wheel_odom` (broken `twist0: /filter/twist` source removed).
- `nav2_params_humble.yaml` declares per-source `class_types` correctly (`semantic_layer.front.class_types`, not at plugin top level — placement bug 5a is fixed).
- `LabelInfo` publishers use `RELIABLE + TRANSIENT_LOCAL + depth=1` so the late-joining layer learns the class map.

What's **not yet on our side** (BEV repo):
- PR3 raytrace-clear patch is **not vendored** in `references/parsa_igvc/` — it lives only in Parsa's live Jetson workspace at `src/avros_bringup/patches/kiwicampus_pr3_raytrace_clear.patch`. Our `KIWICAMPUS_DEBUG.md` already flags this.
- Our BEV adapter (`bev_perception_node.py`) clears gates 1–4 of the five silent-drop gates, but we haven't verified gate 1 (stamp pairing) end-to-end after the v3.2.2 changes.

---

## Files in play

### Read / referenced (don't modify)

- [`references/parsa_igvc/docs/CHANGELOG_2026-04-28.md`](references/parsa_igvc/docs/CHANGELOG_2026-04-28.md) — dual-clock decay diagnosis, wheel-odom EKF fusion fix.
- [`references/parsa_igvc/docs/CHANGELOG_2026-04-29.md`](references/parsa_igvc/docs/CHANGELOG_2026-04-29.md) — full 5-phase audit of the write-only bug, PR3 patch design, A/B test results.
- [`references/parsa_igvc/docs/CHANGELOG_2026-04-27_field.md`](references/parsa_igvc/docs/CHANGELOG_2026-04-27_field.md) — §4.1 explains why slim launches (identity TFs) make the freeze look worse than it is.
- [`references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml`](references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml) lines 304–360 — canonical per-source semantic_layer config with `clearing: true`, `raytrace_max_range: 8.0`, `tile_map_decay_time: 0.3`.
- [`references/parsa_igvc/src/avros_bringup/config/perception_test_params.yaml`](references/parsa_igvc/src/avros_bringup/config/perception_test_params.yaml) lines 73–83 — same config, also already PR3-aware.
- [`references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py`](references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py) lines 343–348 — reference `max(image_stamp, cloud_stamp)` pattern.

### Touched / will-touch (ours)

- [`avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) — lines `:1014-1058` (gates 1–3 implementation), `:525-554` (LabelInfo QoS, gate 4). No edit pending unless gate 1 verification fails.
- [`KIWICAMPUS_DEBUG.md`](KIWICAMPUS_DEBUG.md) — already documents the 5 silent-drop gates and the 4 required patches; will need a "verified on this rebuild" note after PR3 is applied.
- Kiwicampus plugin source under `~/<workspace>/src/semantic_segmentation_layer/` on the Jetson — this is where PR3 actually applies. Not in our repo; pulled from upstream and patched at build time.

### Patches (sequenced)

Application order matters — driver script is `scripts/apply_kiwicampus_patches.sh` in Parsa's workspace:

1. `kiwicampus_pr1_humble_build.patch` — **required**, adds `#include <deque>`, drops modern-Nav2 CMake targets. Without it the layer doesn't build on Humble.
2. `kiwicampus_pr2_mutex.patch` — **required**, mutex around `temporal_tile_map_` to stop racing observation writes.
3. `kiwicampus_pr3_raytrace_clear.patch` — **required**, **this is the main fix**. 414 lines git-format. Pushed upstream as draft [kiwicampus/semantic_segmentation_layer#5](https://github.com/kiwicampus/semantic_segmentation_layer/pull/5).
4. `kiwicampus_align_purge_clocks.patch` — **TODO, not written**. Currently moot because PR3 makes the decay race largely cosmetic.

---

## What changed / decisions that landed

**Decision: PR3 raytrace-clear is the load-bearing fix, not a decay tweak.** The 2026-04-28 attempt to fix the freeze with `tile_map_decay_time: 1.5 → 5.0` made it strictly worse (7 frozen cells → 329 frozen cells). Decay alone cannot clear cells in a write-only plugin. PR3 mirrors `nav2_costmap_2d::ObstacleLayer::raytraceFreespace` line-for-line: for every observed point, walk Bresenham from sensor origin to point, mark intermediate cells FREE. Verified A/B on Parsa's stack:

| | Before PR3 (decay 1.5) | Before PR3 (decay 5.0 workaround) | After PR3 (decay 1.5) |
|---|---|---|---|
| Cells while observations live | 7 (frozen) | 990 | **42 (steady)** |
| Cells after observations stop | 7 (frozen forever) | 329 (frozen forever) | **0 within one update cycle** |
| Decay actually clears? | ❌ | ❌ | ✅ |

**Decision: revert `tile_map_decay_time` to kiwicampus default (`0.3` in `nav2_params_humble.yaml`, `1.5` in `perception_test_params.yaml`).** With PR3 in place, the 5.0 workaround is actively harmful — it just slows down freshness.

**Decision: don't use the slim `perception_test.launch.py` for loop testing.** Per the 2026-04-27 field changelog §4.1: that launch uses identity static TFs, so `base_link` is anchored at `(0, 0, 0)` regardless of motion. Rolling window doesn't scroll → even with PR3 applied, cells outside the camera FOV will sit forever. **Loop tests use `localization_perception_test.launch.py` or `navigation.launch.py`** so real EKF moves `base_link`.

**Decision: don't try to fix the dual-clock race right now.** It's real (`bufferSegmentation:204` uses sensor stamp, `updateBounds:339` uses wall clock — same observation evaluated on two clocks) but with PR3 the race is mostly cosmetic. Defer `kiwicampus_align_purge_clocks.patch` until after first clean loop test.

**Decision: BEV adapter doesn't need new code.** `bev_perception_node.py` already covers gates 1–4. Gate 5 (`class_types` placement and coverage) is in Parsa's `nav2_params_humble.yaml`, not ours. Known mismatch flagged in `KIWICAMPUS_DEBUG.md` §3: our `LabelInfo` advertises class IDs 4 (`person`) and 5 (`drivable`) which Parsa's `class_types` doesn't list — harmless today because Tier 2 ONNX is off so we never emit those IDs, but **must be addressed before turning Tier 2 on**.

---

## Next steps

### Immediate next step

**Pull `kiwicampus_pr3_raytrace_clear.patch` off Parsa's Jetson workspace and apply it to our local kiwicampus checkout, then rebuild.**

```bash
# On the Jetson, in the workspace that contains semantic_segmentation_layer/
scp parsa-jetson:~/<his-ws>/src/avros_bringup/patches/kiwicampus_pr3_raytrace_clear.patch .

cd src/semantic_segmentation_layer
git apply ../../kiwicampus_pr3_raytrace_clear.patch

cd ../..
colcon build --symlink-install --packages-select semantic_segmentation_layer
source install/setup.bash
```

**Success signal at activation:**
```
PR3 raytrace clearing enabled for source front (raytrace_max=8.00m, raytrace_min=0.00m)
PR3 raytrace clearing enabled for source left  (...)
PR3 raytrace clearing enabled for source right (...)
```

Then verify per-source params are present in the active YAML (`nav2_params_humble.yaml` already has them, lines 325–327):
```yaml
clearing: true
raytrace_max_range: 8.0
raytrace_min_range: 0.0
tile_map_decay_time: 0.3   # revert from any 5.0 workaround
```

Once activation logs are clean, do the suppression A/B: kill observations (e.g. `adaptive_k: 10.0` to disable the lane class), watch cell count drop to 0 within ~200 ms, re-enable, watch it recover.

### Possible outcomes from that step

**Outcome A — clean activation, A/B test passes (expected, ~70% likely).**
Move to live loop test. Launch via `localization_perception_test.launch.py` (or `navigation.launch.py`), drive 3 laps with WebUI joystick, monitor `/local_costmap/costmap` cell count via Foxglove. Pass criterion: cells stay ≤ ~50 LETHAL in steady state, no planner freeze. If it passes, write a CHANGELOG entry and consider the issue closed — move on to the deferred dual-clock patch only if we see weird decay artifacts.

**Outcome B — patch fails to apply cleanly (~15% likely).**
PR3 was developed against post-PR1 (Humble build fix) state. If our checkout is at a different commit than Parsa's, `git apply` will reject. Fall back: check out the exact kiwicampus commit Parsa is on (he should have it pinned in his apply script), or do a 3-way merge with `git apply --3way`. Don't hand-edit the patch — it's 414 lines and the geometry math has to be exact.

**Outcome C — patch applies, builds, activates, but loop test still freezes (~10% likely).**
The remaining suspects are the **5 silent-drop gates** in `KIWICAMPUS_DEBUG.md` §1, specifically:
- **Gate 1 (stamp pairing):** verify with `ros2 topic echo /perception/front/semantic_mask --field header.stamp` vs `…/semantic_points --field header.stamp` — must be bit-identical. If they drift, our `max(rgb, cloud)` logic at `bev_perception_node.py:1035-1058` regressed.
- **Gate 4 (LabelInfo QoS):** `ros2 topic info -v /perception/front/label_info` must report `Reliability: RELIABLE` and `Durability: TRANSIENT_LOCAL`.
- **Bug 3 sanity check:** confirm `/odometry/filtered` actually moves while driving (`ros2 topic echo /odometry/filtered --field pose.pose.position`). If x/y stay at 0, wheel-odom fusion regressed and the rolling window won't scroll regardless of PR3.

**Outcome D — patch applies but `colcon build` fails (~5% likely).**
Most likely missing `#include <limits>` or `#include "nav2_costmap_2d/cost_values.hpp"` — PR3 adds both, but if PR1 isn't fully applied first, the build environment may not match. Apply PR1 explicitly, rebuild, then re-apply PR3.

### After the immediate step lands

- Write `kiwicampus_align_purge_clocks.patch` to properly fix the dual-clock decay race (`semantic_segmentation_layer.cpp:339` should use the buffer's last cloud stamp, not `node->now()`). Reference: `references/parsa_igvc/TODO.md:42`.
- Before enabling Tier 2 ONNX, add `person` (id 4) and `drivable` (id 5) to Parsa's `class_types` in `nav2_params_humble.yaml` — either as `danger` or `ignored` — or those mask pixels get silently dropped at `segmentation_buffer.cpp:212`.
- Vendor the PR3 patch into our repo under `patches/` so a fresh checkout doesn't depend on Parsa's Jetson being reachable.
