# KIWICAMPUS_DEBUG.md

Failure modes and fixes for the `avros_perception → kiwicampus/semantic_segmentation_layer → Nav2 local_costmap` path.

This is the path Parsa's stack uses to turn camera segmentation into costmap cells. It has burned him repeatedly during field tests because the layer **drops bad data silently** — no error, no warning, just an empty costmap with a working `tile_map` debug topic. This file is the cause/symptom/fix table for joint bring-up with our `avl_bev_perception` adapter (`kiwicampus.enabled:=true`).

Source of truth for the upstream history: `references/parsa_igvc/CLAUDE.md:403-436`, `references/parsa_igvc/docs/CHANGELOG_2026-04-28.md`, and `references/parsa_igvc/docs/CHANGELOG_2026-04-29.md`.

---

## 1. The five silent-drop gates

Every one of these can produce "perception runs fine, mask topic at 20 Hz, but costmap is all-FREE." Validate them in order.

### Gate 1 — `mask.header.stamp == cloud.header.stamp`

The layer time-syncs `semantic_mask` + `semantic_confidence` + `semantic_points` via `message_filters.ApproximateTimeSynchronizer` (slop 0.02 s). If stamps don't pair, the entire sync group is dropped.

- **Parsa's fix:** `references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py:343-348` — uses `max(image_stamp, cloud_stamp)` and writes it onto **all three** outgoing messages.
- **Our adapter:** `avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py:1035-1058` — same `max()` trick, applied to mask, confidence, and the relayed cloud.

### Gate 2 — cloud must be organized (`height > 1`)

The layer raytraces by walking `(u,v)` indices into the cloud. An unorganized cloud (`height == 1`) silently produces zero tiles.

- **ZED:** `point_cloud/cloud_registered` is organized by default; do not switch to `cloud_filtered` or any downstream unorganized topic.
- **Our adapter:** `bev_perception_node.py:1014-1020` — explicitly checks `cloud_h > 1` and throttle-warns + skips that camera's publish on this tick if not.

### Gate 3 — `mask.shape == (cloud.height, cloud.width)`

The layer indexes the mask with cloud `(u,v)`. A shape mismatch produces zero hits.

- ZED's `point_cloud_res: COMPACT` produces a ~256×448 cloud on HD1080 input. Image and cloud H×W diverge by default.
- **Parsa's fix:** `perception_node.py:303-307` — resizes BGR to cloud H×W *before* running the pipeline.
- **Our adapter:** `bev_perception_node.py:1022-1025` — resizes the post-pipeline mask to cloud H×W with `cv2.INTER_NEAREST` (categorical-safe; bilinear would invent intermediate class IDs).

### Gate 4 — `LabelInfo` QoS = `RELIABLE + TRANSIENT_LOCAL + depth=1`

The layer is a late joiner. If `LabelInfo` isn't latched, the layer never learns the ID→name mapping and silently rejects every mask pixel.

- **Parsa's publisher:** `perception_node.py:221-234` — explicit `QoSProfile(depth=1, RELIABLE, TRANSIENT_LOCAL)`, published once at startup.
- **Our adapter:** `bev_perception_node.py:525-554` — same QoS, published once per camera in `_init_kiwicampus_adapter()`.
- **Verify on the wire:** `ros2 topic info -v /perception/front/label_info` should report `Reliability: RELIABLE` and `Durability: TRANSIENT_LOCAL`.

### Gate 5 — `class_types: [...]` placement and coverage

Two distinct bugs live here.

**5a. Placement.** `class_types: [...]` + the per-type blocks (`danger:`, `ignored:`) MUST be nested inside the per-source block (`semantic_layer.front.class_types`), not at the plugin top level. README is ambiguous; the plugin source only reads `layer_name.source.class_types`. Symptom: `"no class types defined for source X"` at activation.

**5b. Coverage.** Pixels whose class ID isn't declared in *some* `class_types` block are dropped at `segmentation_buffer.cpp:212`. `free` (id 0) and `unknown` (id 255) are advertised by every `LabelInfo` but must not mark cells — so they need an `ignored` block with `samples_to_max_cost: 999999` to satisfy the class-filter without ever painting LETHAL.

- **Reference YAML:** `references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml:330-367` (front), 368-404 (left), 405-440 (right) — the canonical layout, copy this verbatim.

---

## 2. Patches required on the kiwicampus plugin itself

Parsa's stack uses the `kiwicampus/semantic_segmentation_layer` `humble` branch but **does not work without patches**. Three are documented; one is hypothetical.

| Patch | Status | What it fixes | Origin |
|-------|--------|---------------|--------|
| `kiwicampus_pr1_humble_build.patch` | **required** | Adds `#include <deque>`, removes modern Nav2 imported CMake targets that don't exist on Humble. Without it the layer fails to build. | Upstream PR [kiwicampus/semantic_segmentation_layer#1](https://github.com/kiwicampus/semantic_segmentation_layer/pull/1). Mentioned at `references/parsa_igvc/CLAUDE.md:427`. |
| `kiwicampus_pr2_mutex.patch` | **required** | Mutex protection around `temporal_tile_map_` to prevent racing observation writes from corrupting the map. Reference: `CHANGELOG_2026-04-29.md:519`. | Local to Parsa's `src/avros_bringup/patches/`. |
| `kiwicampus_pr3_raytrace_clear.patch` | **required** | Adds a clearing path. Without it the plugin is **write-only** — cells go LETHAL on observation and never go FREE again until the rolling window scrolls them out. Mirrors `nav2_costmap_2d::ObstacleLayer::raytraceFreespace`, 414 lines, git-format. With this applied, `tile_map_decay_time` can return to upstream default `1.5`. | Pushed as upstream PR [kiwicampus/semantic_segmentation_layer#5](https://github.com/kiwicampus/semantic_segmentation_layer/pull/5). Saved at `src/avros_bringup/patches/kiwicampus_pr3_raytrace_clear.patch`. Reference: `CHANGELOG_2026-04-29.md:5,22-23,126-255`. |
| `kiwicampus_align_purge_clocks.patch` | **TODO, not written** | Dual-clock decay bug: `bufferSegmentation::purgeOldObservations` uses cloud sensor stamp, `updateBounds` uses `node->now()` — same observation evaluated against two clocks. Proper fix is to change `semantic_segmentation_layer.cpp:339` to use the buffer's last cloud stamp. Currently worked around by `tile_map_decay_time: 1.5 → 5.0`. Reference: `references/parsa_igvc/TODO.md:42`. | Not yet written. |

Application order: PR1 first (build), then PR2 (mutex), then PR3 (clearing). Driver: `scripts/apply_kiwicampus_patches.sh` in Parsa's workspace (`CLAUDE.md:77` references it).

**The patches are not vendored in our checkout of `references/parsa_igvc/` — they live in his live Jetson workspace under `src/avros_bringup/patches/`.** If you're bringing the stacks up jointly off-Jetson, you'll need to pull them from his actual workspace.

---

## 3. What our adapter covers, what to verify

Our adapter (`avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py`, gated by `kiwicampus.enabled` in `bev_config.yaml`) is built specifically to clear gates 1-4. Mapping:

| Gate | Adapter line(s) | Covered |
|------|------------------|---------|
| 1. Shared stamp | `:1035-1041` (`max(rgb, cloud)`) + `:1045, 1050, 1057` (write to all three msgs) | yes |
| 2. Organized cloud | `:1014-1020` (height > 1 check + throttled warn) | yes — degraded, not crashing |
| 3. Mask H×W = cloud H×W | `:1022-1025` (`cv2.INTER_NEAREST` resize) | yes |
| 4. `LabelInfo` QoS | `:525-528` (TL + RELIABLE + depth=1) + `:554` (publish once at startup) | yes |
| 5. `class_types` config | **not ours to fix** — it's in Parsa's `nav2_params_humble.yaml` | n/a |

**One known mismatch to flag.** Our `LabelInfo` advertises seven classes (`bev_perception_node.py` `_KIWI_CLASS_NAMES`): IDs 0–3 + 4 (`person`) + 5 (`drivable`) + 255. Parsa's `nav2_params_humble.yaml` only declares `free / lane_white / barrel_orange / pothole / unknown` under `class_types`. IDs **4 and 5 will trigger `CRITICAL ERROR: Class '<name>' from label_info is not defined`** at layer activation.

- **Today:** harmless. Tier 2 ONNX is off (`segmentation.tier2_model_path` unset), so we never emit IDs 4 or 5; the error logs at activation but no cells are affected.
- **Risk:** if anyone turns Tier 2 on without first adding `person` and `drivable` to Parsa's `class_types` (either as `danger` or `ignored`), those mask pixels will be silently dropped at `segmentation_buffer.cpp:212`. Document this when enabling Tier 2.

**2026-05-30 — lane class is going SOFT, not lethal.** Per [`references/parsa_igvc/docs/yaw_diag_session_2026_05_28/lane_following_strategy.md`](references/parsa_igvc/docs/yaw_diag_session_2026_05_28/lane_following_strategy.md), Parsa is splitting `lane_white` out of the `danger` class_type into a new **`soft_lane`** (`base_cost: 180`, `max_cost: 220`, `mark_confidence: 0.6`, `samples_to_max_cost: 3`) on both costmaps, so lanes become a centerline gradient instead of a wall that traps the robot in 2–3 m lanes. `barrel_orange` + `pothole` stay in `danger` (254). This does **not** change our class IDs or the gates above — but it changes what the costmap *does* with our `lane_white` (id 1) pixels. Two implications for us:
> 1. The `class_types` list he expects from `LabelInfo` now spans **three** blocks (`danger`, `soft_lane`, `ignored`) — coverage gate 5b still holds; just confirm `lane_white` lands in `soft_lane` not `danger`.
> 2. **His production `sooner25` pipeline is single-class — it cannot separate barrel from lane, so he physically cannot do the danger/soft split with his own perception.** Our multi-class mask (lane=1 vs barrel=2 vs pothole=3) is exactly what unblocks it. This is the strongest reason to stand up our kiwicampus adapter as a second `observation_source`. See the in-conversation "what I can contribute" analysis.

---

## 4. Bring-up validation — the five checks

Run these in order. Each one fails closed: if check N passes but the costmap is still empty, the bug is at check N+1 or in the kiwicampus plugin itself (see §2).

### Check 1 — adapter is publishing

```bash
ros2 launch avl_bev_perception bev_perception.launch.py \
    --ros-args -p kiwicampus.enabled:=true
ros2 topic list | grep perception
```

**Expect** for each camera in `cameras` config:
```
/perception/<cam>/semantic_mask
/perception/<cam>/semantic_confidence
/perception/<cam>/semantic_points
/perception/<cam>/label_info
```

If empty: `kiwicampus.enabled` didn't take, or `vision_msgs` import failed (check node logs for `kiwicampus.enabled=true but vision_msgs is not importable`).

### Check 2 — LabelInfo is latched with the right QoS

```bash
ros2 topic info -v /perception/front/label_info
ros2 topic echo --once /perception/front/label_info
```

**Expect:**
- `Reliability: RELIABLE` and `Durability: TRANSIENT_LOCAL` on the publisher.
- `--once` returns immediately (latched). Body shows 7 entries in `class_map`.

If QoS is wrong: gate 4 is broken; the layer will never learn class names.

### Check 3 — mask is publishing at perception rate

```bash
ros2 topic hz /perception/front/semantic_mask
ros2 topic hz /perception/front/semantic_points
```

**Expect:** both at `perception.fps` (default 20 Hz), within ~10%.

If mask < cloud rate: some cameras lack an organized cloud (gate 2). Check node logs for `cloud is unorganized` warnings; check `cameras.<cam>.cloud_topic` matches a real organized cloud (`ros2 topic echo --once <topic> | grep height`).

### Check 4 — stamps match (gate 1)

```bash
ros2 topic echo --no-arr /perception/front/semantic_mask --once | grep -A3 header
ros2 topic echo --no-arr /perception/front/semantic_points --once | grep -A3 header
```

**Expect:** `stamp.sec` and `stamp.nanosec` **identical** between mask and points. Within-µs is not enough — `ApproximateTimeSynchronizer` slop is 0.02 s but identical is the contract we provide.

If they differ: a frame was dropped between mask and cloud publish. Sanity-check by running both at low rate (`perc_fps:=5`) and re-comparing.

### Check 5 — Nav2 layer received and is marking

With Parsa's Nav2 up (`navigation.launch.py`):

```bash
ros2 topic echo --once /local_costmap/costmap | grep -c "data:" # smoke check
ros2 topic hz /local_costmap/front/tile_map  # kiwicampus debug topic, per-source
```

**Expect:**
- Layer log line: `received N classes` (N = number of names listed across `class_types`). If you see `CRITICAL ERROR: Class '<name>' from label_info is not defined` — that's gate 5b. Either add the name to `class_types` or remove it from our `_KIWI_CLASS_NAMES`.
- `tile_map` updating at ~`perception.fps`. Tiles visible in Foxglove/RViz spatially matching where the camera sees lanes/barrels/potholes.
- Master `/local_costmap/costmap` has LETHAL cells (value 254) at the same locations.

If `tile_map` has tiles but master grid is all-FREE: kiwicampus plugin itself is the bug. Either PR3 isn't applied (no clearing path → cells stuck), the dual-clock decay bug (workaround: bump `tile_map_decay_time` to 5.0), or `class_types` is at the wrong nesting level (§Gate 5a).

---

## 5. Quick-reference symptom → cause table

| Symptom | First suspect | Where to look |
|---------|---------------|---------------|
| No `/perception/*` topics at all | `kiwicampus.enabled` not set, or `vision_msgs` not installed | Node startup logs |
| Topics exist but `ros2 topic hz` is 0 | RGB or cloud subscription not connecting | `cameras.<cam>.{rgb,depth,cloud}_topic` paths (v4 vs v5) |
| Mask publishing, points not | Cloud is unorganized | `ros2 topic echo --once <cloud> \| grep height` |
| All four topics publishing, layer says "Class X not defined" | Gate 5b: our LabelInfo has IDs Parsa's `class_types` doesn't declare | Add to `class_types` or drop from `_KIWI_CLASS_NAMES` |
| Layer activates clean, `tile_map` has tiles, master grid empty | Gate 5a (placement) **or** missing PR3 (no clearing path) | nav2_params indentation; then `apply_kiwicampus_patches.sh` |
| Master grid marks cells but they're stuck LETHAL forever | Missing PR3 raytrace-clear path | Apply PR3; revert `tile_map_decay_time` to 1.5 |
| Master grid flickers / decays before refresh | Dual-clock decay bug | Workaround: `tile_map_decay_time: 5.0`. Real fix: write `kiwicampus_align_purge_clocks.patch` per `TODO.md:42` |
| Pothole class paints huge false-positive blobs on concrete | HSV pothole thresholds too loose | Parsa's `TODO.md:31` — tighten `pothole_*` H/S/V ranges. Our Tier 1c is Otsu-based but worth re-checking on the same surface |
