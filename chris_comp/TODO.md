# IGVC_BEV — Work Log & TODO

Single source of truth for what's been done, what's pending, and what's verified vs. assumed. Organized by status; within each section, ordered by priority.

If you check items off, **delete them** rather than striking through — keeps the file scannable.

*Last updated: 2026-06-01. Re-diffed against Parsa's checkout 2026-05-30 — see "What I can contribute to Parsa's stack" below.*

---

## Done — verified

These edits are on disk and have been sanity-checked (YAML parses, grep sweeps clean, named-constant routing confirmed). **Nothing here has been built with `colcon` or run on the live robot yet** — see "Verified" subsection below for the actual checks performed.

### Parsa handoff prep + drop-in adapter param (2026-06-01)

- **`kiwicampus.topic_prefix` param added** to [bev_perception_node.py](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) (declared) + [bev_config.yaml](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) (default `/perception`). The adapter builds contract topics as `{prefix}/{cam}/semantic_*` (slash-normalized); startup banner prints the active prefix. Lets the node be a zero-config drop-in for Parsa's `left`/`right` sources, or coexist with his front node under `/bev_perception`. *Verified:* `ast.parse` + `yaml.safe_load` clean; no stray hardcoded `/perception/` in active code (comments/docstrings only).
- **[PROPOSAL_FOR_PARSA.md](PROPOSAL_FOR_PARSA.md) written** — self-contained go/no-go doc for Parsa: value vs. his single-class `sooner25`, three integration modes by effort, collision caveat, honest "not yet built/run on hardware" status, churn-insulation note (couples to the contract, not his tuning).
- **Repo reorganized + uploaded.** Everything moved under `chris_comp/`; root [`.gitignore`](.gitignore) excludes `references/parsa_igvc/` (Parsa's IP / 165 MB / nested git), caches, `*.tgz`. Pushed to https://github.com/csgomez25/IGVC_BEV (`origin`/main).
- **All `.md` docs reconciled** against Parsa's 2026-05-28→05-30 field-test commits (sooner25 default, soft-lane pivot, MPPI/EKF/GPS/footprint/inflation changes). See CLAUDE.md "Parsa-stack changes since 2026-05-21".

### v3.2.2 integration prep + standalone test bringup (2026-05-13)

- **`setup.py` version bumped to 3.2.2.**

- **Standalone test launch + recording tools.** New [`launch/vision_test.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/vision_test.launch.py) brings up only ZED cameras + BEV node + RViz, with `use_bag:=true` to swap live cameras for a bag replay. New [`tools/record_session.sh`](avl_bev_perception_v3_2/avl_bev_perception/tools/record_session.sh) records the minimum vision topic set (rgb rect, camera_info, depth_registered per camera + `/tf` + `/tf_static`) into mcap+zstd. Scripted disk-free check; per-camera selection via second arg.

- **Startup ZED serial sanity check.** New `_check_zed_serials()` in [bev_perception_node.py](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py): tries `pyzed.sl.Camera.get_device_list()` for real serials; falls back to a 5-second `CameraInfo`-presence timer if the SDK isn't importable. Warns (never errors). Expected mapping lives in `EXPECTED_SERIALS` class constant — must stay in sync with `zed_cameras.launch.py` `CAMERA_BINDINGS`.

- **READMEs + CLAUDE.md updated.** Top-level [README](README.md) and in-package [README](avl_bev_perception_v3_2/avl_bev_perception/README.md): serial table footnoted with v3.2.2 verification, class table includes IDs 4/5/255 with Parsa-mapping cross-reference, ZED topic paths bumped to v5.x, "Standalone vision testing" subsection added showing both modes. [CLAUDE.md](CLAUDE.md): authoritative-serials section rewritten to celebrate alignment, integration "Conflicts" section marks items 1–3 resolved, new "Standalone vision testing" subsection. Stale v4 path in [`tools/calibrate_hsv.py`](avl_bev_perception_v3_2/avl_bev_perception/tools/calibrate_hsv.py) docstring fixed.

- **Class IDs renumbered to match Parsa's [`class_map.yaml`](references/parsa_igvc/src/avros_perception/config/class_map.yaml).**
  `CLASS_POTHOLE` moved 4 → 3, `CLASS_PERSON` moved 3 → 4, `CLASS_UNKNOWN = 255` added.
  Touched: [seg_inference.py](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/seg_inference.py) (constants, `OBSTACLE_CLASSES`, header docstring) and [bev_perception_node.py](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) (fallback constants, import list, `_SEG_COLOR_LUT`, `_get_seg_colors()`).
  *Compatibility:* Tier 2 ONNX is off, so the person-ID swap has no runtime effect today. If a checkpoint ever exists trained on the old IDs, retrain.

- **ZED topic paths parameterized with v5.x defaults.**
  New per-camera params `cameras.<cam>.{rgb,depth,info}_topic`, declared in [bev_perception_node.py:292-315](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py#L292-L315) and set in [bev_config.yaml:37-67](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml#L37-L67).
  Default rgb path is now `/zed_<cam>/zed_node/rgb/color/rect/image` (v5.x). v4.x users override via YAML or `ros2 param set`.
  *Limitation:* params read once at startup; runtime `ros2 param set` won't move subscriptions.

- **Camera serial mapping aligned with Parsa's authoritative per-port enumeration (verified 2026-04-24 on his side).**
  Left swapped: 49910017 → 43779087. Right swapped: 43779087 → 49910017. Front (42569280) unchanged. Mount poses are keyed by position and stayed in place.
  Touched: [zed_cameras.launch.py](avl_bev_perception_v3_2/avl_bev_perception/launch/zed_cameras.launch.py), [bev_config.yaml](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml), [tf_static.launch.py](avl_bev_perception_v3_2/avl_bev_perception/launch/tf_static.launch.py) (stale comment labels).

### Docs (earlier this session)

- **Created [IGVC_BEV-main/CLAUDE.md](CLAUDE.md)** with package architecture, build/run, authoritative-serials/mount-pose section, and a full "Integration with Parsa's Stack" section walking through the end-to-end pipeline, kiwicampus contract, topic differences, TF tree, class-ID mismatch, and three conflicts to reconcile.

### Verified

- `yaml.safe_load` round-trips `bev_config.yaml` after edits #2 and #3.
- `grep` across `.py` + `.yaml` + `.xml` shows every position↔serial reference agrees (Left↔43779087, Front↔42569280, Right↔49910017).
- All class-ID references in `seg_inference.py` and `bev_perception_node.py` route through named constants — no remaining numeric literals leaked through the rename.
- No stale `image_rect_color` paths in active code (only in v4-override doc-comments).
- `python3 -m ast.parse` passes on both [`bev_perception_node.py`](avl_bev_perception_v3_2/avl_bev_perception/avl_bev_perception/bev_perception_node.py) and the new [`launch/vision_test.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/vision_test.launch.py).
- `bash -n` passes on [`tools/record_session.sh`](avl_bev_perception_v3_2/avl_bev_perception/tools/record_session.sh); executable bit set.

### Not yet verified — must do before/at competition test

- **Has not been `colcon build`-ed** since edits started. Should compile cleanly (pure parameter + constant + new-file additions) but unconfirmed.
- **Has not been launched** against live cameras or a bag.
- **Serial sanity check** has not been run — `pyzed` import path and fallback timer path both untested.
- **`vision_test.launch.py use_bag:=true`** mode has never been exercised. We need an actual recorded bag first.
- **`tools/record_session.sh`** depends on the rosbag2 mcap plugin (`ros-humble-rosbag2-storage-mcap`); falls through to default sqlite3 if missing but the explicit storage flags will error out — confirm the plugin is installed on the Jetson before relying on the script for the first competition recording.
- **`ImportError` fallback branch** in `bev_perception_node.py` for `seg_inference` not exercised.

---

## In progress

(none — Phase 1, 2, and 3 are landed)

## Pending — medium priority (post-Phase 1/2/3)

Improvements that would noticeably help the competition but aren't on the current plan.

### Perception correctness / field-validation

- [ ] **Validate Tier 1c (Otsu) against Parsa's "concrete-as-pothole at 14% of frame" failure.** *(Partly mooted 2026-05-30: this HSV-pipeline failure is exactly why Parsa abandoned `hsv` for `sooner25` and neutered his pothole class — V-floor 250. His production no longer runs the pipeline that had this bug.)* Still worth checking my Otsu pass + auto-HSV-cal don't reproduce it: record a bag of concrete walkway near our test course, replay through my node, measure pothole-class pixel fraction.
- [ ] **Auto-HSV calibration regression check.** No proof it succeeds (vs. silently falling back) on a real outdoor frame. Add a single-frame test fixture image and assert it computes thresholds rather than tripping `MAX_MASK_COVERAGE`.
- [ ] **Live HSV tuning via `on_set_parameters_callback`.** Parsa added this in his perception_node so field tuning doesn't need a restart. Worth porting to my node — HSV thresholds, BEV grid bounds, obstacle dilation. Pattern reference: [parsa_igvc/.../perception_node.py:255-291](references/parsa_igvc/src/avros_perception/avros_perception/perception_node.py).

### Integration to Nav2 (the redundancy/fail-safe story)

- [ ] **Emit kiwicampus contract per camera as a parallel output.** Add `/perception/<cam>/semantic_{mask,confidence,points}` + `label_info` publishers to my node so it can be added as a second `observation_source` alongside Parsa's existing perception_node in his [`nav2_params_humble.yaml`](references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml). Costmap max-overlap handles voting; if either perception_node dies, the other keeps marking. Adds `vision_msgs` dep to [package.xml](avl_bev_perception_v3_2/avl_bev_perception/package.xml).
- [ ] **`/bev/health` aggregate topic.** Float32 or Bool combining per-camera stale-frame detection, perception loop latency, and lane-pixel count. Lets a downstream planner / safety monitor gate on "is BEV trustworthy right now?".
- [ ] **`/bev/camera_health/<cam>` per-camera Bool.** Trip false when a ZED stops sending frames for N consecutive ticks and exclude that camera from the mosaic, rather than poisoning it with stale depth.

### Performance

- [ ] **Push `perception.fps` to 30 and watch `/bev/perception_latency_ms`.** If the budget blows past ~30 ms with 3 cameras, parallelize the projection step (currently the seg is parallel across cameras but the projection writes the shared BEV canvas serially). Per-camera local canvases → bitwise-max merge is the cheapest fix.

---

## What I can contribute to Parsa's stack (added 2026-05-30)

Ranked by leverage against his *current* (post-05-30 field-test) pain points. The unifying theme: his production perception regressed to `sooner25` (single-class) and his whole 05-28 lane-following pivot is bottlenecked on perception he can't produce himself.

- [ ] **HIGH — Multi-class mask as a second `observation_source` (unblocks his soft-lane plan).** His `soft_lane` vs `danger` split *requires* telling a lane from a barrel, which his single-class `sooner25` cannot do. My `seg_inference.py` already emits distinct IDs (lane=1, barrel=2, pothole=3). Stand up the existing `kiwicampus.enabled` adapter and add `/perception/<cam>/semantic_*` as a parallel `observation_source` in his `nav2_params_humble.yaml`. He keeps `sooner25` as a single-class fallback; my mask carries the semantic split his cost-class design needs. Pairs with `KIWICAMPUS_DEBUG.md` §3.
- [ ] **HIGH — Provide the left + right `semantic_layer` sources he already configured but can't fill.** His `nav2_params_humble.yaml` (2026-05-29) declares front/left/right semantic sources, but his `perception_node` is single-camera. My package is natively 3-camera — wire my adapter's `left`/`right` per-camera publishers to those two empty sources.
- [ ] **MED — Auto-HSV calibration, ported to his node.** His `perception.yaml` is a graveyard of per-scene manual HSV retunes (every field session = re-tune). My `auto_calibrate.py` (sample lower 30%, derive venue thresholds, fall back if >40% coverage) directly removes that toil. Either port the routine into his `perception_node` or have him consume my mask instead.
- [ ] **MED — Connected-component spatial filter for his sooner25 false-positives.** His `lane_following_strategy.md` risk register explicitly asks for "keep N largest connected components" to stop soft-cost asphalt false-positives flooding the corridor. My `seg_inference.py` already runs `connectedComponentsWithStats` (`:242`, `:253`) — factor that out as a shared cleanup he can call on the sooner25 mask.
- [ ] **LOW — `/bev/health` + per-camera staleness as a Nav2 trust gate.** Lets his planner/safety monitor know when to stop trusting camera marks (stale frames, low lane-pixel count). Already a pending item below; flag its value to him.

## Pending — low priority / future

- [ ] **Tier 2 ONNX hook.** Stays off for IGVC 2026; revisit only if Tier 1 fails a real field test. See "ONNX decision" note in conversation 2026-05-13.
- [ ] **Class set extensions in [config/class_map alignment].** If Parsa's planner ever wants to consume `CLASS_PERSON` (4) or `CLASS_DRIVABLE` (5), entries need to be added to *his* [class_map.yaml](references/parsa_igvc/src/avros_perception/config/class_map.yaml) and the `class_types` blocks in his [nav2_params_humble.yaml](references/parsa_igvc/src/avros_bringup/config/nav2_params_humble.yaml). That's a his-side change; coordinate before doing it.
- [ ] **Unit-test projection LUT math.** A small fixture-based test (synthetic intrinsics + known XYZ point) would catch any future regression to the LUT cache. Needs the `tests/` directory + `colcon test` wiring that doesn't exist yet.
- [ ] **Investigate whether to publish TF for `zed_<cam>_camera_center` from my `tf_static.launch.py`.** Parsa's URDF already publishes these via `zed_macro.urdf.xacro`. Running both produces a TF parent conflict on `zed_<cam>_camera_center`. Acceptable for vision-only standalone bringup, but a hazard when both stacks come up together — decide whether mine should be conditional on a launch arg or just dropped entirely.

---

## Known issues / risks to track

Not action items per se — context for future debugging.

- **Mount poses haven't been physically re-verified after the v3.2.2 serial swap.** Position assignments stayed (Left = `mount_yaw +90°`, etc.). If anyone previously *physically swapped cameras between mounts* to compensate for the old wrong serial mapping, the actual robot may now have left/right cameras flipped vs. [bev_config.yaml](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml). Diagnose by viewing each `/zed_<cam>/...` feed and confirming the physical viewing direction matches what the namespace claims.
- **Params load once.** Topic-path and threshold params are read at `_setup_cameras` / `_init_segmentation` startup; `ros2 param set` after launch will not move the subscriptions. Fixed in the "live HSV tuning" pending item above.
- **`pyzed` import is optional.** The forthcoming edit #4 serial check uses `pyzed.sl` if available, else falls back to a topic-presence probe. The fallback is informational only — won't actually verify serials.
- **Mount-pose YAML vs Parsa's URDF.** My [bev_config.yaml](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) mounts come from the team sketch (inches → REP-103 m); Parsa's URDF mounts are flagged "TODO measure" in his [TODO.md](references/parsa_igvc/TODO.md) and may not agree once measured. Source of truth needs to be picked before both stacks run together.
- **READMEs are stale.** Both READMEs still describe the old v3.2.1 serial mapping and old class IDs. Until edit #7 lands, cite [`bev_config.yaml`](avl_bev_perception_v3_2/avl_bev_perception/config/bev_config.yaml) and [`zed_cameras.launch.py`](avl_bev_perception_v3_2/avl_bev_perception/launch/zed_cameras.launch.py) instead.

---

## Decisions made (so we don't relitigate)

- **No Tier 2 ONNX for IGVC 2026.** Tier 1 (HSV + Otsu) covers lane, barrel, pothole — the three IGVC obstacle classes. LiDAR catches anything tall the cameras miss. ONNX costs 10–30 ms/frame and a model-training rabbit hole right before competition. Revisit only if Tier 1 fails a real field test.
- **Class ID 0–3 + 255 must mirror Parsa's `class_map.yaml`.** IDs 4 (person) and 5 (drivable) are local extensions and stay reserved; harmless to Parsa's costmap.
- **Parsa's per-port-verified serial mapping (2026-04-24) is authoritative.** Adopted verbatim in v3.2.2.
- **Dual-output integration story.** Long-term, my package keeps `/bev/*` (for standalone use) AND emits Parsa's kiwicampus contract (for plugging into his Nav2). Same code, two consumers. Not yet implemented but planned.
