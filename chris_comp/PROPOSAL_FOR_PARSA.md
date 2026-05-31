# Proposal: add `avl_bev_perception` to the IGVC stack (Chris → Parsa)

*Written 2026-05-30 for Parsa to evaluate. Go/no-go is yours — you're the one at the competition with the hardware. Nothing here touches your repo until you decide it's worth it.*

---

## TL;DR

I have a 3-camera BEV perception package that publishes the **same kiwicampus contract your Nav2 already consumes** (`/perception/<cam>/semantic_{mask,confidence,points}` + latched `label_info`). It's gated behind one flag (`kiwicampus.enabled`), so it's drop-in or completely off.

The reason it's worth a look *right now*: your 2026-05-28 lane-following plan needs to split `lane_white` (soft cost) from `barrel_orange`/`pothole` (lethal) — but your production `sooner25` pipeline is **single-class** and can't tell them apart. **My pipeline emits distinct class IDs (lane=1, barrel=2, pothole=3), which is exactly what the soft-lane split requires.** I also produce `left`/`right` masks for the two semantic sources you've already configured in `nav2_params_humble.yaml` but aren't filling yet.

**Honest caveat up front:** I haven't `colcon build`-ed this against the live stack or run it on a bag/cameras — I don't have the hardware. It's parameter/constant-level changes over a known-good node, so it *should* build clean, but you'd be the first to run it. Details in "Test status" below.

---

> **Note on your ongoing tuning:** this integration couples to the kiwicampus *contract* (topic names, organized cloud, shared stamp, mask H×W = cloud H×W, `LabelInfo` QoS, class IDs), not to your tuning values. Inflation, footprint, MPPI batch/vx_max, EKF/GPS, decay times — none of that changes whether my topics plug in. Any specific values quoted below are just context as of 2026-05-30; if they've moved, the integration is unaffected.

## Why this helps *your current* problems

| Your situation (from your 05-28→05-30 work) | What my package gives you |
|---|---|
| `soft_lane` vs `danger` cost split needs lane≠barrel, but `sooner25` is single-class | Multi-class mask: `lane_white=1`, `barrel_orange=2`, `pothole=3` — drop lanes to soft, keep barrels lethal |
| `nav2_params_humble.yaml` declares `left` + `right` semantic sources, but `perception_node` is front-only | Natively 3-camera — fills `left`/`right` with no extra work on your side |
| `perception.yaml` is a graveyard of per-session manual HSV retunes | Auto-HSV calibration at startup (samples lower 30%, derives venue thresholds, falls back if >40% coverage) |
| Risk register wants "keep N largest connected components" to kill `sooner25` asphalt false-positives | Already runs `connectedComponentsWithStats` cleanup in the pipeline |

The class IDs already match your `class_map.yaml` verbatim for `free/lane_white/barrel_orange/pothole/unknown` (0,1,2,3,255) — I aligned to yours back in v3.2.2.

---

## How it integrates (drop-in by design)

My adapter publishes the per-camera contract under a configurable namespace. New param `kiwicampus.topic_prefix` (default `/perception`) controls it:

- Looking at your current `nav2_params_humble.yaml`, your `local_costmap.semantic_layer` **already declares `front`, `left`, and `right` sources** all pointing at `/perception/<cam>/semantic_*` — but your `perception_node` is front-only, so `left`/`right` are configured-but-empty.
- With the default prefix, my node publishes exactly those names. So **running my node for `left`+`right` fills your two empty sources with zero YAML changes and zero collision** (you produce nothing there). That's the literal drop-in.

### Three ways in, by effort

1. **Zero-config drop-in — left+right.** Run my node for `left`+`right` with default `topic_prefix: /perception`, leave your front `perception_node` alone. Your existing `semantic_layer.left`/`.right` config consumes it immediately. No collision, no YAML edit.
2. **Swap front to multi-class.** Stop your front `perception_node`, run mine for `front`. Your `semantic_layer.front` config works unchanged (names match). This is the path that gives you a front mask which separates lane/barrel/pothole — the prerequisite for the `soft_lane` split.
3. **Redundant — run both.** Set `kiwicampus.topic_prefix:=/bev_perception` so my front publishes to `/bev_perception/front/semantic_*` (no collision with your node), then add that as a second `observation_source` under your `semantic_layer`. Costmap merges via `updateWithMax`; if either node dies the other keeps marking.

> **On the soft-lane benefit:** your current YAML still has `danger.classes: [lane_white, barrel_orange, pothole]` all at cost 254 — i.e. the `soft_lane` split from your 05-28 strategy isn't applied yet. So the multi-class value is *latent*: it's what lets you eventually move `lane_white` to a `soft_lane` block while keeping barrels lethal, which your single-class `sooner25` can't support. Until you make that split, modes 2/3 behave the same as your current front (everything lethal). The immediate, today win is mode 1 (left+right coverage).

### The soft-lane YAML it's designed to feed (your plan, for reference)

```yaml
semantic_layer:
  ...
  class_types: ["danger", "soft_lane", "ignored"]
  danger:
    classes: ["barrel_orange", "pothole"]   # lethal — real collisions
    base_cost: 254
    max_cost: 254
  soft_lane:                                # lane becomes a gradient, not a wall
    classes: ["lane_white"]
    base_cost: 180
    max_cost: 220                           # below inscribed-inflated 253
    mark_confidence: 0.6
    samples_to_max_cost: 3
  ignored:
    classes: ["free", "unknown"]
    samples_to_max_cost: 999999
```

My `label_info` advertises two extra IDs you don't list: `person`(4) and `drivable`(5). **Today that's harmless** — they only ever appear if I turn on the (off-by-default) Tier-2 ONNX model, which I won't for IGVC. If you'd rather avoid the activation log noise, add them to an `ignored` block, or I can strip them from my `LabelInfo`.

---

## What could break / how to back out

- **Hard off switch:** `kiwicampus.enabled:=false` (the default). With it off, my node only publishes `/bev/*` and touches nothing of yours.
- **Dependency:** needs `ros-${ROS_DISTRO}-vision-msgs` (for `LabelInfo`). The import is lazy — if it's missing, the adapter disables itself and logs, the node still runs.
- **The 5 silent-drop gates** (stamp pairing, organized cloud, mask H×W = cloud H×W, latched `LabelInfo` QoS, `class_types` coverage): my adapter is built to clear gates 1–4 the same way your `perception_node` does (`max(img,cloud)` stamp, `INTER_NEAREST` resize to cloud shape, TL+RELIABLE label QoS). Gate 5 is your `nav2_params` config, unchanged.
- **PR3:** assumes your kiwicampus plugin has the raytrace-clear patch (it does, per your 05-29/30 config). My package doesn't change that.

---

## Test status — please read before trusting it

I built and reasoned about this off-hardware. **Not yet done (needs your Jetson):**

- Has **not** been `colcon build`-ed against the live workspace.
- Has **not** run on live ZED cameras or a bag.
- Serial sanity check, auto-HSV cal, and the `vision_msgs` fallback path are all **un-exercised** on real data.
- Mount poses in my `bev_config.yaml` come from a team sketch; haven't been reconciled against your URDF (which is still "TODO measure").

So treat this as "should work, you're the first to run it." If you want, the fastest validation is bag replay: `kiwicampus.enabled:=true`, then check `ros2 topic hz /perception/left/semantic_mask` and that mask/points stamps are bit-identical (gate 1).

---

## What I'd need from you to go further

- Which integration mode (left+right drop-in / front swap / redundant) you'd actually want — I'll tailor the launch + config to that. (Mode 3 just needs `kiwicampus.topic_prefix:=/bev_perception` + one extra `observation_source` block — I can write that block for you.)
- Confirmation your ZED v5 topic names match my defaults (`/zed_<cam>/zed_node/rgb/color/rect/image`, `.../point_cloud/cloud_registered`).
- Whether you want me to strip `person`/`drivable` from `LabelInfo` or you'll add them as `ignored`.

The whole package is in the repo I'm sending (`avl_bev_perception_v3_2/`). Standalone bring-up and the full integration writeup are in its `README.md` and my `CLAUDE.md` if you want the deep version.
