#!/usr/bin/env python3
"""
Phase-by-phase verification harness for cv_lane_core.

Runs the synthetic scene suite through whatever capability each TODO phase has
landed, scores it against ground truth, prints a per-phase table, and saves
visual panels you can eyeball. Dependency-free: plain `python`, no pytest.

    python tests/check.py                      # run all implemented phases
    python tests/check.py --phase 0            # just Phase 0
    python tests/check.py --out _artifacts     # where panels are written
    python tests/check.py --quiet              # table only, no panels

Exit code is nonzero if any implemented check FAILS (PENDING never fails), so
this doubles as a CI gate once a phase is built.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lane_cv import (                                   # noqa: E402
    CLASS_LANE, CLASS_OBSTACLE, DepthProjector, GroundGrid,
    HomographyProjector, LaneConfig, LaneDetector, Rig, RigConfig,
    fuse_grids,
)
from lane_cv.rig_config import CameraCfg, ProjectionCfg  # noqa: E402
from sim import default_suite, score                    # noqa: E402

# Pass thresholds for the Phase-0 lane-vs-specks capability.
P0_MIN_RECALL = 0.70
P0_MAX_FALSE_LINES = 1


@dataclass
class Check:
    phase: int
    name: str
    status: str          # PASS | FAIL | PENDING
    detail: str


def _panel(scene, result, det):
    overlay = det.draw_overlay(scene.bgr, result)
    gt = cv2.cvtColor(scene.gt_lane, cv2.COLOR_GRAY2BGR)
    lane = cv2.cvtColor(result.lane_mask, cv2.COLOR_GRAY2BGR)
    for img, txt, col in ((overlay, "detected", (0, 0, 255)),
                          (gt, "ground truth", (0, 255, 0)),
                          (lane, "lane_mask", (0, 255, 0))):
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    return np.hstack([scene.bgr, overlay, gt, lane])


def phase0(out_dir, save):
    """Single-camera: keep real lanes, reject specks/cracks/shadow clutter."""
    checks = []
    for scene in default_suite():
        det = LaneDetector(LaneConfig.from_yaml(
            os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")))
        # Run a few frames so temporal voting reaches steady state.
        for _ in range(4):
            result = det.process(scene.bgr)
        s = score(scene.gt_lane, result.lane_mask)
        ok = s.ok(P0_MIN_RECALL, P0_MAX_FALSE_LINES)
        checks.append(Check(
            0, f"scene:{scene.name}", "PASS" if ok else "FAIL",
            f"recall={s.lane_recall:.2f} false_lines={s.false_line_count} "
            f"(comps={s.detected_components})"))
        if save:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"phase0_{scene.name}.png"),
                        _panel(scene, result, det))
    return checks


def phase2(out_dir, save):
    """Multi-camera: one config tree builds N detectors + projectors, runs."""
    checks = []
    grid = GroundGrid(x_range=(0.0, 6.0), y_range=(-3.0, 3.0), resolution=0.05)
    rig = Rig(RigConfig(grid=grid, cameras=[
        CameraCfg("front", projection=ProjectionCfg(
            mode="homography", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        CameraCfg("left", projection=ProjectionCfg(
            mode="homography", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    ]))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = rig.process({"front": frame, "left": frame})
    ok = (len(rig.camera_names) == 2 and out.class_grid.shape == grid.shape
          and out.class_grid.dtype == np.uint8)
    checks.append(Check(2, "rig:2cam-one-tree", "PASS" if ok else "FAIL",
                        f"cams={rig.camera_names} grid={out.class_grid.shape}"))

    # A camera absent from this tick's frames must degrade, not crash.
    try:
        out2 = rig.process({"front": frame})   # 'left' missing
        deg_ok = out2.class_grid.shape == grid.shape
    except Exception as e:                      # noqa: BLE001
        deg_ok, e_txt = False, str(e)
    checks.append(Check(2, "rig:dropped-camera", "PASS" if deg_ok else "FAIL",
                        "missing camera skipped, no crash" if deg_ok else e_txt))
    return checks


def phase3(out_dir, save):
    """BEV geometry: grid coords, both projectors, and fusion priority."""
    checks = []
    grid = GroundGrid(x_range=(0.0, 6.0), y_range=(-3.0, 3.0), resolution=0.05)

    # (a) world_to_px must agree with the meters->px homography matrix.
    M = grid.meters_to_px_matrix()
    pts = [(1.0, 0.0), (3.0, 2.0), (5.5, -2.5)]
    agree = True
    for x, y in pts:
        r, c = grid.world_to_px(x, y)
        col, row, _ = M @ np.array([x, y, 1.0])
        agree &= abs(int(r) - round(row)) <= 1 and abs(int(c) - round(col)) <= 1
    checks.append(Check(3, "grid:world_to_px==matrix", "PASS" if agree else "FAIL",
                        f"checked {len(pts)} points"))

    # (b) Depth projector: a principal-point pixel at depth z, yaw=0, lands at
    #     vehicle (z+mount_x, mount_y).  Exact geometry round-trip.
    dp = DepthProjector(fx=700, fy=700, cx=80, cy=60,
                        mount_x=0.3, mount_y=0.0, mount_z=0.6, mount_yaw=0.0)
    label = np.zeros((120, 160), dtype=np.uint8)
    label[60, 80] = CLASS_LANE                 # principal point (cy, cx)
    depth = np.full((120, 160), 3.0, np.float32)
    g = dp.project(label, grid, depth=depth)
    er, ec = grid.world_to_px(3.0 + 0.3, 0.0)  # expect here
    win = g[max(0, int(er) - 1):int(er) + 2, max(0, int(ec) - 1):int(ec) + 2]
    depth_ok = (win == CLASS_LANE).any()
    checks.append(Check(3, "depth:backproject", "PASS" if depth_ok else "FAIL",
                        f"expected lane near grid({int(er)},{int(ec)})"))

    # (b2) Handedness: a pixel RIGHT of image center, level forward camera, must
    #      land to the vehicle's RIGHT (y_veh < 0) — standard, non-mirrored.
    dpr = DepthProjector(fx=700, fy=700, cx=80, cy=60, mount_x=0.3)
    lab_r = np.zeros((120, 160), dtype=np.uint8)
    lab_r[60, 120] = CLASS_LANE                 # u=120 > cx=80 -> image right
    gr = dpr.project(lab_r, grid, depth=np.full((120, 160), 3.0, np.float32))
    x_cam = (120 - 80) / 700.0 * 3.0
    er2, ec2 = grid.world_to_px(3.0 + 0.3, -x_cam)   # expect y_veh = -x_cam (<0)
    center_col = int(grid.world_to_px(3.3, 0.0)[1])
    win2 = gr[max(0, int(er2) - 1):int(er2) + 2, max(0, int(ec2) - 1):int(ec2) + 2]
    hand_ok = (win2 == CLASS_LANE).any() and int(ec2) < center_col
    checks.append(Check(3, "depth:handedness", "PASS" if hand_ok else "FAIL",
                        "image-right -> vehicle-right (non-mirrored)"))

    # (b3) Pitch applied: a 15-deg-down mount projects a principal-point hit
    #      NEARER than a level mount (smaller x_veh -> larger grid row).
    d = np.full((120, 160), 3.0, np.float32)
    lab_c = np.zeros((120, 160), dtype=np.uint8)
    lab_c[60, 80] = CLASS_LANE
    rows_lvl = np.nonzero(DepthProjector(fx=700, fy=700, cx=80, cy=60,
                                         mount_x=0.3, mount_z=0.6)
                          .project(lab_c, grid, depth=d))[0]
    rows_pit = np.nonzero(DepthProjector(fx=700, fy=700, cx=80, cy=60,
                                         mount_x=0.3, mount_z=0.6, mount_pitch=0.2618)
                          .project(lab_c, grid, depth=d))[0]
    pitch_ok = (rows_lvl.size > 0 and rows_pit.size > 0
                and rows_pit.max() > rows_lvl.max())
    checks.append(Check(3, "depth:pitch-applied", "PASS" if pitch_ok else "FAIL",
                        f"pitched row {int(rows_pit.max()) if rows_pit.size else -1} "
                        f"> level {int(rows_lvl.max()) if rows_lvl.size else -1}"))

    # (c) Homography projector (identity H -> image px == ground meters).
    hp = HomographyProjector(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    hmask = np.zeros((10, 10), dtype=np.uint8)
    hmask[2, 3] = CLASS_LANE                    # image (u=3, v=2) -> ground (3, 2)
    gh = hp.project(hmask, grid)
    hr, hc = grid.world_to_px(3.0, 2.0)
    hwin = gh[max(0, int(hr) - 1):int(hr) + 2, max(0, int(hc) - 1):int(hc) + 2]
    homo_ok = (hwin == CLASS_LANE).any()
    checks.append(Check(3, "homography:warp", "PASS" if homo_ok else "FAIL",
                        f"expected lane near grid({int(hr)},{int(hc)})"))

    # (d) Fusion: obstacle (2) beats lane (1) beats background on overlap.
    a = grid.empty(); a[:] = CLASS_LANE
    b = grid.empty(); b[0, 0] = CLASS_OBSTACLE
    fused = fuse_grids([a, b])
    fuse_ok = fused[0, 0] == CLASS_OBSTACLE and fused[1, 1] == CLASS_LANE
    checks.append(Check(3, "fuse:priority", "PASS" if fuse_ok else "FAIL",
                        "obstacle>lane>bg"))
    return checks


# Phase builders. None = not yet implemented -> reported PENDING.
PHASE_BUILDERS = {
    0: ("single-camera lane vs specks", phase0),
    1: ("YAML-only completeness (validate / extends / class-IDs / barrel+pothole)", None),
    2: ("multi-camera (N cams, one config tree)", phase2),
    3: ("BEV fusion (YAML grid + per-cam intrinsics/extrinsics)", phase3),
    4: ("ROS adapters", None),
    5: ("calibration & portability tooling", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=None, help="run a single phase")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "_artifacts"))
    ap.add_argument("--quiet", action="store_true", help="no panel images")
    args = ap.parse_args()

    phases = [args.phase] if args.phase is not None else sorted(PHASE_BUILDERS)
    results = []
    for p in phases:
        label, builder = PHASE_BUILDERS[p]
        if builder is None:
            results.append(Check(p, label, "PENDING", "not yet implemented"))
        else:
            results.extend(builder(args.out, save=not args.quiet))

    # Report.
    width = max(len(c.name) for c in results) + 2
    cur = None
    fails = 0
    for c in results:
        if c.phase != cur:
            cur = c.phase
            print(f"\n=== Phase {c.phase} — {PHASE_BUILDERS[c.phase][0]} ===")
        icon = {"PASS": "✓", "FAIL": "✗", "PENDING": "·"}[c.status]
        print(f"  {icon} {c.name:<{width}} {c.status:<7} {c.detail}")
        fails += c.status == "FAIL"

    total = sum(c.status in ("PASS", "FAIL") for c in results)
    passed = sum(c.status == "PASS" for c in results)
    pending = sum(c.status == "PENDING" for c in results)
    print(f"\n{passed}/{total} checks passed, {pending} phase(s) pending.")
    if not args.quiet:
        print(f"Panels written to {os.path.abspath(args.out)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
