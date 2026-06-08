"""
rig.py — the one object a path planner talks to.

A `Rig` loads a `RigConfig`, runs a `LaneDetector` per camera, projects each
camera's detections onto the shared `GroundGrid`, and fuses them into a single
metric BEV label grid. N cameras, any mount, any mix of homography/depth — all
behind one `process()` call that returns a `BevResult`.

    rig = Rig.from_yaml("configs/vehicle_example.yaml")
    out = rig.process({"front": bgr, "left": bgr2}, depths={"front": depth})
    out.class_grid      # uint8 HxW, vehicle-frame BEV (0=bg, 1=lane, 2=obstacle)
    out.lane_mask       # uint8 0/255 convenience view
    out.obstacle_mask   # uint8 0/255 convenience view

The core is ROS-free: feed it BGR frames (and depth maps for depth cameras)
from anywhere — a ZED node, a USB cam, a video file, a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .bev import CLASS_LANE, CLASS_OBSTACLE, GroundGrid, fuse_grids
from .classes import collapse_to_bev
from .providers import build_provider
from .rig_config import RigConfig


@dataclass
class BevResult:
    class_grid: np.ndarray      # uint8 HxW label grid, vehicle frame
    grid: GroundGrid            # the canvas it was drawn on (for px<->meters)

    @property
    def lane_mask(self) -> np.ndarray:
        return (self.class_grid == CLASS_LANE).astype(np.uint8) * 255

    @property
    def obstacle_mask(self) -> np.ndarray:
        return (self.class_grid == CLASS_OBSTACLE).astype(np.uint8) * 255


class Rig:
    def __init__(self, config: RigConfig) -> None:
        self.cfg = config
        self.grid = config.grid
        self._cams = []   # list of (name, provider, projector, mode)
        for cam in config.cameras:
            self._cams.append((
                cam.name,
                build_provider(cam.segmentation, resolve=config.resolve),
                cam.projection.build(),
                cam.projection.mode,
            ))

    @classmethod
    def from_yaml(cls, path: str) -> "Rig":
        return cls(RigConfig.from_yaml(path))

    @property
    def camera_names(self):
        return [name for name, _, _, _ in self._cams]

    def reset(self) -> None:
        """Clear every camera's temporal/adaptive state (scene change)."""
        for _, provider, _, _ in self._cams:
            provider.reset()

    def process(self, frames: Dict[str, np.ndarray],
                depths: Optional[Dict[str, np.ndarray]] = None) -> BevResult:
        """
        frames : {camera_name: BGR image}. Cameras absent from `frames` are
                 skipped this tick (degraded, not fatal — a dropped camera
                 just stops contributing).
        depths : {camera_name: depth map} — required for depth-mode cameras.
        """
        depths = depths or {}
        contribs = []
        for name, provider, proj, mode in self._cams:
            frame = frames.get(name)
            if frame is None:
                continue
            depth = depths.get(name)
            # Each backend (classical lanes, YOLO objects, road-seg) emits rich
            # class IDs; collapse_to_bev maps them to the coarse grid set, so a
            # detected person/cone becomes CLASS_OBSTACLE automatically.
            seg = provider.infer(frame, depth)
            label = collapse_to_bev(seg.class_mask)
            if mode == "depth":
                if depth is None:
                    continue   # depth camera with no depth this tick — skip it
                contribs.append(proj.project(label, self.grid, depth=depth))
            else:
                contribs.append(proj.project(label, self.grid))

        grid = fuse_grids(contribs) if contribs else self.grid.empty()
        return BevResult(class_grid=grid, grid=self.grid)
