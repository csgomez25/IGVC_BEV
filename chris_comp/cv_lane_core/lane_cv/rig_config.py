"""
rig_config.py — the ONE file a new vehicle edits.

A `RigConfig` describes a whole multi-camera rig as plain data: the shared BEV
grid, plus a list of cameras, each pointing at a detection profile (a normal
`LaneConfig` YAML) and a projection model (homography for any mono camera, or
depth for a stereo/depth camera). Porting the whole stack to a new robot is
editing this file + running the calibration tool — never a code edit.

    rig = Rig.from_yaml("configs/vehicle_example.yaml")   # see rig.py

Load order mirrors LaneConfig: explicit dict > YAML > built-in defaults.
PyYAML is optional; build a RigConfig from dicts/objects directly without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bev import DepthProjector, GroundGrid, HomographyProjector

try:
    import yaml  # optional, same policy as config.py
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    _HAVE_YAML = False


@dataclass
class ProjectionCfg:
    """How one camera's pixels map to the ground. mode = homography | depth."""
    mode: str = "homography"
    # --- homography mode ---
    H: Optional[list] = None              # 3x3 image(u,v) -> ground(x,y) meters
    # --- depth mode ---
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    extrinsics: dict = field(default_factory=dict)   # {x, y, z, yaw} meters/rad
    depth_range: List[float] = field(default_factory=lambda: [0.3, 20.0])
    height_range: List[float] = field(default_factory=lambda: [-0.5, 2.5])

    def build(self):
        """Instantiate the concrete projector from bev.py."""
        if self.mode == "homography":
            if self.H is None:
                raise ValueError("homography projection needs a 3x3 'H'")
            return HomographyProjector(H=np.asarray(self.H, dtype=np.float64))
        if self.mode == "depth":
            e = self.extrinsics or {}
            return DepthProjector(
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                mount_x=float(e.get("x", 0.0)), mount_y=float(e.get("y", 0.0)),
                mount_z=float(e.get("z", 0.0)),
                mount_roll=float(e.get("roll", 0.0)),
                mount_pitch=float(e.get("pitch", 0.0)),
                mount_yaw=float(e.get("yaw", 0.0)),
                depth_range=tuple(self.depth_range),
                height_range=tuple(self.height_range),
            )
        raise ValueError(f"unknown projection mode {self.mode!r}")


@dataclass
class SegmentationCfg:
    """
    Which detection backend a camera uses (the SegmentationProvider seam).
    backend = classical | yolo | road_seg.
    """
    backend: str = "classical"
    profile: Optional[str] = None     # classical: path to a LaneConfig YAML
    model: Optional[str] = None       # yolo / road_seg: weights path
    conf: float = 0.35                # yolo: detection threshold
    device: Optional[str] = None      # yolo / road_seg: 'cpu' | 'cuda:0' | None
    keep: Optional[list] = None        # yolo: subset of class names to keep


@dataclass
class CameraCfg:
    """One camera: a name, a detection backend, and a projection model."""
    name: str
    detect_profile: Optional[str] = None   # back-compat shorthand for classical
    segmentation: SegmentationCfg = field(default_factory=SegmentationCfg)
    projection: ProjectionCfg = field(default_factory=ProjectionCfg)


@dataclass
class RigConfig:
    """Full rig: shared grid + the cameras feeding it."""
    grid: GroundGrid = field(default_factory=GroundGrid)
    cameras: List[CameraCfg] = field(default_factory=list)
    base_dir: str = ""   # so detect_profile paths resolve relative to the YAML

    @classmethod
    def from_dict(cls, d: dict, base_dir: str = "") -> "RigConfig":
        d = dict(d or {})
        grid = GroundGrid(**d["grid"]) if isinstance(d.get("grid"), dict) else GroundGrid()
        cams = []
        for c in d.get("cameras", []):
            proj = c.get("projection", {})
            seg = c.get("segmentation")
            if isinstance(seg, dict):
                seg_cfg = SegmentationCfg(**seg)
            else:
                # Back-compat: a bare `detect_profile` means classical.
                seg_cfg = SegmentationCfg(backend="classical",
                                          profile=c.get("detect_profile"))
            cams.append(CameraCfg(
                name=c["name"],
                detect_profile=c.get("detect_profile"),
                segmentation=seg_cfg,
                projection=ProjectionCfg(**proj) if isinstance(proj, dict) else ProjectionCfg(),
            ))
        return cls(grid=grid, cameras=cams, base_dir=base_dir)

    @classmethod
    def from_yaml(cls, path: str) -> "RigConfig":
        if not _HAVE_YAML:
            raise RuntimeError("PyYAML not installed; use RigConfig.from_dict().")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def resolve(self, profile: Optional[str]) -> Optional[str]:
        """Resolve a detect_profile path relative to the rig YAML's folder."""
        if not profile:
            return None
        return profile if os.path.isabs(profile) else os.path.join(self.base_dir, profile)
