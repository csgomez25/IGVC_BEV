"""
bev.py — turn per-camera 2D detections into ONE metric top-down (BEV) grid.

This is the portable, ROS-free Bird's-Eye-View layer. It takes a label mask
from the detector (`lane_cv.LaneDetector`) plus a per-camera projection model
and writes it into a shared ground-plane grid measured in METERS, so downstream
path planning consumes a single vehicle-frame occupancy picture regardless of
how many cameras produced it or where they are mounted.

Two projection models, chosen per camera (see rig_config.py):

  * HomographyProjector — flat-ground image->ground homography. Works with ANY
    camera, NO depth sensor. Calibrated once from >=4 ground-point
    correspondences. Correct only for things ON the ground plane (paint,
    sidewalk edges) — a tall object's top projects to the wrong place.

  * DepthProjector — per-pixel depth back-projection using intrinsics + the
    camera mount pose. Needs a depth/stereo camera, but recovers true 3D so it
    handles object height. The math mirrors the field-validated
    avl_bev_perception node (yaw-only mount model), reimplemented here so this
    folder has ZERO dependency on that ROS package.

Frame convention (REP-103 vehicle frame): X forward, Y left, Z up, meters.
Grid raster: row 0 = far-forward (x_max), col 0 = right-most (y_min), so the
image reads like a top-down map with the robot at the bottom-centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Class IDs painted into the BEV grid. Background is 0 so an empty grid is
# "nothing detected". Kept tiny on purpose; Phase 1 (barrel/pothole) extends it.
CLASS_BACKGROUND = 0
CLASS_LANE = 1
CLASS_OBSTACLE = 2

__all__ = [
    "CLASS_BACKGROUND", "CLASS_LANE", "CLASS_OBSTACLE",
    "GroundGrid", "HomographyProjector", "DepthProjector", "fuse_grids",
]


@dataclass
class GroundGrid:
    """
    The shared metric BEV canvas, in vehicle-frame meters.

      x_range : (min, max) forward distance covered, meters
      y_range : (min, max) left distance covered, meters  (negative = right)
      resolution : meters per pixel

    All projectors write into a raster of this shape; `world_to_px` and
    `meters_to_px_matrix` are the only two coordinate conversions anyone needs.
    """
    x_range: Tuple[float, float] = (0.0, 6.0)
    y_range: Tuple[float, float] = (-3.0, 3.0)
    resolution: float = 0.05

    @property
    def height(self) -> int:
        return max(1, int(round((self.x_range[1] - self.x_range[0]) / self.resolution)))

    @property
    def width(self) -> int:
        return max(1, int(round((self.y_range[1] - self.y_range[0]) / self.resolution)))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def empty(self, dtype=np.uint8) -> np.ndarray:
        return np.zeros(self.shape, dtype=dtype)

    def world_to_px(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Vehicle-frame meters (x fwd, y left) -> (row, col) int arrays."""
        row = (self.x_range[1] - np.asarray(x, dtype=np.float64)) / self.resolution
        col = (np.asarray(y, dtype=np.float64) - self.y_range[0]) / self.resolution
        return row.astype(np.int32), col.astype(np.int32)

    def meters_to_px_matrix(self) -> np.ndarray:
        """
        3x3 homography mapping ground (x, y, 1) meters -> (col, row, 1) pixels.
        Composed with an image->ground homography it gives an image->grid
        homography that `cv2.warpPerspective` can apply in one shot.
        """
        res = self.resolution
        return np.array([
            [0.0,       1.0 / res, -self.y_range[0] / res],   # col from y
            [-1.0 / res, 0.0,        self.x_range[1] / res],   # row from x
            [0.0,       0.0,        1.0],
        ], dtype=np.float64)


@dataclass
class HomographyProjector:
    """
    Flat-ground projector. `H` maps image pixel (u, v, 1) -> ground (x, y, 1)
    meters in the vehicle frame; calibrate it once with cv2.findHomography on
    >=4 known ground points (see tools/calibrate_extrinsics.py — Phase 5).
    """
    H: np.ndarray   # 3x3, image -> ground meters

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.float64).reshape(3, 3)

    def project(self, label_mask: np.ndarray, grid: GroundGrid,
                depth: Optional[np.ndarray] = None) -> np.ndarray:
        """Warp a label mask straight into a grid-shaped label raster."""
        h_total = grid.meters_to_px_matrix() @ self.H
        return cv2.warpPerspective(
            label_mask, h_total, (grid.width, grid.height),
            flags=cv2.INTER_NEAREST, borderValue=CLASS_BACKGROUND,
        )


@dataclass
class DepthProjector:
    """
    Depth back-projection projector with a FULL roll/pitch/yaw mount model.

    A pixel + its depth give a 3D point in the camera optical frame
    (REP-103 optical: x right, y down, z forward). We rotate it into the
    vehicle frame (x forward, y left, z up) by:

        P_veh = R_mount @ R_optical_to_body @ P_optical + mount_xyz

    where `R_optical_to_body` is the fixed optical->body rotation the ZED
    publishes between `<cam>_left_camera_frame_optical` and `<cam>_camera_link`,
    and `R_mount = Rz(yaw) @ Ry(pitch) @ Rx(roll)` is the URDF mount `rpy`
    (`base_link <- <cam>_camera_link`).

    This supersedes the earlier yaw-only model lifted from avl_bev_perception,
    which silently dropped pitch/roll — wrong for the real robot, whose FRONT
    ZED is tilted 15 deg down (see avros.urdf.xacro). It also uses the standard
    (non-mirrored) REP-103 optical->body convention; set roll=pitch=yaw=0 for a
    level forward camera and image-right correctly maps to vehicle-right.

    Only the lit (detected) pixels are projected, so this stays cheap. Classes
    in `ground_classes` are exempt from the lower height gate so painted-flat
    features (lanes) survive even with noisy depth.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_z: float = 0.0
    mount_roll: float = 0.0
    mount_pitch: float = 0.0
    mount_yaw: float = 0.0
    depth_range: Tuple[float, float] = (0.3, 20.0)
    height_range: Tuple[float, float] = (-0.5, 2.5)
    ground_classes: Tuple[int, ...] = (CLASS_LANE,)

    # Fixed optical(x-right, y-down, z-fwd) -> body(x-fwd, y-left, z-up).
    _OPTICAL_TO_BODY = np.array([[0.0, 0.0, 1.0],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0]], dtype=np.float64)

    def __post_init__(self) -> None:
        cr, sr = np.cos(self.mount_roll), np.sin(self.mount_roll)
        cp, sp = np.cos(self.mount_pitch), np.sin(self.mount_pitch)
        cy, sy = np.cos(self.mount_yaw), np.sin(self.mount_yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        # R: camera optical point -> vehicle frame.
        self._R = rz @ ry @ rx @ self._OPTICAL_TO_BODY

    def project(self, label_mask: np.ndarray, grid: GroundGrid,
                depth: Optional[np.ndarray] = None) -> np.ndarray:
        out = grid.empty()
        if depth is None:
            raise ValueError("DepthProjector.project needs a depth map")
        if depth.shape[:2] != label_mask.shape[:2]:
            depth = cv2.resize(depth, (label_mask.shape[1], label_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        vs, us = np.nonzero(label_mask)
        if us.size == 0:
            return out

        z = depth[vs, us].astype(np.float64)
        valid = (z > self.depth_range[0]) & (z < self.depth_range[1]) & np.isfinite(z)
        if not np.any(valid):
            return out
        vs, us, z = vs[valid], us[valid], z[valid]

        # Optical-frame ray * depth -> optical-frame 3D point.
        x_cam = (us.astype(np.float64) - self.cx) / self.fx * z
        y_cam = (vs.astype(np.float64) - self.cy) / self.fy * z
        pts_opt = np.stack([x_cam, y_cam, z], axis=0)         # (3, N)
        veh = self._R @ pts_opt                               # (3, N)
        x_veh = veh[0] + self.mount_x
        y_veh = veh[1] + self.mount_y
        z_veh = veh[2] + self.mount_z

        cls = label_mask[vs, us]
        ground_lut = np.zeros(256, dtype=bool)
        for cid in self.ground_classes:
            ground_lut[int(cid)] = True
        is_ground = ground_lut[cls]

        h_min, h_max = self.height_range
        keep = (z_veh <= h_max) & ((z_veh >= h_min) | is_ground)
        if not np.any(keep):
            return out
        x_veh, y_veh, cls = x_veh[keep], y_veh[keep], cls[keep]

        row, col = grid.world_to_px(x_veh, y_veh)
        in_b = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
        out[row[in_b], col[in_b]] = cls[in_b]
        return out


def fuse_grids(grids: Sequence[np.ndarray]) -> np.ndarray:
    """
    Merge per-camera label grids into one. Higher class ID wins on overlap, so
    an obstacle (2) beats a lane (1) beats background (0) — the safe choice for
    a costmap. All grids must share shape (they came from the same GroundGrid).
    """
    grids = [g for g in grids if g is not None]
    if not grids:
        raise ValueError("fuse_grids needs at least one grid")
    out = grids[0].copy()
    for g in grids[1:]:
        np.maximum(out, g, out=out)
    return out
