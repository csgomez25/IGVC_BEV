"""
classes.py — the driving-scene class registry shared by every provider.

Every `SegmentationProvider` (classical, YOLO, road-seg) emits a class mask that
uses THESE ids, so the rest of the stack (BEV projection, fusion, costmap) never
cares which backend produced a pixel. IDs 0/1/2 deliberately mirror bev.py's
coarse set so the classical lane detector's output is already valid here.

  0      background
  1      lane            (classical)
  2      obstacle        (classical barrel / generic)
  3-9    reserved for road-segmentation surfaces (road, sidewalk, ...)
  10+    detected objects (person, car, ...)  — from YOLO / object detectors

`collapse_to_bev()` maps this rich set down to the 3-class BEV grid: anything
flagged `is_obstacle` becomes CLASS_OBSTACLE (a person and a barrel are both
"don't drive here"); lane stays lane; everything else is free space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bev import CLASS_BACKGROUND, CLASS_LANE, CLASS_OBSTACLE


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str
    is_obstacle: bool   # True -> collapses to CLASS_OBSTACLE in the BEV grid


DRIVING_CLASSES = [
    ClassDef(0,  "background",     False),
    ClassDef(1,  "lane",           False),
    ClassDef(2,  "obstacle",       True),
    ClassDef(3,  "road",           False),
    ClassDef(4,  "sidewalk",       False),
    ClassDef(10, "person",         True),
    ClassDef(11, "bicycle",        True),
    ClassDef(12, "car",            True),
    ClassDef(13, "motorcycle",     True),
    ClassDef(14, "bus",            True),
    ClassDef(15, "truck",          True),
    ClassDef(16, "traffic_light",  True),
    ClassDef(17, "stop_sign",      True),
    ClassDef(18, "dog",            True),
    ClassDef(19, "cat",            True),
]

NAME_TO_ID = {c.name: c.id for c in DRIVING_CLASSES}
ID_TO_NAME = {c.id: c.name for c in DRIVING_CLASSES}
_OBSTACLE_IDS = [c.id for c in DRIVING_CLASSES if c.is_obstacle]

# COCO (the 80-class set YOLO ships pretrained on) index -> our class name.
# Driving-relevant subset only; any other COCO detection is dropped by default.
COCO_TO_DRIVING = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic_light",
    11: "stop_sign",
    15: "cat",
    16: "dog",
}


def collapse_to_bev(class_mask: np.ndarray) -> np.ndarray:
    """Rich driving classes -> the coarse BEV set (bg / lane / obstacle)."""
    out = np.full(class_mask.shape, CLASS_BACKGROUND, dtype=np.uint8)
    out[class_mask == NAME_TO_ID["lane"]] = CLASS_LANE
    obstacle_lut = np.zeros(256, dtype=np.uint8)
    for cid in _OBSTACLE_IDS:
        obstacle_lut[cid] = CLASS_OBSTACLE
    np.maximum(out, obstacle_lut[class_mask], out=out)
    return out
