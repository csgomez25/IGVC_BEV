"""
Synthetic IGVC scene generator — ground-truth "simulation" for testing.

Renders top-down-ish forward-camera frames of an asphalt course with a known
lane corridor, plus controllable adversarial clutter: bright road specks, tar
cracks, shadows, grass borders, and orange barrels. Every scene returns its
ground-truth lane mask so the harness can measure recall (did we keep the real
lines?) and false-line count (did clutter become phantom lanes?).

No ROS, no files — pure numpy + opencv, deterministic given a seed. This is the
adversary the line detector has to beat; difficulty knobs map straight to the
ISSUES.md root causes (specks=#3, shadow=#2, grass/glints=#1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Scene:
    bgr: np.ndarray                       # rendered BGR frame
    gt_lane: np.ndarray                   # uint8 0/255 — TRUE lane-line pixels
    barrels: List[Tuple[int, int, int]] = field(default_factory=list)  # (x,y,r)
    name: str = "scene"
    meta: dict = field(default_factory=dict)


def _asphalt(h, w, rng, base=80, texture=7):
    img = np.full((h, w, 3), base, np.uint8)
    noise = rng.normal(0, texture, (h, w, 1)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _lane_points(h, w, x_top, x_bot, curve=0.0, n=60):
    """Polyline from the horizon (37% down) to the bottom, optional sideways bow."""
    ys = np.linspace(int(h * 0.37), h - 1, n)
    t = (ys - ys[0]) / max(1.0, (ys[-1] - ys[0]))
    xs = x_top + (x_bot - x_top) * t + curve * np.sin(t * np.pi) * w
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _draw_lane(img, gt, pts, width_bot=14, color=(243, 243, 243), dashed=False):
    """Draw a lane line that thins with distance; write the same into gt."""
    n = len(pts)
    for i in range(n - 1):
        if dashed and (i // 3) % 2 == 1:
            continue
        thick = max(2, int(width_bot * (0.35 + 0.65 * i / n)))
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thick, cv2.LINE_AA)
        cv2.line(gt, tuple(pts[i]), tuple(pts[i + 1]), 255, thick)


def _add_specks(img, rng, n, y_lo, y_hi, bright=(232, 232, 232)):
    """Small bright blobs — pebbles / paint chips / glints (ISSUES #3)."""
    h, w = img.shape[:2]
    for _ in range(n):
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(y_lo, y_hi))
        r = int(rng.integers(2, 6))
        cv2.circle(img, (x, y), r, bright, -1, cv2.LINE_AA)


def _add_cracks(img, rng, n, y_lo, y_hi):
    """Short jagged bright/dark tar seams — the classic 'looks like a line' trap."""
    h, w = img.shape[:2]
    for _ in range(n):
        x0, y0 = int(rng.integers(20, w - 20)), int(rng.integers(y_lo, y_hi))
        pts = [(x0, y0)]
        for _ in range(int(rng.integers(2, 4))):
            pts.append((pts[-1][0] + int(rng.integers(-18, 18)),
                        pts[-1][1] + int(rng.integers(-14, 14))))
        bright = int(rng.integers(0, 2))
        col = (225, 225, 225) if bright else (40, 40, 40)
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, int(rng.integers(2, 4)), cv2.LINE_AA)


def _add_shadow(img, rng):
    """A darkened band — stresses the adaptive brightness floor (ISSUES #2)."""
    h, w = img.shape[:2]
    y0 = int(rng.integers(int(h * 0.45), int(h * 0.8)))
    band = slice(y0, min(h, y0 + int(h * 0.18)))
    img[band] = (img[band].astype(np.int16) - 35).clip(0, 255).astype(np.uint8)


def _add_grass(img):
    """Green borders on the far left/right — bright-ish clutter off the road."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, int(h * 0.37)), (int(w * 0.1), h), (40, 110, 45), -1)
    cv2.rectangle(img, (int(w * 0.9), int(h * 0.37)), (w, h), (40, 110, 45), -1)


def _add_barrels(img, rng, n):
    """Orange traffic drums — future multi-class target (not lanes)."""
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        x, y = int(rng.integers(int(w * 0.2), int(w * 0.8))), int(rng.integers(int(h * 0.55), h - 30))
        r = int(rng.integers(14, 24))
        cv2.circle(img, (x, y), r, (35, 110, 230), -1, cv2.LINE_AA)   # BGR orange
        out.append((x, y, r))
    return out


def make_scene(name="medium", size=(480, 640), seed=0, *,
               curve=0.0, dashed=False, specks=0, cracks=0,
               shadow=False, grass=False, barrels=0,
               brightness=0) -> Scene:
    """Build one labelled scene. Difficulty = how much clutter you turn on."""
    h, w = size
    rng = np.random.default_rng(seed)
    img = _asphalt(h, w, rng)
    gt = np.zeros((h, w), np.uint8)

    if grass:
        _add_grass(img)

    # Two lane lines forming a corridor.
    left = _lane_points(h, w, int(w * 0.40), int(w * 0.18), curve=-curve)
    right = _lane_points(h, w, int(w * 0.60), int(w * 0.86), curve=curve)
    _draw_lane(img, gt, left, dashed=dashed)
    _draw_lane(img, gt, right, dashed=dashed)

    if shadow:
        _add_shadow(img, rng)
    if specks:
        _add_specks(img, rng, specks, int(h * 0.40), h - 10)
    if cracks:
        _add_cracks(img, rng, cracks, int(h * 0.42), h - 10)
    bar = _add_barrels(img, rng, barrels) if barrels else []
    if brightness:
        img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    return Scene(bgr=img, gt_lane=gt, barrels=bar, name=name,
                 meta=dict(seed=seed, curve=curve, specks=specks, cracks=cracks,
                           shadow=shadow, grass=grass, barrels=barrels))


# Preset suite spanning easy -> hard, reused by the harness.
def default_suite() -> List[Scene]:
    return [
        make_scene("easy_straight", seed=1),
        make_scene("dashed", seed=2, dashed=True),
        make_scene("curve", seed=3, curve=0.12),
        make_scene("specks_80", seed=4, specks=80),
        make_scene("cracks_30", seed=5, cracks=30),
        make_scene("shadow", seed=6, shadow=True, specks=30),
        make_scene("grass_glint", seed=7, grass=True, specks=40, brightness=15),
        make_scene("hard_all", seed=8, curve=0.10, specks=70, cracks=20,
                   shadow=True, grass=True, barrels=3),
    ]
