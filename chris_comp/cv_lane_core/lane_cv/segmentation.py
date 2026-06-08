"""
Candidate-mask generation — the COLOR stage of lane detection.

This stage answers "which pixels *could* be paint?" using only color/brightness
cues. It is deliberately permissive: it will happily light up bright road
specks, tar-filled cracks, and sun-glints. Rejecting those is NOT done here —
it is done geometrically in line_filter.py. Keeping the two concerns separate
is the whole point: color is venue-tunable, shape is physics.

Cues implemented (all from prior IGVC-winning classical pipelines):
  * adaptive brightness floor  (iscumd white_line_detection)
  * low-saturation white gate   (Sooner 2023/2024 per-class HSV)
  * asphalt-invert gate         (Sooner 2025 'threshold drivable, invert')
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import LaneConfig


class _AdaptiveFloor:
    """Stateful mean+k*sigma V-floor, refreshed every `period` frames."""

    def __init__(self) -> None:
        self._tick = 0
        self._floor: float | None = None

    def value(self, v_channel: np.ndarray, cfg: LaneConfig) -> float:
        ac = cfg.adaptive
        if not ac.enabled:
            return float(cfg.white.v_min)
        if self._floor is None or self._tick == 0:
            h = v_channel.shape[0]
            top = int(max(0.0, min(1.0, ac.band[0])) * h)
            bot = int(max(0.0, min(1.0, ac.band[1])) * h)
            band = v_channel[top:bot, :]
            if band.size == 0:
                band = v_channel
            self._floor = float(band.mean() + ac.k * band.std())
        self._tick = (self._tick + 1) % max(ac.period, 1)
        # Never drop below the static floor (the venue's hard 'paint is at least
        # this bright' prior); never exceed 255 (a bright scene can push
        # mean+k*sigma past the uint8 range, which both overflows the inRange
        # bound and means "match nothing" — the daylight failure Parsa hit by
        # disabling adaptive_k. Clamp so it degrades to a near-empty mask, not a
        # crash; a contrast-relative gate is the real fix — see TODO Phase 1).
        return float(min(255.0, max(self._floor, float(cfg.white.v_min))))


def preprocess(bgr: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """Box-blur a few times to kill pixel-scale speckle, then return HSV."""
    k = max(1, int(cfg.blur_ksize))
    out = bgr
    for _ in range(max(0, int(cfg.blur_iters))):
        out = cv2.blur(out, (k, k))
    return cv2.cvtColor(out, cv2.COLOR_BGR2HSV)


def roi_mask(h: int, w: int, cfg: LaneConfig) -> np.ndarray | None:
    """Build a uint8 keep-mask (255 inside roi_poly) or None if no polygon."""
    poly = cfg.roi_poly
    if not poly:
        return None
    pts = np.array(
        [[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in poly],
        dtype=np.int32,
    )
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


def white_gate(hsv: np.ndarray, v_floor: float, cfg: LaneConfig) -> np.ndarray:
    wc = cfg.white
    lo = np.array([wc.h_min, wc.s_min, int(v_floor)], dtype=np.uint8)
    hi = np.array([wc.h_max, wc.s_max, wc.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def tophat_gate(hsv: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """
    Contrast-relative white gate (combine: tophat).

    A white top-hat = V - opening(V) responds to pixels BRIGHTER THAN THEIR LOCAL
    NEIGHBORHOOD, so a painted line is detected by its contrast against the
    surface right next to it rather than an absolute V floor. This survives both
    shadow (real line's absolute V is low but it is still locally bright) and
    bright concrete (background is bright but flat, so its top-hat is ~0) — the
    two cases the absolute white/asphalt gates erase. Saturation and a soft
    absolute floor still gate out colored / very-dark clutter.
    """
    cc = cfg.contrast
    v = hsv[:, :, 2]
    k = max(3, int(cc.tophat_ksize) | 1)   # force odd, >= 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)
    mask = (th >= int(cc.min_contrast)).astype(np.uint8) * 255
    mask[hsv[:, :, 1] > cfg.white.s_max] = 0   # colored -> not white paint
    mask[v < int(cc.v_min)] = 0                # faint dark-region texture -> drop
    return mask


def not_asphalt_gate(hsv: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """Threshold the drivable asphalt, invert -> 255 where NOT asphalt."""
    ac = cfg.asphalt
    lo = np.array([ac.h_min, ac.s_min, ac.v_min], dtype=np.uint8)
    hi = np.array([ac.h_max, ac.s_max, ac.v_max], dtype=np.uint8)
    asphalt = cv2.inRange(hsv, lo, hi)
    return cv2.bitwise_not(asphalt)


def candidate_mask(
    hsv: np.ndarray, floor: _AdaptiveFloor, cfg: LaneConfig
) -> np.ndarray:
    """Combine the configured color cues into one binary candidate mask."""
    if cfg.combine == "tophat":
        mask = tophat_gate(hsv, cfg)
    elif cfg.combine == "asphalt_inv":
        mask = not_asphalt_gate(hsv, cfg)
    else:
        v_floor = floor.value(hsv[:, :, 2], cfg)
        mask = white_gate(hsv, v_floor, cfg)
        if cfg.combine == "white_gated" and cfg.asphalt.enabled:
            mask = cv2.bitwise_and(mask, not_asphalt_gate(hsv, cfg))

    h, w = hsv.shape[:2]
    rm = roi_mask(h, w, cfg)
    if rm is not None:
        mask = cv2.bitwise_and(mask, rm)
    return mask
