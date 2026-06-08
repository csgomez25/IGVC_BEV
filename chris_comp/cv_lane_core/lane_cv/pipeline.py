"""
LaneDetector — the portable, ROS-free lane-detection pipeline.

    from lane_cv import LaneDetector, LaneConfig
    det = LaneDetector(LaneConfig.from_yaml("configs/default.yaml"))
    result = det.process(bgr_frame)
    result.lane_mask    # uint8 0/255 — confirmed painted lines
    result.segments     # list[LineSegment]
    result.candidate    # uint8 0/255 — pre-shape-filter (debug)

Stages, in order:
    preprocess (blur->HSV)
      -> candidate_mask   (color cues + ROI)            segmentation.py
      -> filter_lines     (geometric speck rejection)   line_filter.py
      -> temporal confirm (N-of-window voting)          here

The detector holds the only mutable state (adaptive floor + temporal ring
buffer), so a fresh LaneDetector per camera is the unit of reuse. Nothing in
here imports ROS, OpenCV windows, or anything robot-specific — drop it into any
vehicle and feed it BGR frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import LaneConfig
from .line_filter import LineSegment, filter_lines
from .segmentation import _AdaptiveFloor, candidate_mask, preprocess


@dataclass
class LaneResult:
    lane_mask: np.ndarray                 # uint8 0/255 — final, confirmed lines
    candidate: np.ndarray                 # uint8 0/255 — color stage only
    segments: List[LineSegment] = field(default_factory=list)


class LaneDetector:
    def __init__(self, config: Optional[LaneConfig] = None) -> None:
        self.cfg = config or LaneConfig()
        self._floor = _AdaptiveFloor()
        self._history: list[np.ndarray] = []  # ring buffer of recent lane_masks

    def reset(self) -> None:
        """Clear temporal + adaptive state (e.g. on a camera/scene change)."""
        self._floor = _AdaptiveFloor()
        self._history.clear()

    # ------------------------------------------------------------------ core
    def process(self, bgr: np.ndarray) -> LaneResult:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR frame, got {getattr(bgr, 'shape', None)}")

        hsv = preprocess(bgr, self.cfg)
        candidate = candidate_mask(hsv, self._floor, self.cfg)
        clean, segments = filter_lines(candidate, self.cfg)
        lane = self._temporal_confirm(clean)
        return LaneResult(lane_mask=lane, candidate=candidate, segments=segments)

    # -------------------------------------------------------------- temporal
    def _temporal_confirm(self, clean: np.ndarray) -> np.ndarray:
        """Keep pixels lit in >= min_hits of the last `window` frames."""
        tc = self.cfg.temporal
        window = max(1, int(tc.window))
        if window == 1:
            return clean

        self._history.append((clean > 0).astype(np.uint8))
        if len(self._history) > window:
            self._history.pop(0)

        # Until the buffer fills, don't suppress — avoids a blind first second.
        if len(self._history) < window:
            return clean

        votes = np.sum(self._history, axis=0)
        confirmed = (votes >= int(tc.min_hits)).astype(np.uint8) * 255
        return confirmed

    # --------------------------------------------------------------- overlay
    def draw_overlay(self, bgr: np.ndarray, result: LaneResult) -> np.ndarray:
        """Render a human-readable debug overlay (lanes green, segments red)."""
        import cv2
        out = bgr.copy()
        out[result.lane_mask > 0] = (0, 255, 0)
        for s in result.segments:
            cv2.line(out, s.p1, s.p2, (0, 0, 255), 2)
        return out
