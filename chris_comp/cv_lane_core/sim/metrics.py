"""
Metrics for scoring a detection against a scene's ground truth.

Two numbers carry the whole story for the lane-vs-specks problem:

  lane_recall      — of the TRUE lane pixels, what fraction did we detect
                     (within a few px tolerance). High = we kept the lanes.
  false_line_count — how many detected components do NOT overlap a real lane,
                     i.e. phantom lines manufactured from specks/cracks.
                     Low (ideally 0) = clutter was rejected.

A good detector maximizes recall while keeping false_line_count near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _kernel(px: int):
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


@dataclass
class Score:
    lane_recall: float
    false_line_count: int
    detected_components: int
    gt_px: int
    det_px: int

    def ok(self, min_recall: float, max_false_lines: int) -> bool:
        return self.lane_recall >= min_recall and self.false_line_count <= max_false_lines


def score(gt_lane: np.ndarray, det_mask: np.ndarray,
          recall_tol_px: int = 6, fp_tol_px: int = 10,
          fp_min_overlap: float = 0.2) -> Score:
    gt_b = gt_lane > 0
    det_b = (det_mask > 0).astype(np.uint8)
    gt_px = int(gt_b.sum())
    det_px = int(det_b.sum())

    # Recall: dilate detection by tolerance, see how much GT it covers.
    if gt_px == 0:
        recall = 1.0
    else:
        det_d = cv2.dilate(det_b, _kernel(recall_tol_px)) > 0
        recall = float((gt_b & det_d).sum()) / float(gt_px)

    # False lines: detected components with little overlap with (dilated) GT.
    gt_d = cv2.dilate(gt_b.astype(np.uint8), _kernel(fp_tol_px)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(det_b, connectivity=8)
    fp = 0
    for i in range(1, n):
        comp = labels == i
        area = comp.sum()
        if area == 0:
            continue
        overlap = float((comp & gt_d).sum()) / float(area)
        if overlap < fp_min_overlap:
            fp += 1

    return Score(lane_recall=recall, false_line_count=fp,
                 detected_components=max(0, n - 1), gt_px=gt_px, det_px=det_px)
