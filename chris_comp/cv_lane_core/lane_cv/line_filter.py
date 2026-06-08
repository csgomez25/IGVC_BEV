"""
Geometric line filter — the fix for "road specks show up as lane lines".

A color threshold can't tell white paint from a bright tar crack, a pebble, or
a sun-glint: they're all "bright + low-saturation". What separates them is
SHAPE. A painted lane line is a long, thin, fairly straight stroke. A speck or
a crack-cluster is short and/or stubby.

So after color thresholding we examine each connected component and keep it
ONLY if it is line-shaped:

  area          in [min_area, max_area]
  elongation    = long_side / short_side  >= min_elongation
  fill          = area / minAreaRect_area  <= max_fill   (a filled square != line)
  orientation   (optional) long axis within +/- tol of a target angle
  hough         (optional) contains a straight run >= hough_min_len_px

Everything that fails is dropped — which is exactly what removes the specks
without ever touching the genuine lines. We intentionally do NOT run an
aggressive MORPH_CLOSE before this step: closing first would weld neighboring
specks into a line-shaped streak and defeat the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import LaneConfig


@dataclass
class LineSegment:
    """One accepted line-like component, described as a centered segment."""
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    length_px: float
    angle_deg: float        # 0 = horizontal, 90 = vertical (image space)
    area_px: int
    elongation: float
    fill: float


def _angle_in_gate(angle_deg: float, cfg: LaneConfig) -> bool:
    lf = cfg.line
    if not lf.orientation_gate:
        return True
    # Compare modulo 180 so a line and its flip are equivalent.
    diff = abs((angle_deg - lf.angle_center_deg + 90.0) % 180.0 - 90.0)
    return diff <= lf.angle_tol_deg


def _has_straight_run(component_mask: np.ndarray, cfg: LaneConfig) -> bool:
    """Optional HoughLinesP confirmation that a straight segment exists."""
    min_len = cfg.line.hough_min_len_px
    if min_len <= 0:
        return True
    lines = cv2.HoughLinesP(
        component_mask, rho=1, theta=np.pi / 180.0, threshold=20,
        minLineLength=float(min_len), maxLineGap=float(max(3, min_len // 4)),
    )
    return lines is not None and len(lines) > 0


def _segment_from_rect(rect, area: int, elong: float, fill: float) -> LineSegment:
    (cx, cy), (rw, rh), ang = rect
    # Long axis direction. cv2's angle convention flips with which side is
    # 'width', so normalize to the longer side.
    if rw >= rh:
        length, theta = rw, ang
    else:
        length, theta = rh, ang + 90.0
    theta_rad = np.deg2rad(theta)
    dx, dy = np.cos(theta_rad) * length / 2.0, np.sin(theta_rad) * length / 2.0
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    angle_deg = float(np.rad2deg(np.arctan2(dy, dx)) % 180.0)
    return LineSegment(
        p1=p1, p2=p2, length_px=float(length), angle_deg=angle_deg,
        area_px=int(area), elongation=float(elong), fill=float(fill),
    )


def _link_dashes(
    pending: List[Tuple[int, "LineSegment"]], cfg: LaneConfig
) -> Tuple[List[int], List["LineSegment"]]:
    """
    Recover dashed lines. `pending` is the line-ISH-but-too-short components
    (label index + its segment). Two dashes belong to the same line when they
    are (a) close in heading, (b) within `dash_link_max_gap_px`, and (c) the
    vector joining their centroids is itself aligned with that heading — i.e.
    they sit END-TO-END, not side by side (which would wrongly weld two parallel
    lane lines). Chains of >= `dash_link_min_segments` are accepted. Random
    specks/cracks almost never form such collinear chains, so clutter stays out.
    """
    lf = cfg.line
    n = len(pending)
    if n < lf.dash_link_min_segments:
        return [], []

    cents = [((s.p1[0] + s.p2[0]) / 2.0, (s.p1[1] + s.p2[1]) / 2.0)
             for _, s in pending]
    angs = [s.angle_deg for _, s in pending]
    gap, atol = lf.dash_link_max_gap_px, lf.dash_link_angle_tol_deg

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs((angs[i] - angs[j] + 90.0) % 180.0 - 90.0) > atol:
                continue                                   # different heading
            dx = cents[j][0] - cents[i][0]
            dy = cents[j][1] - cents[i][1]
            if (dx * dx + dy * dy) ** 0.5 > gap:
                continue                                   # too far apart
            conn = float(np.rad2deg(np.arctan2(dy, dx))) % 180.0
            if abs((conn - angs[i] + 90.0) % 180.0 - 90.0) > atol:
                continue                                   # parallel, not collinear
            adj[i].append(j)
            adj[j].append(i)

    seen = [False] * n
    keep_labels: List[int] = []
    keep_segs: List["LineSegment"] = []
    for s0 in range(n):
        if seen[s0]:
            continue
        comp, stack = [], [s0]
        seen[s0] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) >= lf.dash_link_min_segments:
            for idx in comp:
                keep_labels.append(pending[idx][0])
                keep_segs.append(pending[idx][1])
    return keep_labels, keep_segs


def filter_lines(
    candidate: np.ndarray, cfg: LaneConfig
) -> Tuple[np.ndarray, List[LineSegment]]:
    """
    Input  : binary candidate mask (uint8, 0/255) from segmentation.
    Output : (clean_mask, segments) — clean_mask keeps only line-shaped blobs.
    """
    lf = cfg.line

    # Gentle open to peel off single-pixel noise WITHOUT merging blobs.
    if cfg.open_ksize and cfg.open_ksize >= 2:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8)
    clean = np.zeros_like(candidate)
    segments: List[LineSegment] = []
    pending: List[Tuple[int, LineSegment]] = []   # line-ish but too short = dashes

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < lf.min_area or area > lf.max_area:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        short, long_ = (min(rw, rh), max(rw, rh))
        if short < 1e-3:
            continue

        elong = long_ / short
        fill = area / max(1.0, rw * rh)
        # max_fill rejects solid bright PATCHES (a filled square is not a line).
        # It must only judge near-round blobs: a SOLID line and a SOLID dash are
        # both line-like with high fill, so applying it to elongated shapes would
        # throw away real (merged) lines and dashes. Elongation is the real
        # patch/line discriminator; chaining handles dashes vs lone blobs.
        if (lf.max_fill < 1.0 and fill > lf.max_fill
                and elong < lf.dash_min_elongation):
            continue

        seg = _segment_from_rect(rect, area, elong, fill)
        if not _angle_in_gate(seg.angle_deg, cfg):
            continue

        if elong < lf.min_elongation:
            # Too short to be a line on its own, but long enough to be a dash
            # (not a round speck) -> hold it for collinear linking below.
            if lf.dash_link and elong >= lf.dash_min_elongation:
                pending.append((i, seg))
            continue
        if not _has_straight_run(comp, cfg):
            continue

        clean[labels == i] = 255
        segments.append(seg)

    if lf.dash_link and pending:
        dash_labels, dash_segs = _link_dashes(pending, cfg)
        for li in dash_labels:
            clean[labels == li] = 255
        segments.extend(dash_segs)

    return clean, segments
