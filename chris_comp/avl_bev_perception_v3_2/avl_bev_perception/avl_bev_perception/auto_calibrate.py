#!/usr/bin/env python3
"""
Auto-HSV Calibration — IGVC AutoNav (v3.2)
==========================================
Inspired by Oklahoma's Twistopher (IGVC 2025 Auto-Nav 1st place, 2:20).
Their key insight: HSV thresholds tuned in the lab don't survive changing
outdoor lighting. They auto-calibrate at the start of every run.

This module captures N frames from each camera at startup, samples the
asphalt color distribution, and writes back venue-adapted HSV thresholds
to the parameter server BEFORE the perception loop begins.

Strategy:

  1. Sample the lower 30% of each frame (closest to the robot, most
     likely to be pure asphalt with no obstacles).
  2. Compute the median HSV of that region across N frames.
  3. Set white V_min   = median_V + 2 * std_V    (anything brighter than asphalt)
  4. Set orange S_min  = median_S + 3 * std_S    (anything more saturated than asphalt)
  5. Validate: run the new thresholds against a full frame and check that
     the resulting masks cover <40% of the image. If they cover more,
     calibration likely caught something other than asphalt — fall back
     to defaults.

Skip auto-calibration entirely by setting `segmentation.auto_calibrate: false`
in the config.
"""

from typing import Dict, Tuple

import cv2
import numpy as np


class AutoHsvCalibrator:
    """Captures samples from a camera and produces venue-tuned HSV thresholds."""

    # Maximum fraction of image any single class mask is allowed to cover.
    # If the calibrated mask exceeds this, the result is rejected as a false
    # positive (e.g. accidentally calibrated against a white wall).
    MAX_MASK_COVERAGE = 0.40

    # Region of the image to sample asphalt from (fraction of height from bottom).
    ASPHALT_ROI_BOTTOM_FRAC = 0.30

    def __init__(
        self,
        n_samples: int = 30,
        white_v_offset_sigma: float = 2.0,
        orange_s_offset_sigma: float = 3.0,
    ):
        self.n_samples = n_samples
        self.white_v_offset_sigma = white_v_offset_sigma
        self.orange_s_offset_sigma = orange_s_offset_sigma
        self._samples: Dict[str, list] = {}

    def add_sample(self, cam_name: str, bgr_image: np.ndarray) -> None:
        """Stash one frame from a camera. Call repeatedly during warm-up."""
        if cam_name not in self._samples:
            self._samples[cam_name] = []
        if len(self._samples[cam_name]) >= self.n_samples:
            return
        self._samples[cam_name].append(self._extract_asphalt_roi(bgr_image))

    def is_ready(self, expected_cameras: list) -> bool:
        """True when every expected camera has hit n_samples."""
        if not self._samples:
            return False
        for cam in expected_cameras:
            if len(self._samples.get(cam, [])) < self.n_samples:
                return False
        return True

    def n_collected(self, cam_name: str) -> int:
        return len(self._samples.get(cam_name, []))

    # ------------------------------------------------------------------

    def _extract_asphalt_roi(self, bgr: np.ndarray) -> np.ndarray:
        """Return the bottom strip of the image (presumed to be asphalt)."""
        h = bgr.shape[0]
        y0 = int(h * (1.0 - self.ASPHALT_ROI_BOTTOM_FRAC))
        return bgr[y0:, :, :].copy()

    def compute_thresholds(
        self,
        default_white: dict,
        default_orange: dict,
    ) -> Tuple[dict, dict, dict]:
        """
        Compute calibrated HSV thresholds from collected samples.

        Returns:
          (white_hsv, orange_hsv, debug)

        If calibration fails validation, the default thresholds are
        returned unchanged with debug['fallback'] = True.
        """
        if not self._samples:
            return default_white, default_orange, {
                'fallback': True, 'reason': 'no samples'}

        # Pool asphalt regions from ALL cameras into one population for a
        # more stable estimate, and so all three cameras share thresholds.
        all_pixels = []
        for samples in self._samples.values():
            for sample in samples:
                hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
                all_pixels.append(hsv.reshape(-1, 3))

        if not all_pixels:
            return default_white, default_orange, {
                'fallback': True, 'reason': 'empty samples'}

        pop = np.concatenate(all_pixels, axis=0)

        # Drop obvious non-asphalt pixels (very saturated, very dark, very
        # bright) — these would skew the asphalt baseline.
        s = pop[:, 1]
        v = pop[:, 2]
        keep = (s < 80) & (v > 40) & (v < 220)
        if keep.sum() < 1000:
            return default_white, default_orange, {
                'fallback': True, 'reason': 'insufficient asphalt pixels'}
        pop = pop[keep]

        median_h = float(np.median(pop[:, 0]))
        median_s = float(np.median(pop[:, 1]))
        median_v = float(np.median(pop[:, 2]))
        std_s    = float(np.std(pop[:, 1]))
        std_v    = float(np.std(pop[:, 2]))

        white = dict(default_white)
        orange = dict(default_orange)

        white['v_min'] = int(np.clip(
            median_v + self.white_v_offset_sigma * std_v, 150, 250))
        orange['s_min'] = int(np.clip(
            median_s + self.orange_s_offset_sigma * std_s, 80, 200))

        debug = {
            'fallback': False,
            'asphalt_median_hsv': (median_h, median_s, median_v),
            'asphalt_std_sv':     (std_s, std_v),
            'n_pixels_used':      int(pop.shape[0]),
            'white_v_min':        white['v_min'],
            'orange_s_min':       orange['s_min'],
        }

        # Validate on a full sample frame.
        first_cam = list(self._samples.keys())[0]
        validation_frame = self._samples[first_cam][0]
        validation_full = cv2.cvtColor(validation_frame, cv2.COLOR_BGR2HSV)

        w_mask = cv2.inRange(
            validation_full,
            np.array([white['h_min'], white['s_min'], white['v_min']], dtype=np.uint8),
            np.array([white['h_max'], white['s_max'], white['v_max']], dtype=np.uint8),
        )
        o_mask = cv2.inRange(
            validation_full,
            np.array([orange['h_min'], orange['s_min'], orange['v_min']], dtype=np.uint8),
            np.array([orange['h_max'], orange['s_max'], orange['v_max']], dtype=np.uint8),
        )
        w_cov = float(w_mask.mean()) / 255.0
        o_cov = float(o_mask.mean()) / 255.0
        debug['white_coverage']  = w_cov
        debug['orange_coverage'] = o_cov

        if w_cov > self.MAX_MASK_COVERAGE or o_cov > self.MAX_MASK_COVERAGE:
            debug['fallback'] = True
            debug['reason'] = (
                f'mask coverage too high (white={w_cov:.2f}, '
                f'orange={o_cov:.2f}) — likely calibrated against '
                f'non-asphalt surface. Using defaults.'
            )
            return default_white, default_orange, debug

        return white, orange, debug
