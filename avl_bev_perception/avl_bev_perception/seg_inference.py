#!/usr/bin/env python3
"""
Segmentation Engine — IGVC AutoNav (avl_bev_perception v3)
==========================================================
Hybrid two-tier semantic segmentation tuned for the IGVC AutoNav course.

The course is an asphalt parking lot with painted white lane lines, orange
construction barrels (50-gallon striped drums), occasional pedestrians /
judges, and "potholes" (flat painted circles, ~30-60 cm). One section of
the course has no lane lines and is navigated GPS-waypoint-to-waypoint.

Two tiers run in series:

  Tier 1 — Classical CV (always on, ~1-3 ms / frame)
    HSV color thresholding for white lane lines and orange barrels. These
    two classes have an unmistakable color signature on an IGVC course and
    are detected far faster and more reliably with a threshold than with
    a learned model.

  Tier 2 — Optional ONNX model (off by default)
    For people, potholes, and drivable-area classification. Triggered only
    when `model_path` is set in config. Output classes are merged into the
    Tier 1 mask (Tier 1 wins on overlap because it's higher confidence).

Output mask class IDs (uint8, single channel):
  0 = background
  1 = lane line          (white, painted)       — Tier 1
  2 = barrel             (orange traffic drum)  — Tier 1
  3 = person             (pedestrian / judge)   — Tier 2
  4 = pothole            (painted flat circle)  — Tier 2
  5 = drivable area      (asphalt)              — Tier 2

The BEV node renders these into colored overlays AND derives two binary
masks for the planner:
  drivable_mask  = (class == 5)   OR  asphalt-by-default if no Tier 2
  obstacle_mask  = (class IN {1, 2, 3, 4})
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# Class IDs — keep in sync with BevPerceptionNode._get_seg_colors()
CLASS_BACKGROUND = 0
CLASS_LANE_LINE  = 1
CLASS_BARREL     = 2
CLASS_PERSON     = 3
CLASS_POTHOLE    = 4
CLASS_DRIVABLE   = 5

OBSTACLE_CLASSES = (CLASS_LANE_LINE, CLASS_BARREL, CLASS_PERSON, CLASS_POTHOLE)


class SegmentationEngine:
    """Hybrid HSV + optional ONNX segmentation for IGVC."""

    # ---- HSV thresholds (OpenCV: H 0-179, S 0-255, V 0-255) -----------
    # Tuned for outdoor daylight on asphalt. Re-tune at the venue with
    # the included tools/calibrate_hsv.py if lighting is unusual.
    DEFAULT_WHITE_HSV = {
        'h_min':   0, 'h_max': 179,
        's_min':   0, 's_max':  60,   # low saturation = white/gray
        'v_min': 180, 'v_max': 255,   # high value     = bright
    }
    DEFAULT_ORANGE_HSV = {
        # Orange wraps near H=0; we use a single band that covers the
        # safety-orange of IGVC barrels (H ~5-20 in OpenCV's 0-179 scale).
        'h_min':   5, 'h_max':  20,
        's_min': 130, 's_max': 255,
        'v_min': 100, 'v_max': 255,
    }

    def __init__(
        self,
        # Tier 1 (always on)
        white_hsv: Optional[dict] = None,
        orange_hsv: Optional[dict] = None,
        min_line_area_px: int = 30,
        min_barrel_area_px: int = 200,
        morph_kernel_px: int = 3,
        # Tier 2 (optional)
        model_path: str = '',
        device: str = 'cuda',
        input_size: Tuple[int, int] = (512, 384),
    ):
        self.white_hsv = white_hsv or dict(self.DEFAULT_WHITE_HSV)
        self.orange_hsv = orange_hsv or dict(self.DEFAULT_ORANGE_HSV)
        self.min_line_area_px = min_line_area_px
        self.min_barrel_area_px = min_barrel_area_px
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_px, morph_kernel_px)
        )

        # Tier 2
        self.device = device
        self.input_size = input_size
        self.tier2_model = None
        self._tier2_input_name = None
        self._tier2_output_names = None
        if model_path and os.path.isfile(model_path):
            try:
                self._load_tier2(model_path)
            except Exception as e:
                print(f'[SegEngine] Tier 2 load failed ({e}). '
                      f'Running Tier 1 only.')
                self.tier2_model = None
        else:
            if model_path:
                print(f'[SegEngine] Tier 2 model_path not found: {model_path}. '
                      f'Running Tier 1 only.')

    # ----------------------------------------------------------------- API

    def infer(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run full pipeline. Returns (H, W) uint8 class mask."""
        h, w = bgr_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Tier 2 first so Tier 1 can overwrite (Tier 1 = higher confidence).
        if self.tier2_model is not None:
            try:
                mask = self._infer_tier2(bgr_image)
            except Exception as e:
                # Don't crash the BEV loop — just fall back to Tier 1 only.
                print(f'[SegEngine] Tier 2 inference error: {e}. '
                      f'Falling back to Tier 1 for this frame.')
                mask = np.zeros((h, w), dtype=np.uint8)

        self._infer_tier1(bgr_image, mask)
        return mask

    # ------------------------------------------------------- Tier 1 (HSV)

    def _infer_tier1(self, bgr: np.ndarray, mask: np.ndarray) -> None:
        """Threshold lane lines (white) and barrels (orange). Modifies mask."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ---- White lane lines ----
        w_mask = cv2.inRange(
            hsv,
            np.array([self.white_hsv['h_min'], self.white_hsv['s_min'],
                      self.white_hsv['v_min']], dtype=np.uint8),
            np.array([self.white_hsv['h_max'], self.white_hsv['s_max'],
                      self.white_hsv['v_max']], dtype=np.uint8),
        )
        # Open then close to suppress speckle but keep thin lines connected.
        w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_OPEN,  self.morph_kernel)
        w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        w_mask = self._filter_by_area(w_mask, self.min_line_area_px)
        mask[w_mask > 0] = CLASS_LANE_LINE

        # ---- Orange barrels ----
        o_mask = cv2.inRange(
            hsv,
            np.array([self.orange_hsv['h_min'], self.orange_hsv['s_min'],
                      self.orange_hsv['v_min']], dtype=np.uint8),
            np.array([self.orange_hsv['h_max'], self.orange_hsv['s_max'],
                      self.orange_hsv['v_max']], dtype=np.uint8),
        )
        o_mask = cv2.morphologyEx(o_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        o_mask = self._filter_by_area(o_mask, self.min_barrel_area_px)
        mask[o_mask > 0] = CLASS_BARREL

    @staticmethod
    def _filter_by_area(binary: np.ndarray, min_area: int) -> np.ndarray:
        """Drop connected components smaller than min_area pixels."""
        if min_area <= 1:
            return binary
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    # ------------------------------------------------------- Tier 2 (ONNX)

    def _load_tier2(self, path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                'onnxruntime not installed. Run: '
                'pip install onnxruntime-gpu --break-system-packages'
            ) from e

        available = ort.get_available_providers()
        if self.device == 'cuda' and 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.tier2_model = ort.InferenceSession(path, providers=providers)
        self._tier2_input_name = self.tier2_model.get_inputs()[0].name
        self._tier2_output_names = [o.name for o in self.tier2_model.get_outputs()]

        # If the ONNX graph declares a fixed input size, use it.
        in_shape = self.tier2_model.get_inputs()[0].shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int):
            self.input_size = (in_shape[3], in_shape[2])

        prov = providers[0].replace('ExecutionProvider', '')
        print(f'[SegEngine] Tier 2 loaded: {os.path.basename(path)} ({prov})')
        print(f'  Input: {self.input_size[0]}x{self.input_size[1]}  '
              f'Outputs: {self._tier2_output_names}')

    def _infer_tier2(self, bgr: np.ndarray) -> np.ndarray:
        """
        Custom ONNX model expected to output (1, C, H, W) class logits where
        C corresponds to a class set including {person, pothole, drivable}.

        Class indices in the model output are assumed to map directly to
        our CLASS_* IDs above. If your model uses different indices, remap
        them here.
        """
        orig_h, orig_w = bgr.shape[:2]
        iw, ih = self.input_size

        img, _, (pad_w, pad_h) = self._letterbox(bgr, (ih, iw))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)

        outputs = self.tier2_model.run(
            self._tier2_output_names, {self._tier2_input_name: img}
        )
        logits = np.squeeze(outputs[0])
        if logits.ndim == 3:
            small = logits.argmax(0).astype(np.uint8)
        elif logits.ndim == 2:
            small = logits.astype(np.uint8)
        else:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

        return self._unpad_and_resize(small, pad_w, pad_h, iw, ih, orig_w, orig_h)

    # ------------------------------------------------------------ Helpers

    @staticmethod
    def _letterbox(img, new_shape=(384, 512), color=(114, 114, 114)):
        """Resize keeping aspect ratio, padding the rest."""
        h, w = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    @staticmethod
    def _unpad_and_resize(mask, pad_w, pad_h, iw, ih, orig_w, orig_h):
        crop = mask[int(pad_h):int(ih - pad_h), int(pad_w):int(iw - pad_w)]
        if crop.size == 0:
            crop = mask
        return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
