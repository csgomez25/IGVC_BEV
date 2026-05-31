#!/usr/bin/env python3
"""
Segmentation Engine — IGVC AutoNav (avl_bev_perception v3.1)
============================================================
Hybrid two-tier semantic segmentation tuned for the IGVC AutoNav course.

Two tiers run in series:

  Tier 1 — Classical CV (always on, ~2-4 ms / frame total)
    1a. HSV thresholding for white lane lines (CLASS_LANE_LINE)
    1b. HSV thresholding for orange barrels   (CLASS_BARREL)
    1c. Grayscale + Otsu pass for bright unclassified objects, mapped to
        CLASS_POTHOLE. Inspired by Monash MCAV's unified lane+pothole
        detection in their IGVC 2025 design report — catches IGVC's painted
        flat pothole circles and any other "bright stuff on dark asphalt"
        the HSV pass missed, with no learned model required.

  Tier 2 — Optional ONNX model (off by default)
    For higher-quality person, pothole, and drivable-area classification
    when a trained model is available. Tier 1 overwrites Tier 2 on
    overlap because Tier 1 is more reliable for the colors it covers.
    Set `model_path` in config to enable.

Output mask class IDs (uint8, single channel):
  0   = background       (a.k.a. "free")               — none
  1   = lane line        (white, painted)              — Tier 1a
  2   = barrel           (orange traffic drum)         — Tier 1b
  3   = pothole          (painted circle / bright)     — Tier 1c or Tier 2
  4   = person           (pedestrian / judge)          — Tier 2
  5   = drivable area    (asphalt)                     — Tier 2
  255 = unknown          (reserved for LabelInfo)      — none

IDs 0–3 + 255 match Parsa's class_map.yaml so this mask can be published
directly to the kiwicampus semantic_segmentation_layer contract.

The BEV node renders these into colored overlays AND derives two binary
masks for the planner:
  drivable_mask  = (class == CLASS_DRIVABLE)   OR  asphalt-by-default if no Tier 2
  obstacle_mask  = (class IN OBSTACLE_CLASSES) = {lane, barrel, pothole, person}
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# Class IDs.
#
# 0–3 mirror references/parsa_igvc/src/avros_perception/config/class_map.yaml
# so this package can publish directly to Parsa's kiwicampus
# semantic_segmentation_layer contract without an ID-remap step.
# 4–5 are local extensions not (yet) in Parsa's class_map. They're harmless
# to Parsa's costmap (it just doesn't have class_types entries for them).
# 255 mirrors his "unknown" sentinel for future LabelInfo publication.
#
# Keep in sync with the fallback block in bev_perception_node.py and with
# _get_seg_colors() in the same file.
CLASS_BACKGROUND = 0    # ↔ "free"          in Parsa's class_map
CLASS_LANE_LINE  = 1    # ↔ "lane_white"
CLASS_BARREL     = 2    # ↔ "barrel_orange"
CLASS_POTHOLE    = 3    # ↔ "pothole"        (was 4 pre-v3.2.2)
CLASS_PERSON     = 4    # local extension    (was 3 pre-v3.2.2)
CLASS_DRIVABLE   = 5    # local extension
CLASS_UNKNOWN    = 255  # ↔ "unknown"        — reserved for LabelInfo

OBSTACLE_CLASSES = (CLASS_LANE_LINE, CLASS_BARREL, CLASS_POTHOLE, CLASS_PERSON)


class SegmentationEngine:
    """Hybrid HSV + optional ONNX segmentation for IGVC."""

    # ---- HSV thresholds (OpenCV: H 0-179, S 0-255, V 0-255) -----------
    # Tuned for outdoor daylight on asphalt. Re-tune at the venue with
    # the included tools/calibrate_hsv.py if lighting is unusual.
    DEFAULT_WHITE_HSV = {
        'h_min':   0, 'h_max': 179,
        's_min':   0, 's_max':  60,   # low saturation = white/gray
        'v_min': 200, 'v_max': 255,   # high value = bright; matches Parsa's
                                      # 2026-05-13 white-on-grass retune
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
        # Tier 1b — bright-object (Otsu) pass for potholes / unclassified bright stuff.
        # Inspired by Monash MCAV's unified lane+pothole detection in their IGVC 2025
        # report: grayscale + Otsu threshold catches "bright stuff on dark asphalt"
        # extremely cheaply (~0.5 ms / frame). We run it AFTER the HSV pass and only
        # paint cells the HSV pass left as background, so it never overwrites a
        # confident lane-line or barrel detection.
        bright_pass_enabled: bool = True,
        bright_min_area_px: int = 40,
        bright_max_area_px: int = 8000,    # discard huge bright regions (sky, bright walls)
        bright_blur_px: int = 5,
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

        self.bright_pass_enabled = bright_pass_enabled
        self.bright_min_area_px  = bright_min_area_px
        self.bright_max_area_px  = bright_max_area_px
        # Force odd kernel size for GaussianBlur.
        self.bright_blur_px = bright_blur_px if bright_blur_px % 2 == 1 else bright_blur_px + 1

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

    def set_hsv_thresholds(self, white_hsv: dict, orange_hsv: dict) -> None:
        """
        v3.2: Update HSV thresholds at runtime (used by auto-calibration).

        Replaces the full dict atomically — reads during this method are
        fine because HSV inference makes local copies of the min/max bounds
        each call.
        """
        self.white_hsv  = dict(white_hsv)
        self.orange_hsv = dict(orange_hsv)

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

        # ---- Bright-object (Otsu) pass for potholes / unclassified bright stuff ----
        # Only paint cells the HSV pass left as background, so we never overwrite a
        # confident lane-line or barrel detection. Sized between bright_min/max so
        # we drop both speckle (paint chips) and huge regions (overexposed sky).
        if self.bright_pass_enabled:
            self._infer_tier1_bright(bgr, mask)

    def _infer_tier1_bright(self, bgr: np.ndarray, mask: np.ndarray) -> None:
        """
        Grayscale + Otsu threshold pass. Inspired by Monash MCAV's IGVC 2025
        unified lane+pothole detection. Catches anything visibly brighter than
        the asphalt that wasn't already labeled by HSV.

        Class assigned: CLASS_POTHOLE. The planner treats potholes as obstacles
        in /bev/obstacle_mask, so even false positives just become "go around"
        decisions, which is the safe failure mode.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.bright_blur_px, self.bright_blur_px), 0)

        # Otsu picks an adaptive threshold per frame — handles lighting drift
        # between cameras and across the day.
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Mask out anything the HSV pass already classified as a real class so
        # we don't double-label or overwrite.
        already_classified = (mask != CLASS_BACKGROUND)
        bright[already_classified] = 0

        # Morph close to consolidate broken pothole rings.
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, self.morph_kernel)

        # Component filter with both lower and upper area bounds.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.bright_min_area_px <= area <= self.bright_max_area_px:
                mask[labels == i] = CLASS_POTHOLE

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
