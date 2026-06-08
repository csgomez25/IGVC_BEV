"""
providers.py — the SegmentationProvider seam: interchangeable detection backends.

One contract, swappable middle. Every backend implements:

    provider.infer(bgr, depth=None) -> SegResult(class_mask, confidence)

both uint8 HxW; `class_mask` uses lane_cv.classes IDs, `confidence` is 0-255.
Because they all share that contract, the BEV layer (rig.py) and everything
downstream never know — or care — which backend ran. Pick the backend per camera
in vehicle.yaml; mix them freely (classical lanes on one camera, YOLO objects on
another, both fused into the same BEV grid).

Backends:
  ClassicalProvider — wraps the ROS-free LaneDetector. Zero extra deps, the
                      portable baseline that works on a bare laptop.
  YoloProvider      — ultralytics YOLO (nano by default); detection boxes painted
                      into the class mask. **Lazy import** — `import lane_cv`
                      never pulls in ultralytics; the dep is only touched when a
                      YoloProvider is actually constructed.
  RoadSegProvider   — placeholder for a Cityscapes-pretrained road/sidewalk
                      segmenter. Not implemented — pick a model first (TODO §6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .classes import COCO_TO_DRIVING, NAME_TO_ID
from .config import LaneConfig
from .pipeline import LaneDetector

_LANE_ID = NAME_TO_ID["lane"]

__all__ = [
    "SegResult", "SegmentationProvider", "ClassicalProvider", "YoloProvider",
    "RoadSegProvider", "build_provider", "rasterize_boxes",
]


@dataclass
class SegResult:
    class_mask: np.ndarray   # uint8 HxW, lane_cv.classes IDs
    confidence: np.ndarray   # uint8 HxW, 0-255


class SegmentationProvider(ABC):
    @abstractmethod
    def infer(self, bgr: np.ndarray, depth: Optional[np.ndarray] = None) -> SegResult:
        ...

    def reset(self) -> None:
        """Clear any per-stream state (scene change). Default: nothing."""
        return None


class ClassicalProvider(SegmentationProvider):
    """The current classical pipeline, adapted to the provider contract."""

    def __init__(self, config: Optional[LaneConfig] = None) -> None:
        self._det = LaneDetector(config or LaneConfig())

    @classmethod
    def from_profile(cls, path: Optional[str]) -> "ClassicalProvider":
        return cls(LaneConfig.from_yaml(path) if path else LaneConfig())

    def infer(self, bgr, depth=None) -> SegResult:
        res = self._det.process(bgr)
        lit = res.lane_mask > 0
        class_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        class_mask[lit] = _LANE_ID
        return SegResult(class_mask=class_mask, confidence=lit.astype(np.uint8) * 255)

    def reset(self) -> None:
        self._det.reset()


def rasterize_boxes(shape, boxes, class_ids, confidences):
    """
    Paint filled detection boxes into (class_mask, confidence) rasters — the
    bridge from box detectors (YOLO) to the per-pixel provider contract so the
    result projects through the same BEV layer as a segmentation mask.

    boxes : iterable of (x1, y1, x2, y2) in pixels. Higher-confidence boxes are
    painted last so they win on overlap.
    """
    h, w = shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    conf_mask = np.zeros((h, w), dtype=np.uint8)
    confidences = np.asarray(confidences, dtype=np.float32)
    for i in np.argsort(confidences):              # ascending -> high conf on top
        x1, y1, x2, y2 = (int(round(v)) for v in boxes[i])
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        class_mask[y1:y2, x1:x2] = int(class_ids[i])
        conf_mask[y1:y2, x1:x2] = int(round(float(confidences[i]) * 255))
    return class_mask, conf_mask


class YoloProvider(SegmentationProvider):
    """
    ultralytics YOLO backend (nano by default for the latency budget). Maps the
    driving-relevant COCO classes to our IDs and rasterizes boxes into the mask.
    """

    def __init__(self, model: str = "yolo11n.pt", conf: float = 0.35,
                 device: Optional[str] = None,
                 keep: Optional[Sequence[str]] = None) -> None:
        try:
            from ultralytics import YOLO   # lazy: optional dependency
        except Exception as exc:           # noqa: BLE001
            raise ImportError(
                "YoloProvider needs `pip install ultralytics`. It is an OPTIONAL "
                "backend — the classical provider has no such dependency, so the "
                "core library still imports and runs without it."
            ) from exc
        self._model = YOLO(model)
        self._conf = float(conf)
        self._device = device
        self._coco_map = {
            coco_id: NAME_TO_ID[name]
            for coco_id, name in COCO_TO_DRIVING.items()
            if keep is None or name in keep
        }

    def infer(self, bgr, depth=None) -> SegResult:
        res = self._model.predict(bgr, conf=self._conf, device=self._device,
                                  verbose=False)[0]
        boxes, ids, confs = [], [], []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for b, c, p in zip(xyxy, cls, cf):
                if c in self._coco_map:
                    boxes.append(b)
                    ids.append(self._coco_map[c])
                    confs.append(p)
        if not boxes:
            empty = np.zeros(bgr.shape[:2], dtype=np.uint8)
            return SegResult(class_mask=empty, confidence=empty.copy())
        class_mask, conf = rasterize_boxes(bgr.shape, boxes, ids, confs)
        return SegResult(class_mask=class_mask, confidence=conf)


class RoadSegProvider(SegmentationProvider):
    """
    Placeholder for a road/sidewalk semantic segmenter (Cityscapes classes:
    road, sidewalk, ...). NOT implemented — choose a model first: a small
    Cityscapes-pretrained net (e.g. SegFormer-B0, BiSeNet, or DeepLab-mobile),
    ONNX-exported for the Jetson, behind this same contract. See TODO §6.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "RoadSegProvider is a stub. Pick a Cityscapes-pretrained model "
            "(SegFormer-B0 / BiSeNet ONNX), then implement infer() to emit the "
            "'road'/'sidewalk' class IDs from lane_cv.classes. See TODO Phase 6."
        )

    def infer(self, bgr, depth=None) -> SegResult:   # pragma: no cover
        raise NotImplementedError


def build_provider(cfg, resolve=None) -> SegmentationProvider:
    """
    Construct a provider from a SegmentationCfg (rig_config.py). `resolve` is an
    optional path resolver for the classical profile (relative to the rig YAML).
    """
    backend = (getattr(cfg, "backend", None) or "classical").lower()
    if backend == "classical":
        profile = cfg.profile
        if resolve is not None:
            profile = resolve(profile)
        return ClassicalProvider.from_profile(profile)
    if backend == "yolo":
        return YoloProvider(model=cfg.model or "yolo11n.pt", conf=cfg.conf,
                            device=cfg.device, keep=cfg.keep)
    if backend == "road_seg":
        return RoadSegProvider(model=cfg.model, device=cfg.device)
    raise ValueError(f"unknown segmentation backend {backend!r}")
