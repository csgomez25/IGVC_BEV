"""
lane_cv — portable, ROS-free lane / line detection for outdoor robots.

Public API:
    LaneDetector, LaneResult    (pipeline.py)   — single-camera 2D detection
    LaneConfig + sub-configs     (config.py)
    LineSegment                  (line_filter.py)
    Rig, BevResult               (rig.py)        — N-camera metric BEV fusion
    RigConfig                    (rig_config.py) — the one file a vehicle edits
    GroundGrid, *Projector       (bev.py)        — the BEV building blocks

The package depends only on numpy + opencv (PyYAML optional, for config I/O).
See ../README.md for the porting guide and the failure-mode rationale.
"""

from .bev import (
    CLASS_BACKGROUND,
    CLASS_LANE,
    CLASS_OBSTACLE,
    DepthProjector,
    GroundGrid,
    HomographyProjector,
    fuse_grids,
)
from .config import (
    AdaptiveCfg,
    AsphaltCfg,
    LaneConfig,
    LineFilterCfg,
    TemporalCfg,
    WhiteCfg,
)
from .classes import DRIVING_CLASSES, ID_TO_NAME, NAME_TO_ID, collapse_to_bev
from .line_filter import LineSegment
from .pipeline import LaneDetector, LaneResult
from .providers import (
    ClassicalProvider,
    RoadSegProvider,
    SegmentationProvider,
    SegResult,
    YoloProvider,
    build_provider,
)
from .rig import BevResult, Rig
from .rig_config import CameraCfg, ProjectionCfg, RigConfig, SegmentationCfg

__all__ = [
    "LaneDetector",
    "LaneResult",
    "LaneConfig",
    "WhiteCfg",
    "AsphaltCfg",
    "AdaptiveCfg",
    "LineFilterCfg",
    "TemporalCfg",
    "LineSegment",
    "Rig",
    "BevResult",
    "RigConfig",
    "CameraCfg",
    "ProjectionCfg",
    "GroundGrid",
    "HomographyProjector",
    "DepthProjector",
    "fuse_grids",
    "CLASS_BACKGROUND",
    "CLASS_LANE",
    "CLASS_OBSTACLE",
    "SegResult",
    "SegmentationProvider",
    "ClassicalProvider",
    "YoloProvider",
    "RoadSegProvider",
    "build_provider",
    "SegmentationCfg",
    "DRIVING_CLASSES",
    "NAME_TO_ID",
    "ID_TO_NAME",
    "collapse_to_bev",
]

__version__ = "0.1.0"
