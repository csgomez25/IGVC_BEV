"""
Configuration for the portable lane-CV pipeline.

Everything that is robot- or venue-specific lives here, as plain data, so
porting the detector to another vehicle is a YAML edit — never a code edit.
A new car needs, at most:

  * its camera resolution / ROI polygon  (where is the sky / hood)
  * white-paint + asphalt HSV ranges     (venue lighting + surface color)
  * the line-shape gates                 (how thin/long is a real lane line
                                          at this camera's mount + resolution)

Load order of precedence: explicit kwargs > YAML file > built-in defaults.
PyYAML is optional; without it the built-in defaults (or a dict you pass in)
still work, so the library has a hard dependency only on numpy + opencv.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import yaml  # optional
    _HAVE_YAML = True
except Exception:  # pragma: no cover - yaml is optional
    _HAVE_YAML = False


# Normalized image polygon (x, y in 0..1) describing the region we KEEP.
# Default keeps the lower 65% of the frame and drops the top (sky / trees /
# distant over-exposed pavement that wrecks the adaptive brightness floor).
_DEFAULT_ROI = [[0.0, 0.35], [1.0, 0.35], [1.0, 1.0], [0.0, 1.0]]


@dataclass
class WhiteCfg:
    """Low-saturation 'bright achromatic' gate for painted white lines."""
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 60       # white/gray paint is desaturated
    v_min: int = 200      # static floor; raised dynamically by the adaptive gate
    v_max: int = 255


@dataclass
class AsphaltCfg:
    """
    Sooner-2025 style 'drivable surface' gate. We threshold the asphalt
    (dim, low-saturation) and INVERT, giving a 'not-drivable' mask. Used only
    as an optional AND-gate to suppress white-ish *grass/sky*; it does NOT by
    itself reject road specks (specks are also not-asphalt) — that is the job
    of the geometric line filter.
    """
    enabled: bool = True
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 95
    v_min: int = 0
    v_max: int = 210


@dataclass
class ContrastCfg:
    """
    Contrast-relative ('local') white detection, used by `combine: tophat`.

    A white top-hat (V minus a morphological opening of V) keeps pixels that are
    brighter THAN THEIR LOCAL NEIGHBORHOOD rather than brighter than an absolute V
    floor. That is what lets one profile survive BOTH shadow (where a real line's
    absolute V drops below any fixed floor) and bright concrete (where the floor
    also passes the background) — the two failure modes the absolute white/asphalt
    gates hit. Saturation is still gated (via WhiteCfg.s_max) so colored clutter
    (grass, barrels) can't pass as 'bright'.
    """
    # Structuring-element size for the opening. MUST exceed the painted line's
    # stroke width in px (else the line is removed by the opening and the top-hat
    # is empty); a few x the stroke is ideal.
    tophat_ksize: int = 25
    # How many V counts a pixel must exceed its local background to count as paint.
    min_contrast: int = 18
    # Absolute brightness floor: drop faint texture lit up in very dark regions.
    v_min: int = 60


@dataclass
class AdaptiveCfg:
    """
    iscumd-style adaptive brightness floor: V_floor = mean + k*sigma, sampled
    ONLY inside a near-field band so sky / hood / distant pavement can't drag
    it around. Refreshed every `period` frames. This is what lets one config
    survive sun/shadow transitions.
    """
    enabled: bool = True
    k: float = 2.5
    period: int = 5
    # Vertical band [top_frac, bottom_frac] of the frame to sample stats from.
    band: List[float] = field(default_factory=lambda: [0.55, 0.95])


@dataclass
class LineFilterCfg:
    """
    THE speck killer. After color thresholding we keep a connected component
    only if its *shape* looks like a painted line, not a blob:

      * area in [min_area, max_area]            — drop dust and giant glare
      * elongation = long_side/short_side       — lines are long & thin; tar
        >= min_elongation                          cracks/pebbles/specks are
                                                   stubby, so they fail here
      * fill = area / minAreaRect_area          — reject solid bright patches
        <= max_fill (1.0 disables)                 (a filled square is not a line)

    Optional orientation gate keeps only components whose long axis falls in
    [angle_center +/- angle_tol] degrees (0 = horizontal, 90 = vertical in
    image space). Off by default because lane orientation is car/scene
    specific; turn it on per venue if specks share a color but not a heading.
    """
    min_area: int = 80
    max_area: int = 40000
    min_elongation: float = 4.0
    max_fill: float = 0.75
    orientation_gate: bool = False
    angle_center_deg: float = 90.0
    angle_tol_deg: float = 35.0
    # Hough confirmation: require each kept component to contain a straight
    # segment at least this many px long (0 disables the extra check).
    hough_min_len_px: int = 0
    # --- Dashed-line recovery -------------------------------------------------
    # A single dash is too short to clear min_elongation, so the speck filter
    # would drop it. Instead, components that are line-ISH but too short (here:
    # dash_min_elongation <= elong < min_elongation) are collected and linked
    # into one dashed line when several are COLLINEAR (end-to-end, same heading).
    # Random specks/cracks don't form collinear chains, so this recovers dashes
    # without re-admitting clutter. Set dash_link: false to disable.
    dash_link: bool = True
    dash_min_elongation: float = 1.8      # below this = round speck, never a dash
    dash_link_max_gap_px: float = 70.0    # max centroid spacing to chain two dashes
    dash_link_angle_tol_deg: float = 14.0 # heading + collinearity tolerance
    dash_link_min_segments: int = 3       # chain must have at least this many dashes


@dataclass
class TemporalCfg:
    """
    Frame-to-frame confirmation. A lane pixel must survive in N of the last
    `window` frames before it is emitted, so a one-frame glint off a wet speck
    never becomes a line. Set window=1 to disable (pure per-frame).
    """
    window: int = 3
    min_hits: int = 2


@dataclass
class LaneConfig:
    """Top-level config; the only object the pipeline needs."""
    # Preprocessing
    blur_ksize: int = 5
    blur_iters: int = 1   # 1 denoises without fragmenting thin lines; the
                          # geometric line filter handles residual speckle
    # Region of interest (normalized polygon we keep)
    roi_poly: List[List[float]] = field(default_factory=lambda: list(_DEFAULT_ROI))
    # Candidate-generation cues
    white: WhiteCfg = field(default_factory=WhiteCfg)
    asphalt: AsphaltCfg = field(default_factory=AsphaltCfg)
    adaptive: AdaptiveCfg = field(default_factory=AdaptiveCfg)
    contrast: ContrastCfg = field(default_factory=ContrastCfg)
    # How to combine the cues:
    #   'tophat'       : contrast-relative local white gate (shadow/concrete robust)
    #   'white'        : absolute white gate only
    #   'white_gated'  : white AND (not-asphalt)  -> suppresses bright grass/sky
    #   'asphalt_inv'  : not-asphalt only (single-class, Sooner-25 style)
    combine: str = "tophat"
    # Cleanup + the geometric speck filter
    open_ksize: int = 3
    line: LineFilterCfg = field(default_factory=LineFilterCfg)
    temporal: TemporalCfg = field(default_factory=TemporalCfg)
    # Optional metadata (not used by the math, handy for downstream/porting)
    meters_per_px: float = 0.0
    notes: str = ""

    # ---------------------------------------------------------------- loaders
    @classmethod
    def from_dict(cls, d: dict) -> "LaneConfig":
        d = dict(d or {})
        sub = {
            "white": WhiteCfg, "asphalt": AsphaltCfg, "adaptive": AdaptiveCfg,
            "contrast": ContrastCfg, "line": LineFilterCfg, "temporal": TemporalCfg,
        }
        kwargs = {}
        for key, value in d.items():
            if key in sub and isinstance(value, dict):
                kwargs[key] = sub[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "LaneConfig":
        if not _HAVE_YAML:
            raise RuntimeError(
                "PyYAML not installed; pass a dict to LaneConfig.from_dict() "
                "or `pip install pyyaml`."
            )
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_yaml(self, path: str) -> None:
        if not _HAVE_YAML:
            raise RuntimeError("PyYAML not installed; cannot serialize.")
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)
