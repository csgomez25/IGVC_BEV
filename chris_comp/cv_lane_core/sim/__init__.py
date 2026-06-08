"""Synthetic scene generator + scoring for phase-by-phase CV testing."""

from .metrics import Score, score
from .scene import Scene, default_suite, make_scene

__all__ = ["Scene", "make_scene", "default_suite", "Score", "score"]
