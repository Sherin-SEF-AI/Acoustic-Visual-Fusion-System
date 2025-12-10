"""Fusion module for audio-visual correlation."""

from .audio_visual_fusion import AudioVisualFusion, FusionResult
from .spatial_correlator import SpatialCorrelator

__all__ = ["AudioVisualFusion", "FusionResult", "SpatialCorrelator"]
