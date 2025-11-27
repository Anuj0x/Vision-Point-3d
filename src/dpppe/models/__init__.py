"""
Modernized model components for 3DPPE.
Consolidated from multiple legacy files into clean, efficient implementations.
"""

from .detector import PETR3DDetector
from .backbones import VoVNetBackbone
from .transformers import MultiViewTransformer, PointPositionalEncoder
from .depth import DepthEncoder, HybridDepthModule
from .losses import DPPESetCriterion

__all__ = [
    "PETR3DDetector",
    "VoVNetBackbone",
    "MultiViewTransformer",
    "PointPositionalEncoder",
    "DepthEncoder",
    "HybridDepthModule",
    "DPPESetCriterion",
]
