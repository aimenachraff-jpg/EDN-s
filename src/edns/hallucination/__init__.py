"""
EDN-S Hallucination Detection
=============================

Hallucination taxonomy and detection methods for satellite super-resolution.
Four categories: Spectral, Structural, Textural, Semantic.

This module provides:
- Hallucination taxonomy definitions
- Detection algorithms for each category
- Expert annotation interface utilities
- Metric validation tools (for H2)
"""

from .detection import (
    HallucinationType,
    HallucinationThresholds,
    HallucinationDetector,
    AnnotationSample,
    AnnotationProtocol,
    validate_hallucination_metrics,
)

__all__ = [
    "HallucinationType",
    "HallucinationThresholds",
    "HallucinationDetector",
    "AnnotationSample",
    "AnnotationProtocol",
    "validate_hallucination_metrics",
]