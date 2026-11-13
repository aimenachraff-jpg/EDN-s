"""Hallucination Detection Framework."""

from .detection import (
    HallucinationDetector,
    HallucinationThresholds,
    HallucinationType,
    AnnotationProtocol,
    AnnotationSample,
)

__all__ = [
    'HallucinationDetector',
    'HallucinationThresholds',
    'HallucinationType',
    'AnnotationProtocol',
    'AnnotationSample',
]