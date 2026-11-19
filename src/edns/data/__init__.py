"""
EDN-S Data Module
=================

Utilities for loading and preprocessing satellite imagery.
"""

from .io import (
    load_sentinel2_band,
    load_sentinel2_stack,
    save_geotiff,
    find_sentinel2_bands,
)

from .normalization import (
    normalize_band,
    denormalize_band,
    normalize_stack,
    denormalize_stack,
    load_normalization_stats,
)

__all__ = [
    "load_sentinel2_band",
    "load_sentinel2_stack",
    "save_geotiff",
    "find_sentinel2_bands",
    "normalize_band",
    "denormalize_band",
    "normalize_stack",
    "denormalize_stack",
    "load_normalization_stats",
]