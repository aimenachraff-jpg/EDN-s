"""
EDN-S Normalization
===================

Z-score normalization for Sentinel-2 bands using training statistics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


# Default normalization stats path
DEFAULT_STATS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "metadata" / "sentinel2" / "normalization_stats.json"


def load_normalization_stats(path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load normalization statistics from JSON file.
    
    Args:
        path: Path to normalization stats JSON (uses default if None)
        
    Returns:
        Dict with band statistics (mean, std, p2, p98)
    """
    if path is None:
        path = DEFAULT_STATS_PATH
    
    with open(path) as f:
        stats = json.load(f)
    
    return stats


def normalize_band(
    band: np.ndarray,
    band_name: str,
    stats: Optional[Dict] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize a single band using training statistics.
    
    Args:
        band: Input band array [H, W]
        band_name: Band name (B2, B3, B4, B8)
        stats: Normalization stats dict (loads default if None)
        method: Normalization method ('zscore' or 'percentile')
        
    Returns:
        Normalized band array
    """
    if stats is None:
        stats = load_normalization_stats()
    
    if band_name not in stats['bands']:
        raise ValueError(f"Band {band_name} not found in stats")
    
    band_stats = stats['bands'][band_name]
    
    # Handle NaN values
    nan_mask = np.isnan(band)
    band = np.nan_to_num(band, nan=0.0)
    
    if method == "zscore":
        mean = band_stats['mean']
        std = band_stats['std']
        normalized = (band - mean) / (std + 1e-8)
    elif method == "percentile":
        p2 = band_stats.get('p2', band_stats['mean'] - 2 * band_stats['std'])
        p98 = band_stats.get('p98', band_stats['mean'] + 2 * band_stats['std'])
        normalized = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Restore NaN
    normalized[nan_mask] = np.nan
    
    return normalized


def denormalize_band(
    band: np.ndarray,
    band_name: str,
    stats: Optional[Dict] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Denormalize a band back to original range.
    
    Args:
        band: Normalized band array [H, W]
        band_name: Band name (B2, B3, B4, B8)
        stats: Normalization stats dict (loads default if None)
        method: Normalization method ('zscore' or 'percentile')
        
    Returns:
        Denormalized band array
    """
    if stats is None:
        stats = load_normalization_stats()
    
    if band_name not in stats['bands']:
        raise ValueError(f"Band {band_name} not found in stats")
    
    band_stats = stats['bands'][band_name]
    
    # Handle NaN values
    nan_mask = np.isnan(band)
    band = np.nan_to_num(band, nan=0.0)
    
    if method == "zscore":
        mean = band_stats['mean']
        std = band_stats['std']
        denormalized = band * std + mean
    elif method == "percentile":
        p2 = band_stats.get('p2', band_stats['mean'] - 2 * band_stats['std'])
        p98 = band_stats.get('p98', band_stats['mean'] + 2 * band_stats['std'])
        denormalized = band * (p98 - p2) + p2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Restore NaN
    denormalized[nan_mask] = np.nan
    
    return denormalized


def normalize_stack(
    stack: np.ndarray,
    band_order: List[str] = ['B2', 'B3', 'B4', 'B8'],
    stats: Optional[Dict] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize a multi-band stack.
    
    Args:
        stack: Input stack [C, H, W]
        band_order: List of band names matching stack channels
        stats: Normalization stats dict (loads default if None)
        method: Normalization method
        
    Returns:
        Normalized stack [C, H, W]
    """
    if stats is None:
        stats = load_normalization_stats()
    
    if len(band_order) != stack.shape[0]:
        raise ValueError(f"Band order length {len(band_order)} != channels {stack.shape[0]}")
    
    normalized = np.zeros_like(stack)
    for i, band_name in enumerate(band_order):
        normalized[i] = normalize_band(stack[i], band_name, stats, method)
    
    return normalized


def denormalize_stack(
    stack: np.ndarray,
    band_order: List[str] = ['B2', 'B3', 'B4', 'B8'],
    stats: Optional[Dict] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Denormalize a multi-band stack.
    
    Args:
        stack: Normalized stack [C, H, W]
        band_order: List of band names matching stack channels
        stats: Normalization stats dict (loads default if None)
        method: Normalization method
        
    Returns:
        Denormalized stack [C, H, W]
    """
    if stats is None:
        stats = load_normalization_stats()
    
    if len(band_order) != stack.shape[0]:
        raise ValueError(f"Band order length {len(band_order)} != channels {stack.shape[0]}")
    
    denormalized = np.zeros_like(stack)
    for i, band_name in enumerate(band_order):
        denormalized[i] = denormalize_band(stack[i], band_name, stats, method)
    
    return denormalized