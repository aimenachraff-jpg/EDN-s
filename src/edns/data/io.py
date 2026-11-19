"""
EDN-S Data I/O
==============

Reading and writing satellite GeoTIFF files.
"""

import rasterio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union


def load_sentinel2_band(
    filepath: Union[str, Path], 
    band_idx: int = 1
) -> Tuple[np.ndarray, dict]:
    """
    Load a single band from a Sentinel-2 GeoTIFF.
    
    Args:
        filepath: Path to GeoTIFF file
        band_idx: 1-based band index (default: 1)
        
    Returns:
        Tuple of (array [H, W], profile_dict)
    """
    with rasterio.open(filepath) as src:
        arr = src.read(band_idx).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def load_sentinel2_stack(
    band_paths: List[Union[str, Path]],
    band_order: List[str] = ['B2', 'B3', 'B4', 'B8']
) -> Tuple[np.ndarray, dict]:
    """
    Load multiple bands as a stack [C, H, W].
    
    Args:
        band_paths: List of paths to band GeoTIFFs (order should match band_order)
        band_order: Expected band order (default: B2, B3, B4, B8)
        
    Returns:
        Tuple of (array [C, H, W], profile_dict from first band)
    """
    if len(band_paths) != len(band_order):
        raise ValueError(f"Expected {len(band_order)} bands, got {len(band_paths)}")
    
    bands = []
    profile = None
    for i, path in enumerate(band_paths):
        arr, p = load_sentinel2_band(path)
        bands.append(arr)
        if profile is None:
            profile = p
    
    # Stack to [C, H, W]
    stack = np.stack(bands, axis=0)
    return stack, profile


def save_geotiff(
    array: np.ndarray,
    filepath: Union[str, Path],
    profile: dict,
    dtype: str = "float32",
    compress: str = "lzw"
) -> str:
    """
    Save array as GeoTIFF with updated profile.
    
    Args:
        array: Array of shape [C, H, W] or [H, W]
        filepath: Output path
        profile: Rasterio profile dict (will be updated)
        dtype: Output data type
        compress: Compression method
        
    Returns:
        Output filepath
    """
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    
    out_profile = profile.copy()
    out_profile.update(
        dtype=dtype,
        count=array.shape[0],
        height=array.shape[1],
        width=array.shape[2],
        compress=compress
    )
    
    with rasterio.open(filepath, 'w', **out_profile) as dst:
        dst.write(array.astype(dtype))
    
    return str(filepath)


def find_sentinel2_bands(
    directory: Union[str, Path],
    pattern: str = "*_{band}.tif"
) -> dict:
    """
    Find Sentinel-2 band files in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern with {band} placeholder
        
    Returns:
        Dict mapping band names to file paths
    """
    directory = Path(directory)
    bands = {}
    
    for band in ['B2', 'B3', 'B4', 'B8']:
        files = list(directory.glob(pattern.format(band=band)))
        if files:
            bands[band] = files[0]
    
    return bands