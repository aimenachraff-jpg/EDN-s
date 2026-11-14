"""
Tiling utilities for satellite imagery processing.

Provides functions to tile large satellite images for efficient processing:
- tile_image: Split large images into overlapping tiles
- untile_image: Reconstruct image from tiles with blending
- calculate_tiles: Calculate tile grid parameters
- create_weight_kernel: Create Gaussian blending weights
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter


def tile_image(image: np.ndarray, tile_size: int, overlap: int = 0,
               min_tile_size: int = 32) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Tile a large image into overlapping tiles.
    
    Args:
        image: Input image array [C, H, W] or [H, W]
        tile_size: Size of tiles (pixels)
        overlap: Overlap between adjacent tiles
        min_tile_size: Minimum tile size before splitting
    
    Returns:
        tiles: List of tile arrays
        positions: List of (row, col) tile positions
    """
    if image.ndim == 3:
        channels, height, width = image.shape
    else:
        channels, height, width = 1, image.shape[0], image.shape[1]
    
    step = tile_size - overlap
    if step < min_tile_size:
        raise ValueError(f"Step size {step} < min_tile_size {min_tile_size}")
    
    # Calculate number of tiles needed
    n_tiles_h = max(1, (height + step - 1) // step)
    n_tiles_w = max(1, (width + step - 1) // step)
    
    tiles = []
    positions = []
    
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile boundaries
            h_start = i * step
            w_start = j * step
            h_end = min(h_start + tile_size, height)
            w_end = min(w_start + tile_size, width)
            
            # Handle edge tiles
            if h_end - h_start < min_tile_size:
                h_start = max(0, height - tile_size)
                h_end = h_start + tile_size
            if w_end - w_start < min_tile_size:
                w_start = max(0, width - tile_size)
                w_end = w_start + tile_size
            
            # Extract tile
            tile = image[:, h_start:h_end, w_start:w_end] if image.ndim == 3 else image[h_start:h_end, w_start:w_end]
            
            # Pad to tile_size if needed
            if tile.shape[-2] < tile_size or tile.shape[-1] < tile_size:
                pad_h = tile_size - tile.shape[-2]
                pad_w = tile_size - tile.shape[-1]
                if image.ndim == 3:
                    tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                else:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            tiles.append(tile)
            positions.append((i, j))
    
    return tiles, positions


def untile_image(tiles: List[np.ndarray], positions: List[Tuple[int, int]],
                output_shape: Tuple[int, int], tile_size: int,
                overlap: int = 0, blend_weights: Optional[np.ndarray] = None,
                weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct an image from tiles.
    
    Args:
        tiles: List of tile arrays
        positions: List of (row, col) tile positions
        output_shape: Output image shape (H, W)
        tile_size: Original tile size
        overlap: Overlap between tiles
        blend_weights: Optional blending weights for edge tiles
        weights: Optional weight map for tile overlap blending
    
    Returns:
        Reconstructed image
    """
    if len(tiles) == 0:
        raise ValueError("No tiles provided")
    
    # Determine if 3D or 2D
    sample_tile = tiles[0]
    is_3d = len(sample_tile.shape) == 3

    # Handle output_shape: can be (H, W) or (C, H, W)
    if is_3d:
        if len(output_shape) == 2:
            height, width = output_shape
            channels = sample_tile.shape[0]
        else:
            channels, height, width = output_shape
    else:
        height, width = output_shape[0], output_shape[1]
        channels = 1

    # Initialize output and weight maps
    if is_3d:
        output = np.zeros((channels, height, width), dtype=np.float32)
    else:
        output = np.zeros((height, width), dtype=np.float32)

    weight_map = np.zeros((height, width), dtype=np.float32)

    step = tile_size - overlap
    
    # Place tiles and accumulate weights
    for tile, (i, j) in zip(tiles, positions):
        # Calculate tile position in output
        h_start = i * step
        w_start = j * step
        h_end = min(h_start + tile_size, height)
        w_end = min(w_start + tile_size, width)
        
        # Ensure tile fits
        tile_h, tile_w = tile.shape[-2], tile.shape[-1]
        if is_3d:
            tile_h, tile_w = tile.shape[-2], tile.shape[-1]
        
        actual_h = min(tile_h, h_end - h_start)
        actual_w = min(tile_w, w_end - w_start)
        
        if is_3d:
            output[:, h_start:h_start + actual_h, w_start:w_start + actual_w] += \
                tile[:, :actual_h, :actual_w]
        else:
            output[h_start:h_start + actual_h, w_start:w_start + actual_w] += \
                tile[:actual_h, :actual_w]
        
        # Create weight map
        if blend_weights is not None:
            weight = blend_weights[:actual_h, :actual_w]
        else:
            weight = np.ones((actual_h, actual_w))
        
        weight_map[h_start:h_start + actual_h, w_start:w_start + actual_w] += weight
    
    # Normalize by weight map
    weight_map = np.maximum(weight_map, 1e-8)
    if is_3d:
        output /= weight_map[np.newaxis, :, :]
    else:
        output /= weight_map
    
    return np.clip(output, 0, 1)


def calculate_tiles(image_shape: Tuple[int, int], tile_size: int,
                   overlap: int = 0) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Calculate tile grid parameters for an image.
    
    Args:
        image_shape: Image shape (H, W) or (C, H, W)
        tile_size: Tile size
        overlap: Overlap between tiles
    
    Returns:
        n_tiles_h: Number of tiles in height direction
        n_tiles_w: Number of tiles in width direction
        tile_positions: List of all tile positions
    """
    if len(image_shape) == 3:
        height, width = image_shape[1], image_shape[2]
    else:
        height, width = image_shape[0], image_shape[1]
    
    step = tile_size - overlap
    
    n_tiles_h = max(1, (height + step - 1) // step)
    n_tiles_w = max(1, (width + step - 1) // step)
    
    tile_positions = [(i, j) for i in range(n_tiles_h) for j in range(n_tiles_w)]
    
    return n_tiles_h, n_tiles_w, tile_positions


def create_weight_kernel(tile_size: int, sigma: Optional[float] = None,
                        border_handling: str = 'reflect') -> np.ndarray:
    """
    Create a Gaussian weight kernel for tile blending.
    
    Args:
        tile_size: Size of the kernel
        sigma: Standard deviation for Gaussian
        border_handling: How to handle borders ('reflect', 'constant', 'gaussian')
    
    Returns:
        Weight kernel normalized to sum to 1
    """
    if sigma is None:
        sigma = tile_size / 6
    
    # Create 2D Gaussian kernel
    x = np.arange(tile_size) - tile_size // 2
    y = np.arange(tile_size) - tile_size // 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel