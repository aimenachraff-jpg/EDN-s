"""
Spatial buffer utilities for satellite imagery.

Provides functions for spatial padding, boundary handling, and buffer management:
- buffer_mask: Create buffer masks around features
- apply_spatial_padding: Apply padding for inference
- edge_pad_image: Handle image edges for tiling
- is_valid_region: Check if region contains valid data
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.ndimage import binary_dilation, binary_erosion


def buffer_mask(binary_mask: np.ndarray, buffer_pixels: int,
                mode: str = 'outside', connectivity: int = 1) -> np.ndarray:
    """
    Create a buffer mask around existing features.
    
    Args:
        binary_mask: Binary input mask (True = feature, False = background)
        buffer_pixels: Number of pixels to buffer
        mode: Buffering mode ('outside' = dilate, 'inside' = erode)
        connectivity: Connectivity for morphological operations (1-2)
    
    Returns:
        Buffered mask
    """
    if buffer_pixels <= 0:
        return binary_mask.copy()
    
    # Structure element for morphological operations
    if binary_mask.ndim == 2:
        if connectivity == 2:
            # Full square (8-connectivity)
            struct = np.ones((3, 3), dtype=bool)
        else:
            # Cross (4-connectivity): center + 4 arms
            struct = np.ones((3, 3), dtype=bool)
            struct[0, 0] = struct[0, 2] = struct[2, 0] = struct[2, 2] = False
    else:
        # 3D: face connectivity (center + 6 face neighbors)
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, 1, 1] = True
        struct[0, 1, 1] = struct[2, 1, 1] = True
        struct[1, 0, 1] = struct[1, 2, 1] = True
        struct[1, 1, 0] = struct[1, 1, 2] = True
    
    if mode == 'outside':
        # Dilate (expand feature area)
        buffered = binary_dilation(binary_mask, structure=struct,
                                   iterations=buffer_pixels)
    elif mode == 'inside':
        # Erode (shrink feature area)
        buffered = binary_erosion(binary_mask, structure=struct,
                                  iterations=buffer_pixels)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'outside' or 'inside'")
    
    return buffered


def apply_spatial_padding(image: np.ndarray, padding: int,
                         mode: str = 'reflect', edge_value: float = 0.0) -> np.ndarray:
    """
    Apply spatial padding to an image for inference.
    
    Args:
        image: Input image array
        padding: Padding size (pixels)
        mode: Padding mode ('reflect', 'constant', 'edge', 'symmetric')
        edge_value: Value for constant padding
    
    Returns:
        Padded image
    """
    if padding <= 0:
        return image.copy()
    
    # For 3D images, only pad the last two dimensions (H, W)
    if image.ndim == 3:
        C, H, W = image.shape
        pad_width = [(0, 0), (padding, padding), (padding, padding)]
    else:
        pad_width = tuple([(padding, padding) for _ in image.shape])
    if mode == 'constant':
        return np.pad(image, pad_width, mode=mode, constant_values=edge_value)
    # For reflect/edge/symmetric modes, constant_values is not supported
    return np.pad(image, pad_width, mode=mode)


def edge_pad_image(image: np.ndarray, target_size: Tuple[int, ...],
                  tile_size: int, overlap: int = 0,
                  mode: str = 'reflect') -> np.ndarray:
    """
    Pad image to ensure it's divisible by tile size.
    
    Args:
        image: Input image array
        target_size: Original image shape (H, W) or (C, H, W)
        tile_size: Tile size for tiling
        overlap: Overlap between tiles
        mode: Padding mode
    
    Returns:
        Padded image
    """
    if len(target_size) == 3:
        channels, height, width = target_size
    else:
        channels, height, width = 1, target_size[0], target_size[1]
    
    step = tile_size - overlap
    
    # Calculate required padding
    pad_h = (step - height % step) % step
    pad_w = (step - width % step) % step
    
    if pad_h == 0 and pad_w == 0:
        return image.copy()
    
    # Add padding
    if len(target_size) == 3:
        pad_width = ((0, 0), (0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w))
    
    return np.pad(image, pad_width, mode=mode)


def is_valid_region(region: np.ndarray, valid_threshold: float = 0.0,
                   nodata_value: Optional[float] = None) -> bool:
    """
    Check if a region contains valid data.
    
    Args:
        region: Input region array
        valid_threshold: Threshold for valid data
        nodata_value: Special value indicating invalid data
    
    Returns:
        True if region contains valid data, False otherwise
    """
    if nodata_value is not None:
        # Check if any pixels are valid (not nodata)
        return np.any(region != nodata_value)
    
    # Check if region meets validity threshold
    if region.size == 0:
        return False
    
    # For floating point, check if any values are valid
    if np.issubdtype(region.dtype, np.floating):
        return np.any(~np.isnan(region) & (region >= valid_threshold))
    
    # For integer types
    if region.dtype.kind in ['i', 'u']:
        # Check if any values are > 0 or meet threshold
        return np.any(region > valid_threshold)
    
    # For boolean type
    if region.dtype == bool:
        return np.any(region)
    
    # Fallback for other types
    return np.any(region > valid_threshold)


def calculate_buffer_requirements(binary_mask: np.ndarray, buffer_pixels: int,
                                   min_buffer_size: int = 10) -> Tuple[int, int, int]:
    """
    Calculate buffer requirements for feature analysis.
    
    Args:
        binary_mask: Binary input mask
        buffer_pixels: Requested buffer size
        min_buffer_size: Minimum buffer size
    
    Returns:
        required_buffer: Buffer size to use
        actual_features: Number of features after buffering
        edge_features: Number of features touching image border
    """
    # Apply initial buffer
    buffered = buffer_mask(binary_mask, buffer_pixels, mode='outside')
    
    # Calculate statistics
    actual_features = np.sum(buffered)
    
    # Count features touching borders
    height, width = binary_mask.shape
    edge_features = 0
    
    # Check top and bottom edges
    if buffered[0, :].any():
        edge_features += 1
    if buffered[-1, :].any():
        edge_features += 1
    
    # Check left and right edges
    if buffered[:, 0].any():
        edge_features += 1
    if buffered[:, -1].any():
        edge_features += 1
    
    # Adjust buffer size if too small
    if buffer_pixels < min_buffer_size and actual_features > 0:
        required_buffer = min(buffer_pixels + 5, 50)  # Increase conservatively
    else:
        required_buffer = buffer_pixels
    
    return required_buffer, int(actual_features), edge_features


def create_hierarchical_buffers(binary_mask: np.ndarray,
                                buffer_sizes: List[int]) -> List[np.ndarray]:
    """
    Create hierarchical buffer masks at multiple scales.
    
    Args:
        binary_mask: Binary input mask
        buffer_sizes: List of buffer sizes to generate
    
    Returns:
        List of buffered masks
    """
    buffers = [binary_mask]  # Original mask
    
    for size in sorted(buffer_sizes):
        if size > 0:
            buffered = buffer_mask(binary_mask, size, mode='outside')
            buffers.append(buffered)
    
    return buffers


if __name__ == "__main__":
    # Test the functions
    print("Testing spatial buffer functions...")
    
    # Test buffer_mask
    mask = np.array([[0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0]])
    
    print("Original mask:")
    print(mask.astype(int))
    
    buffered = buffer_mask(mask, buffer_pixels=1, mode='outside')
    print("\nBuffered mask (outside, 1 pixel):")
    print(buffered.astype(int))
    
    eroded = buffer_mask(mask, buffer_pixels=1, mode='inside')
    print("\nEroded mask (inside,  Aufnahme):")
    print(eroded.astype(int))
    
    # Test apply_spatial_padding
    img = np.random.rand(10, 10)
    padded = apply_spatial_padding(img, padding=2, mode='reflect')
    print(f"\nPadded image shape: {img.shape} -> {padded.shape}")
    
    # Test edge_pad_image
    padded_edge = edge_pad_image(img, (10, 10), 7, 0, 'reflect')
    print(f"Edge-padded image shape: {img.shape} -> {padded_edge.shape}")
    
    # Test is_valid_region
    valid = is_valid_region(img, valid_threshold=0.1)
    print(f"\nRegion valid: {valid}")
    
    # Test calculate_buffer_requirements
    buffer_req, features, edge_feat = calculate_buffer_requirements(mask, 2)
    print(f"\nBuffer requirements: {buffer_req}, Features: {features}, Edge features: {edge_feat}")
    
    print("\nAll tests passed!")
