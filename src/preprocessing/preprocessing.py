"""
Preprocessing Utilities

Common preprocessing operations for satellite imagery:
- Cloud masking
- Normalization
- Tiling/untilling
- Band alignment
- Quality filtering
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def percentile_normalize(image: np.ndarray,
                         p_low: float = 2,
                         p_high: float = 98,
                         clip: bool = True,
                         output_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize image using percentile clipping.

    Args:
        image: Input image array
        p_low: Lower percentile
        p_high: Upper percentile
        clip: Whether to clip values
        output_range: Target range (min, max)

    Returns:
        Normalized image
    """
    # Flatten and compute percentiles (ignore zeros/nodata)
    valid = image[image > 0]
    if len(valid) == 0:
        return np.zeros_like(image, dtype=np.float32)

    low_val = np.percentile(valid, p_low)
    high_val = np.percentile(valid, p_high)

    if clip:
        image = np.clip(image, low_val, high_val)

    # Normalize
    normalized = (image - low_val) / (high_val - low_val + 1e-8)

    # Scale to output range
    out_min, out_max = output_range
    normalized = normalized * (out_max - out_min) + out_min

    return normalized.astype(np.float32)


def apply_normalization_params(image: np.ndarray,
                                params: Dict[str, float],
                                clip: bool = True,
                                output_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Apply pre-computed normalization parameters.

    Args:
        image: Input image
        params: Dict with 'p2' and 'p98' keys
        clip: Whether to clip
        output_range: Target range

    Returns:
        Normalized image
    """
    return percentile_normalize(
        image,
        p_low=params['p2'] * 100 / params.get('max', 1),  # Approximate - better to store actual percentiles
        p_high=params['p98'] * 100 / params.get('max', 1),
        clip=clip,
        output_range=output_range
    )


def cloud_mask_from_scl(scl_band: np.ndarray,
                        cloud_classes: List[int] = None) -> np.ndarray:
    """
    Create cloud mask from Sen2Cor Scene Classification Layer (SCL).

    SCL classes:
    0: NO_DATA
    1: SATURATED_OR_DEFECTIVE
    2: DARK_AREA_PIXELS
    3: CLOUD_SHADOWS
    4: VEGETATION
    5: NOT_VEGETATED
    6: WATER
    7: UNCLASSIFIED
    8: CLOUD_MEDIUM_PROBABILITY
    9: CLOUD_HIGH_PROBABILITY
    10: THIN_CIRRUS
    11: SNOW

    Args:
        scl_band: SCL band array
        cloud_classes: Classes to mask (default: shadows, clouds, cirrus)

    Returns:
        Boolean mask (True = clear, False = masked)
    """
    if cloud_classes is None:
        cloud_classes = [1, 3, 8, 9, 10, 11]  # defective, shadow, clouds, cirrus, snow

    mask = np.ones_like(scl_band, dtype=bool)
    for cls in cloud_classes:
        mask &= (scl_band != cls)

    return mask


def buffer_mask(mask: np.ndarray, buffer_pixels: int = 3) -> np.ndarray:
    """Dilate mask to buffer around cloud edges."""
    from scipy.ndimage import binary_dilation
    if buffer_pixels > 0:
        structure = np.ones((buffer_pixels * 2 + 1, buffer_pixels * 2 + 1))
        mask = binary_dilation(~mask, structure=structure)
        mask = ~mask
    return mask


def tile_image(image: np.ndarray,
               tile_size: int,
               overlap: int = 0) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Split image into overlapping tiles.

    Args:
        image: [H, W] or [C, H, W] array
        tile_size: Tile size in pixels
        overlap: Overlap between tiles

    Returns:
        List of tiles, list of (y, x) positions
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    elif image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    C, H, W = image.shape
    stride = tile_size - overlap

    tiles = []
    positions = []

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[:, y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions


def untile_image(tiles: List[np.ndarray],
                 positions: List[Tuple[int, int]],
                 output_shape: Tuple[int, int],
                 tile_size: int,
                 overlap: int = 0,
                 blend: bool = True) -> np.ndarray:
    """
    Reconstruct image from tiles with optional blending in overlaps.

    Args:
        tiles: List of [C, tile_size, tile_size] arrays
        positions: List of (y, x) top-left positions
        output_shape: (H, W) of output
        tile_size: Tile size
        overlap: Overlap used during tiling
        blend: Whether to blend overlaps (feathering)

    Returns:
        Reconstructed image [C, H, W]
    """
    C = tiles[0].shape[0]
    H, W = output_shape
    output = np.zeros((C, H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    stride = tile_size - overlap

    for tile, (y, x) in zip(tiles, positions):
        if blend and overlap > 0:
            # Create feathering weights
            w = np.ones((tile_size, tile_size), dtype=np.float32)
            fade = np.linspace(0, 1, overlap)
            # Top edge
            if y > 0:
                w[:overlap, :] *= fade[:, np.newaxis]
            # Bottom edge
            if y + tile_size < H:
                w[-overlap:, :] *= fade[::-1, np.newaxis]
            # Left edge
            if x > 0:
                w[:, :overlap] *= fade[np.newaxis, :]
            # Right edge
            if x + tile_size < W:
                w[:, -overlap:] *= fade[::-1, np.newaxis]
        else:
            w = np.ones((tile_size, tile_size), dtype=np.float32)

        output[:, y:y+tile_size, x:x+tile_size] += tile * w[np.newaxis, :, :]
        weight[y:y+tile_size, x:x+tile_size] += w

    # Avoid division by zero
    weight[weight == 0] = 1
    output = output / weight[np.newaxis, :, :]

    return output


def align_bands(bands_dict: Dict[str, np.ndarray],
                reference_band: str,
                method: str = 'nearest') -> Dict[str, np.ndarray]:
    """
    Align multiple bands to a reference band.

    Args:
        bands_dict: Dict of band_name -> array [H, W]
        reference_band: Band to align others to
        method: Resampling method

    Returns:
        Aligned bands dict
    """
    if reference_band not in bands_dict:
        raise ValueError(f"Reference band {reference_band} not in bands")

    ref_shape = bands_dict[reference_band].shape
    aligned = {reference_band: bands_dict[reference_band]}

    for name, band in bands_dict.items():
        if name == reference_band:
            continue
        if band.shape != ref_shape:
            # Resize
            from skimage.transform import resize
            aligned[name] = resize(band, ref_shape, order=1 if method == 'bilinear' else 0,
                                   preserve_range=True, anti_aliasing=False)
        else:
            aligned[name] = band

    return aligned


def compute_band_statistics(image: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for an image band."""
    valid = image[image > 0]
    if len(valid) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p2': 0, 'p98': 0}

    return {
        'mean': float(valid.mean()),
        'std': float(valid.std()),
        'min': float(valid.min()),
        'max': float(valid.max()),
        'p2': float(np.percentile(valid, 2)),
        'p98': float(np.percentile(valid, 98))
    }


def spectral_angle_mapper(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM) between two spectra.

    Args:
        spectrum1: [C] or [C, H, W] - first spectrum
        spectrum2: [C] or [C, H, W] - second spectrum

    Returns:
        Mean SAM in radians
    """
    if spectrum1.ndim == 1:
        spectrum1 = spectrum1[:, np.newaxis, np.newaxis]
    if spectrum2.ndim == 1:
        spectrum2 = spectrum2[:, np.newaxis, np.newaxis]

    # Compute dot product and norms
    dot = np.sum(spectrum1 * spectrum2, axis=0)
    norm1 = np.linalg.norm(spectrum1, axis=0)
    norm2 = np.linalg.norm(spectrum2, axis=0)

    # Cosine of angle
    cos_angle = dot / (norm1 * norm2 + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)

    # Angle in radians
    sam = np.arccos(cos_angle)

    return float(np.mean(sam))


def ergas(index: np.ndarray, reference: np.ndarray, scale: int = 4) -> float:
    """
    Compute ERGAS (Relative Global Dimensionless Synthesis) index.

    Args:
        index: SR image [C, H, W]
        reference: HR reference [C, H, W]
        scale: Scale factor

    Returns:
        ERGAS value
    """
    C = index.shape[0]
    ergas_sum = 0

    for c in range(C):
        mse = np.mean((index[c] - reference[c]) ** 2)
        mean_ref = np.mean(reference[c])
        if mean_ref > 0:
            ergas_sum += mse / (mean_ref ** 2)

    return 100 / scale * np.sqrt(ergas_sum / C)


def sam_index(index: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute mean SAM across all bands.

    Args:
        index: SR image [C, H, W]
        reference: HR reference [C, H, W]

    Returns:
        Mean SAM in degrees
    """
    sam_vals = []
    for c in range(index.shape[0]):
        sam = spectral_angle_mapper(index[c], reference[c])
        sam_vals.append(sam)
    return float(np.mean(sam_vals)) * 180 / np.pi  # Convert to degrees


def main():
    """Test preprocessing functions."""
    # Test normalization
    img = np.random.rand(100, 100).astype(np.float32) * 1000
    norm = percentile_normalize(img)
    assert norm.min() >= 0 and norm.max() <= 1

    # Test tiling
    tiles, pos = tile_image(np.random.rand(512, 512), 256, 64)
    assert len(tiles) == 9  # 3x3 grid
    reconstructed = untile_image(tiles, pos, (512, 512), 256, 64)
    assert reconstructed.shape == (512, 512)

    # Test SAM
    s1 = np.array([1, 2, 3, 4], dtype=np.float32)
    s2 = np.array([1, 2, 3, 4], dtype=np.float32)
    sam = spectral_angle_mapper(s1, s2)
    assert sam == 0.0

    print("All preprocessing tests passed!")


if __name__ == "__main__":
    main()