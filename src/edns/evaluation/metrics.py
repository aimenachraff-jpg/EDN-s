"""
EDN-S Evaluation Metrics
========================

Comprehensive metrics for super-resolution quality assessment.
All metrics operate on numpy arrays or PyTorch tensors.
Compatible with CPU-only execution.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Union, Tuple, Optional, Any, Callable

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


def _validate_shapes(hr: ArrayLike, sr: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to matching numpy arrays."""
    hr_np = _to_numpy(hr)
    sr_np = _to_numpy(sr)
    if hr_np.shape != sr_np.shape:
        raise ValueError(f"Shape mismatch: HR {hr_np.shape} vs SR {sr_np.shape}")
    return hr_np, sr_np


def compute_psnr(
    hr: ArrayLike, 
    sr: ArrayLike, 
    max_val: float = 1.0
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    Args:
        hr: High-resolution reference image [H, W] or [C, H, W] or [B, C, H, W]
        sr: Super-resolved image (same shape as hr)
        max_val: Maximum possible pixel value (1.0 for [0,1] normalized data)
    
    Returns:
        PSNR value in dB (higher is better)
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    mse = np.mean((hr_np - sr_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def compute_mse(hr: ArrayLike, sr: ArrayLike) -> float:
    """Mean Squared Error."""
    hr_np, sr_np = _validate_shapes(hr, sr)
    return float(np.mean((hr_np - sr_np) ** 2))


def compute_mae(hr: ArrayLike, sr: ArrayLike) -> float:
    """Mean Absolute Error."""
    hr_np, sr_np = _validate_shapes(hr, sr)
    return float(np.mean(np.abs(hr_np - sr_np)))


def compute_ssim(
    hr: ArrayLike,
    sr: ArrayLike,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    channel_axis: Optional[int] = None,
) -> float:
    """
    Structural Similarity Index Measure (SSIM).
    
    Args:
        hr: Reference image [H, W], [C, H, W], or [B, C, H, W]
        sr: Super-resolved image (same shape)
        max_val: Maximum pixel value
        window_size: Gaussian window size
        sigma: Gaussian sigma
        channel_axis: Axis containing channels (None for single-channel)
    
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    from scipy.ndimage import gaussian_filter
    
    def _ssim_single(hr_2d: np.ndarray, sr_2d: np.ndarray) -> float:
        """Compute SSIM for single 2D arrays."""
        if hr_2d.ndim != 2:
            raise ValueError(f"_ssim_single expects 2D arrays, got {hr_2d.ndim}D")
        
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        
        mu_x = gaussian_filter(hr_2d, sigma=sigma, truncate=window_size / (2 * sigma))
        mu_y = gaussian_filter(sr_2d, sigma=sigma, truncate=window_size / (2 * sigma))
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = gaussian_filter(hr_2d ** 2, sigma=sigma, truncate=window_size / (2 * sigma)) - mu_x_sq
        sigma_y_sq = gaussian_filter(sr_2d ** 2, sigma=sigma, truncate=window_size / (2 * sigma)) - mu_y_sq
        sigma_xy = gaussian_filter(hr_2d * sr_2d, sigma=sigma, truncate=window_size / (2 * sigma)) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return float(np.mean(ssim_map))
    
    # Handle multi-channel
    if hr_np.ndim == 3:
        if channel_axis is not None:
            if channel_axis == 0:
                channels = [hr_np[i] for i in range(hr_np.shape[0])]
                sr_channels = [sr_np[i] for i in range(sr_np.shape[0])]
            elif channel_axis == -1:
                channels = [hr_np[..., i] for i in range(hr_np.shape[-1])]
                sr_channels = [sr_np[..., i] for i in range(sr_np.shape[-1])]
            else:
                raise ValueError(f"Unsupported channel_axis: {channel_axis}")
            return float(np.mean([_ssim_single(c, s) for c, s in zip(channels, sr_channels)]))
        else:
            # No channel_axis specified - average over channels
            return float(np.mean([_ssim_single(hr_np[c], sr_np[c]) for c in range(hr_np.shape[0])]))
    
    elif hr_np.ndim == 4:
        # [B, C, H, W] - average over batch and channels
        ssims = []
        for b in range(hr_np.shape[0]):
            for c in range(hr_np.shape[1]):
                ssims.append(_ssim_single(hr_np[b, c], sr_np[b, c]))
        return float(np.mean(ssims))
    
    else:
        # 2D single channel
        return _ssim_single(hr_np, sr_np)


def compute_sam(hr: ArrayLike, sr: ArrayLike, eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper (SAM) in radians.
    
    Measures spectral distortion between HR and SR multispectral images.
    Lower is better (0 = identical spectra).
    
    Args:
        hr: [C, H, W] or [H, W, C] reference
        sr: [C, H, W] or [H, W, C] super-resolved
        eps: Small epsilon for numerical stability
    
    Returns:
        Mean SAM angle in radians
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    if hr_np.ndim != 3:
        raise ValueError(f"SAM expects 3D arrays [C, H, W] or [H, W, C], got {hr_np.ndim}D")
    
    # Move channel axis to last if needed
    if hr_np.shape[0] < hr_np.shape[-1]:  # likely [C, H, W]
        hr_np = np.moveaxis(hr_np, 0, -1)
        sr_np = np.moveaxis(sr_np, 0, -1)
    
    # Normalize spectra
    hr_norm = np.linalg.norm(hr_np, axis=-1, keepdims=True)
    sr_norm = np.linalg.norm(sr_np, axis=-1, keepdims=True)
    
    hr_unit = hr_np / (hr_norm + eps)
    sr_unit = sr_np / (sr_norm + eps)
    
    # Cosine similarity -> angle
    cos_sim = np.sum(hr_unit * sr_unit, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.arccos(cos_sim)
    
    return float(np.mean(angles))


def compute_ergas(hr: ArrayLike, sr: ArrayLike, scale: int = 4) -> float:
    """
    ERGAS (Relative Global Dimensionless Synthesis Error).
    
    Args:
        hr: [C, H, W] reference
        sr: [C, H, W] super-resolved
        scale: Upscaling factor
    
    Returns:
        ERGAS value (lower is better)
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    if hr_np.ndim != 3:
        raise ValueError(f"ERGAS expects 3D arrays [C, H, W], got {hr_np.ndim}D")
    
    n_bands = hr_np.shape[0]
    ergas_sum = 0.0
    
    for b in range(n_bands):
        mse_b = np.mean((hr_np[b] - sr_np[b]) ** 2)
        mean_hr = np.mean(hr_np[b])
        if mean_hr > 0:
            ergas_sum += mse_b / (mean_hr ** 2)
    
    return float(100 / scale ** 2 * ergas_sum / n_bands)


def compute_spectral_consistency_index(
    hr: ArrayLike,
    sr: ArrayLike,
    bands: Tuple[int, int, int] = (0, 1, 2),
) -> float:
    """
    Spectral Consistency Index (SCI) - physics-informed metric for RS-SR.
    
    Measures preservation of inter-band spectral relationships.
    SCI = 1 - (spectral_distortion / max_possible_distortion)
    
    Args:
        hr: [C, H, W] reference
        sr: [C, H, W] super-resolved
        bands: Indices of bands to use for SCI (default first 3)
    
    Returns:
        SCI in [0, 1] (higher is better, 1 = perfect spectral consistency)
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    if hr_np.ndim != 3:
        raise ValueError(f"SCI expects 3D arrays [C, H, W], got {hr_np.ndim}D")
    
    n_bands = hr_np.shape[0]
    if n_bands < 3:
        raise ValueError(f"SCI requires at least 3 bands, got {n_bands}")
    
    # Select bands
    b1, b2, b3 = bands[0] % n_bands, bands[1] % n_bands, bands[2] % n_bands
    
    # Compute band ratios (spectral relationships)
    hr_ratio12 = hr_np[b1] / (hr_np[b2] + 1e-8)
    hr_ratio13 = hr_np[b1] / (hr_np[b3] + 1e-8)
    hr_ratio23 = hr_np[b2] / (hr_np[b3] + 1e-8)
    
    sr_ratio12 = sr_np[b1] / (sr_np[b2] + 1e-8)
    sr_ratio13 = sr_np[b1] / (sr_np[b3] + 1e-8)
    sr_ratio23 = sr_np[b2] / (sr_np[b3] + 1e-8)
    
    # RMSE of ratios
    def ratio_rmse(hr_r, sr_r):
        return np.sqrt(np.mean((hr_r - sr_r) ** 2))
    
    rmse12 = ratio_rmse(hr_ratio12, sr_ratio12)
    rmse13 = ratio_rmse(hr_ratio13, sr_ratio13)
    rmse23 = ratio_rmse(hr_ratio23, sr_ratio23)
    
    mean_rmse = (rmse12 + rmse13 + rmse23) / 3
    
    # Normalize by expected dynamic range
    max_expected = 10.0
    sci = 1.0 - min(mean_rmse / max_expected, 1.0)
    
    return float(np.clip(sci, 0.0, 1.0))


def compute_wavelet_hh_energy(img: ArrayLike, wavelet: str = 'db1', level: int = 1) -> float:
    """
    High-frequency (HH subband) wavelet energy.
    Higher energy in HH = more high-frequency detail.
    
    Args:
        img: [H, W] or [C, H, W] image
        wavelet: PyWavelets wavelet name
        level: Decomposition level
    
    Returns:
        Mean HH energy across channels
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets required: pip install pywt")
    
    img_np = _to_numpy(img)
    
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, ...]
    
    energies = []
    for c in range(img_np.shape[0]):
        coeffs = pywt.wavedec2(img_np[c], wavelet, level=level)
        cH, cV, cD = coeffs[1]
        hh_energy = np.mean(cD ** 2)
        energies.append(hh_energy)
    
    return float(np.mean(energies))


def compute_glcm_entropy(
    img: ArrayLike, 
    distances: Tuple[int] = (1,), 
    angles: Tuple[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4)
) -> float:
    """
    GLCM (Gray-Level Co-occurrence Matrix) Entropy.
    Measures texture complexity.
    
    Args:
        img: [H, W] single channel image
        distances: Pixel pair distances
        angles: Angles in radians
    
    Returns:
        Mean GLCM entropy
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")
    
    img_np = _to_numpy(img)
    
    if img_np.ndim > 2:
        return float(np.mean([compute_glcm_entropy(img_np[i]) for i in range(img_np.shape[0])]))
    
    # Normalize to 0-255 for GLCM
    img_norm = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
    
    glcm = graycomatrix(img_norm, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    entropy = graycoprops(glcm, 'entropy')
    return float(np.mean(entropy))


def compute_task_utility_score(
    hr_metric: float,
    sr_metric: float,
    baseline_metric: float,
    higher_is_better: bool = True,
) -> float:
    """
    Task Utility Score: How much of the HR performance does SR recover?
    
    TUS = (SR - Baseline) / (HR - Baseline) clipped to [0, 1]
    
    Args:
        hr_metric: Metric value on HR (target)
        sr_metric: Metric value on SR
        baseline_metric: Metric value on baseline (e.g., bicubic upsampling)
        higher_is_better: Whether higher metric values are better
    
    Returns:
        Task Utility Score in [0, 1] (1 = matches HR, 0 = matches baseline)
    """
    if higher_is_better:
        numerator = sr_metric - baseline_metric
        denominator = hr_metric - baseline_metric
    else:
        numerator = baseline_metric - sr_metric
        denominator = baseline_metric - hr_metric
    
    if abs(denominator) < 1e-8:
        return 0.0
    
    tus = numerator / denominator
    return float(np.clip(tus, 0.0, 1.0))


def compute_all_metrics(
    sr: ArrayLike,
    hr: ArrayLike,
    baseline: Optional[ArrayLike] = None,
    scale: int = 4,
) -> dict:
    """
    Compute all available metrics for a pair of images.
    
    Args:
        hr: High-resolution reference [H, W] or [C, H, W]
        sr: Super-resolved image (same shape)
        baseline: Optional baseline (e.g., bicubic) for TUS
        scale: Upscaling factor for ERGAS
    
    Returns:
        Dictionary of metric names to values
    """
    hr_np, sr_np = _validate_shapes(hr, sr)
    
    results = {
        'psnr': compute_psnr(hr, sr),
        'mse': compute_mse(hr, sr),
        'mae': compute_mae(hr, sr),
        'ssim': compute_ssim(hr, sr),
    }
    
    # ERGAS, SAM, SCI only for multi-band (3D)
    if hr_np.ndim == 3:
        results['ergas'] = compute_ergas(hr, sr, scale=scale)
        results['sam'] = compute_sam(hr, sr)
        results['sci'] = compute_spectral_consistency_index(hr, sr)
    else:
        results['ergas'] = None
        results['sam'] = None
        results['sci'] = None
    
    # Texture metrics
    try:
        results['wavelet_hh_energy'] = compute_wavelet_hh_energy(sr)
        results['glcm_entropy'] = compute_glcm_entropy(sr)
    except ImportError:
        results['wavelet_hh_energy'] = None
        results['glcm_entropy'] = None
    
    # Task Utility Score if baseline provided
    if baseline is not None:
        baseline_np = _to_numpy(baseline)
        if baseline_np.shape == hr_np.shape:
            for metric_name in ['psnr', 'ssim']:
                hr_val = results[metric_name]
                baseline_val = compute_all_metrics(hr, baseline).get(metric_name, 0)
                if baseline_val is not None:
                    tus = compute_task_utility_score(hr_val, hr_val, baseline_val, higher_is_better=True)
                    results[f'tus_{metric_name}'] = tus
    
    return results