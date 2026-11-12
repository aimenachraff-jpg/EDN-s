"""
Evaluation Metrics for Super-Resolution

Implements all metrics defined in the experimental protocol (Section 5):
1. Reconstruction Quality: PSNR, SSIM, LPIPS
2. Spectral Fidelity: SAM, ERGAS, Band Correlation
3. Structural Fidelity: Wavelet Energy, QNR
4. Hallucination Detection: SCI, Structural Hallucination Score
5. Downstream Task Performance: Fixed-backbone protocol metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import signal
from scipy.ndimage import gaussian_filter
import pywt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =============================================================================
# 1. RECONSTRUCTION QUALITY METRICS
# =============================================================================

def psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    
    Args:
        sr: Super-resolved image [C, H, W] or [H, W]
        hr: High-resolution reference [C, H, W] or [H, W]
        data_range: Maximum possible pixel value (default 1.0 for [0,1])
    
    Returns:
        PSNR in dB (average across channels if multi-channel)
    """
    if sr.ndim == 3 and hr.ndim == 3:
        psnr_vals = []
        for c in range(sr.shape[0]):
            psnr_vals.append(peak_signal_noise_ratio(hr[c], sr[c], data_range=data_range))
        return float(np.mean(psnr_vals))
    else:
        return float(peak_signal_noise_ratio(hr, sr, data_range=data_range))


def ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = 1.0,
         channel_axis: int = 0) -> float:
    """
    Structural Similarity Index.
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
        data_range: Maximum possible pixel value
        channel_axis: Axis for channels (0 for [C,H,W])
    
    Returns:
        SSIM (average across channels if multi-channel)
    """
    if sr.ndim == 3 and hr.ndim == 3:
        ssim_vals = []
        for c in range(sr.shape[0]):
            ssim_vals.append(structural_similarity(
                hr[c], sr[c], data_range=data_range, channel_axis=None
            ))
        return float(np.mean(ssim_vals))
    else:
        return float(structural_similarity(hr, sr, data_range=data_range, channel_axis=None))


def lpips_score(sr: np.ndarray, hr: np.ndarray, net: str = 'alex',
                device: str = 'cpu') -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        sr: Super-resolved image [C, H, W] in [0, 1]
        hr: High-resolution reference [C, H, W] in [0, 1]
        net: 'alex' or 'vgg'
        device: 'cpu' or 'cuda'
    
    Returns:
        LPIPS distance (lower = more similar)
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("lpips not installed. Install with: pip install lpips")
    
    # Convert to torch tensors [1, C, H, W] in [-1, 1] range
    sr_t = torch.from_numpy(sr).unsqueeze(0).float() * 2 - 1
    hr_t = torch.from_numpy(hr).unsqueeze(0).float() * 2 - 1
    
    # Move to device
    sr_t = sr_t.to(device)
    hr_t = hr_t.to(device)
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    
    with torch.no_grad():
        dist = loss_fn(sr_t, hr_t)
    
    return float(dist.item())


# =============================================================================
# 2. SPECTRAL FIDELITY METRICS
# =============================================================================

def sam(sr: np.ndarray, hr: np.ndarray, bands: Optional[List[int]] = None) -> float:
    """
    Spectral Angle Mapper (SAM) - mean angle across all pixels.
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
        bands: List of band indices to compute (default: all)
    
    Returns:
        Mean SAM in radians
    """
    C, H, W = sr.shape
    if bands is None:
        bands = list(range(C))
    
    sr_bands = sr[bands]
    hr_bands = hr[bands]
    
    # Reshape to [N_pixels, C]
    sr_flat = sr_bands.reshape(len(bands), -1).T  # [H*W, C]
    hr_flat = hr_bands.reshape(len(bands), -1).T
    
    # Compute dot products and norms
    dot = np.sum(sr_flat * hr_flat, axis=1)
    norm_sr = np.linalg.norm(sr_flat, axis=1)
    norm_hr = np.linalg.norm(hr_flat, axis=1)
    
    # Cosine similarity
    cos_theta = dot / (norm_sr * norm_hr + 1e-10)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Spectral angle
    angles = np.arccos(cos_theta)
    
    return float(np.mean(angles))


def sam_per_band_pair(sr: np.ndarray, hr: np.ndarray) -> Dict[Tuple[int, int], float]:
    """SAM for each pair of bands."""
    C = sr.shape[0]
    results = {}
    for i in range(C):
        for j in range(i+1, C):
            sam_ij = sam(sr[[i,j]], hr[[i,j]])
            results[(i, j)] = sam_ij
    return results


def ergas(sr: np.ndarray, hr: np.ndarray, scale: int = 4) -> float:
    """
    ERGAS (Relative Global Dimensionless Synthesis Error).
    
    ERGAS = 100/scale^2 * mean( (RMSE_i / mean_i)^2 )
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
        scale: Spatial resolution ratio (HR/LR)
    
    Returns:
        ERGAS value (lower = better)
    """
    C, H, W = sr.shape
    ergas_bands = []
    
    for c in range(C):
        sr_c = sr[c]
        hr_c = hr[c]
        
        rmse = np.sqrt(np.mean((sr_c - hr_c) ** 2))
        mean_hr = np.mean(hr_c) + 1e-10
        
        ergas_bands.append((rmse / mean_hr) ** 2)
    
    return 100.0 / (scale ** 2) * np.mean(ergas_bands)


def band_correlation_deviation(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    Mean absolute deviation of inter-band correlation matrices.
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
    
    Returns:
        Mean absolute correlation difference
    """
    C = sr.shape[0]
    
    # Flatten spatial dims
    sr_flat = sr.reshape(C, -1)  # [C, N]
    hr_flat = hr.reshape(C, -1)
    
    # Correlation matrices
    corr_sr = np.corrcoef(sr_flat)
    corr_hr = np.corrcoef(hr_flat)
    
    # Mean absolute deviation (upper triangle only)
    mask = np.triu_indices(C, k=1)
    diff = np.abs(corr_sr[mask] - corr_hr[mask])
    
    return float(np.mean(diff))


# =============================================================================
# 3. SPECTRAL CONSISTENCY INDEX (SCI) - H3 METRIC
# =============================================================================

def spectral_consistency_index(sr: np.ndarray, hr: np.ndarray,
                               band_names: Optional[List[str]] = None) -> float:
    """
    Spectral Consistency Index (SCI) - Protocol H3 metric.
    
    Mean SAM across band pairs, weighted by vegetation/water sensitivity.
    Bands: B1=Blue(0), B2=Green(1), B3=Red(2), B4=NIR(3)
    
    Weights:
    - Red-NIR (2,3): vegetation sensitivity → 2.0
    - Green-NIR (1,3): vegetation sensitivity → 2.0  
    - Red-SWIR* (if available): → 2.0
    - Other pairs: → 1.0
    
    Args:
        sr: Super-resolved image [4, H, W] (B, G, R, NIR)
        hr: High-resolution reference [4, H, W]
        band_names: Optional band names for verification
    
    Returns:
        SCI value (weighted mean SAM in radians)
    """
    # Standard 4-band pairs for ALSAT-2B / Sentinel-2
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    # Weights: Red-NIR and Green-NIR are most sensitive for vegetation
    weights = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]  # (2,3)=Red-NIR, (1,3)=Green-NIR
    
    sam_vals = []
    for (i, j) in pairs:
        sam_ij = sam(sr[[i, j]], hr[[i, j]])
        sam_vals.append(sam_ij)
    
    return float(np.average(sam_vals, weights=weights))


# =============================================================================
# 4. STRUCTURAL FIDELITY METRICS
# =============================================================================

def wavelet_highfreq_energy(img: np.ndarray, wavelet: str = 'db1',
                            level: int = 2, homogeneous_threshold: float = 0.01) -> np.ndarray:
    """
    Mean |HH| (diagonal detail) energy in homogeneous regions per channel.
    
    Used for structural hallucination detection: spurious high-frequency
    energy in homogeneous regions indicates hallucinated structure.
    
    Args:
        img: Image [C, H, W] or [H, W]
        wavelet: PyWavelets wavelet name
        level: Decomposition level
        homogeneous_threshold: Local variance threshold for homogeneous mask
    
    Returns:
        Per-channel mean HH energy in homogeneous regions
    """
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    
    C, H, W = img.shape
    energies = []
    
    for c in range(C):
        # 2D DWT
        coeffs = pywt.wavedec2(img[c], wavelet, level=level)
        
        # Get HH coefficients at each level (last element of detail tuple)
        # coeffs = [cA, (cH, cV, cD)_level1, (cH, cV, cD)_level2, ...]
        # HH is the diagonal detail (cD)
        hh_energy = 0.0
        count = 0
        
        for lvl in range(1, level + 1):
            cH, cV, cD = coeffs[lvl]
            hh_energy += np.mean(cD ** 2)
            count += 1
        
        energies.append(hh_energy / count if count > 0 else 0.0)
    
    return np.array(energies)


def wavelet_homogeneous_mask(img: np.ndarray, window_size: int = 8,
                             variance_threshold: float = 0.01) -> np.ndarray:
    """
    Create mask of homogeneous regions based on local variance.
    
    Args:
        img: Image [C, H, W] - uses first channel or mean across channels
        window_size: Local window size
        variance_threshold: Threshold for homogeneous classification
    
    Returns:
        Binary mask [H, W] (1 = homogeneous)
    """
    # Use mean across channels for variance calculation
    if img.ndim == 3:
        gray = np.mean(img, axis=0)
    else:
        gray = img
    
    # Local variance via uniform filter
    mean_local = gaussian_filter(gray, sigma=window_size/2)
    mean_sq_local = gaussian_filter(gray**2, sigma=window_size/2)
    var_local = mean_sq_local - mean_local**2
    
    mask = (var_local < variance_threshold).astype(float)
    return mask


def structural_hallucination_score(sr: np.ndarray, hr: np.ndarray,
                                   wavelet: str = 'db1', level: int = 2,
                                   var_threshold: float = 0.01) -> float:
    """
    Structural Hallucination Score (Protocol H2 metric).
    
    Ratio of HH energy in homogeneous regions (SR vs HR).
    High ratio = spurious high-frequency detail in smooth areas.
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
        wavelet: Wavelet type
        level: Decomposition level
        var_threshold: Homogeneous region variance threshold
    
    Returns:
        Mean structural hallucination score across channels (1.0 = no hallucination)
    """
    # Homogeneous mask from HR (ground truth smooth regions)
    mask = wavelet_homogeneous_mask(hr, variance_threshold=var_threshold)
    
    if mask.sum() < 100:  # Not enough homogeneous pixels
        return 1.0
    
    C = sr.shape[0]
    scores = []
    
    for c in range(C):
        # Get HH coefficients at all levels
        coeffs_sr = pywt.wavedec2(sr[c], wavelet, level=level)
        coeffs_hr = pywt.wavedec2(hr[c], wavelet, level=level)
        
        total_energy_sr = 0.0
        total_energy_hr = 0.0
        weight_sum = 0.0
        
        for lvl in range(1, level + 1):
            cD_sr = coeffs_sr[lvl][2]  # HH diagonal
            cD_hr = coeffs_hr[lvl][2]
            
            # Resize mask to match coefficient size
            h, w = cD_sr.shape
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if CV2_AVAILABLE else \
                          mask[::hr.shape[0]//h, ::hr.shape[1]//w]
            
            # Energy in homogeneous regions only
            energy_sr = np.mean((cD_sr ** 2) * mask_resized)
            energy_hr = np.mean((cD_hr ** 2) * mask_resized)
            
            if energy_hr > 1e-10:
                # Ratio of energies (should be ~1.0 if no hallucination)
                ratio = energy_sr / energy_hr
                scores.append(ratio)
                weight_sum += mask_resized.sum()
        
        if scores:
            # Weighted mean across levels
            scores = np.array(scores)
            scores = scores[scores > 0]  # Remove zeros
            if len(scores) > 0:
                scores = np.mean(scores)
            else:
                scores = 1.0
        else:
            scores = 1.0
    
    return float(np.mean(scores)) if scores else 1.0


def qnr(sr: np.ndarray, hr: np.ndarray, pan: Optional[np.ndarray] = None,
        q_blocks: int = 32) -> float:
    """
    QNR (Quality with No Reference) - Pansharpening quality metric.
    Adapted for SR: uses HR as reference for spectral/spatial distortion.
    
    Args:
        sr: Super-resolved MS [C, H, W]
        hr: High-resolution MS reference [C, H, W]
        pan: PAN image [1, H, W] (optional, for spectral distortion)
        q_blocks: Block size for local analysis
    
    Returns:
        QNR index (higher = better)
    """
    # Simplified QNR for SR (no PAN)
    # Spectral distortion: SAM
    D_lambda = 1 - np.exp(-sam(sr, hr))
    
    # Spatial distortion: correlation of high-frequency components
    # Use gradient correlation
    def gradient_correlation(x, y):
        gx = np.gradient(x)
        gy = np.gradient(y)
        # Mean gradient magnitude correlation across channels
        corrs = []
        for c in range(x.shape[0]):
            gx_mag = np.sqrt(gx[0][c]**2 + gx[1][c]**2)
            gy_mag = np.sqrt(gy[0][c]**2 + gy[1][c]**2)
            corr = np.corrcoef(gx_mag.flatten(), gy_mag.flatten())[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0)
        return np.mean(corrs)
    
    D_s = 1 - gradient_correlation(sr, hr)
    
    # QNR = (1 - D_lambda) * (1 - D_s)
    return float((1 - D_lambda) * (1 - D_s))


# =============================================================================
# 5. HALLUCINATION INDEX (COMPOSITE)
# =============================================================================

def hallucination_index(sr: np.ndarray, hr: np.ndarray,
                        sci_weight: float = 0.3,
                        struct_weight: float = 0.3,
                        semantic_weight: float = 0.4,
                        task_model=None, task_metric=None) -> Dict[str, float]:
    """
    Composite Hallucination Index (Protocol Section 5.2).
    
    HALLUCINATION_INDEX = 0.3 * norm(SCI) + 0.3 * norm(Structural) + 0.4 * norm(Semantic)
    
    Args:
        sr: Super-resolved image [C, H, W]
        hr: High-resolution reference [C, H, W]
        sci_weight: Weight for Spectral Consistency Index
        struct_weight: Weight for Structural Hallucination Score
        semantic_weight: Weight for Semantic Hallucination (requires task model)
        task_model: Fixed-backbone task model (optional)
        task_metric: Task metric function (optional)
    
    Returns:
        Dict with component scores and composite index
    """
    # SCI (spectral)
    sci = spectral_consistency_index(sr, hr)
    sci_norm = min(sci / 0.17, 1.0)  # Normalize by threshold ~10° = 0.17 rad
    
    # Structural
    struct = structural_hallucination_score(sr, hr)
    struct_norm = max(0, min((struct - 1.0) / 0.15, 1.0))  # Normalize by threshold
    
    # Semantic (requires downstream task)
    if task_model is not None and task_metric is not None:
        # Semantic = task performance drop
        pred_sr = task_model(sr)
        pred_hr = task_model(hr)
        semantic = task_metric(pred_hr, hr) - task_metric(pred_sr, hr)
        semantic_norm = max(0, min(semantic / 0.15, 1.0))
    else:
        semantic = 0.0
        semantic_norm = 0.0
    
    composite = (sci_weight * sci_norm + 
                 struct_weight * struct_norm + 
                 semantic_weight * semantic_norm)
    
    return {
        'sci': sci,
        'sci_norm': sci_norm,
        'structural': struct,
        'structural_norm': struct_norm,
        'semantic': semantic,
        'semantic_norm': semantic_norm,
        'composite': composite
    }


# =============================================================================
# 6. TASK UTILITY SCORE
# =============================================================================

def task_utility_score(task_metrics_sr: Dict[str, float],
                       task_metrics_hr: Dict[str, float]) -> float:
    """
    Task Utility Score (Protocol Section 5.2).
    
    Mean of normalized task metrics (SR performance / HR performance).
    
    Args:
        task_metrics_sr: Dict of task metrics on SR output
        task_metrics_hr: Dict of task metrics on HR reference
    
    Returns:
        Mean utility ratio (1.0 = SR matches HR performance)
    """
    ratios = []
    for key in task_metrics_sr:
        if key in task_metrics_hr and task_metrics_hr[key] > 0:
            ratios.append(task_metrics_sr[key] / task_metrics_hr[key])
    
    return float(np.mean(ratios)) if ratios else 0.0


def ratio_metric(task_utility: float, hallucination_index: float) -> float:
    """
    Ratio = TASK_UTILITY / (1 + HALLUCINATION_INDEX) (Protocol H4).
    
    Args:
        task_utility: Task utility score [0, 1]
        hallucination_index: Composite hallucination index [0, 1+]
    
    Returns:
        Ratio metric (higher = better trade-off)
    """
    return task_utility / (1.0 + hallucination_index)


# =============================================================================
# 7. COMPUTATIONAL EFFICIENCY
# =============================================================================

def compute_flops_params(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 4, 64, 64)) -> Dict[str, float]:
    """
    Compute FLOPs and parameters using thop.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (N, C, H, W)
    
    Returns:
        Dict with 'flops' (G), 'params' (M)
    """
    try:
        from thop import profile
        dummy_input = torch.randn(input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {'flops_g': flops / 1e9, 'params_m': params / 1e6}
    except ImportError:
        return {'flops_g': 0.0, 'params_m': 0.0}


def benchmark_latency(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 4, 256, 256),
                      device: str = 'cpu', warmup: int = 10, runs: int = 50) -> Dict[str, float]:
    """
    Benchmark inference latency.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        warmup: Warmup iterations
        runs: Timed iterations
    
    Returns:
        Dict with mean/std latency in ms
    """
    model = model.to(device)
    model.eval()
    
    dummy = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    
    # Timed
    import time
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    times_ms = np.array(times) * 1000
    return {
        'mean_ms': float(np.mean(times_ms)),
        'std_ms': float(np.std(times_ms)),
        'min_ms': float(np.min(times_ms)),
        'max_ms': float(np.max(times_ms))
    }


# =============================================================================
# 8. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                 confidence: float = 0.95, statistic: callable = np.mean) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic.
    
    Args:
        values: Array of values
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (0.95 = 95%)
        statistic: Function to compute statistic (default: mean)
    
    Returns:
        (lower_bound, upper_bound)
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    
    boot_stats = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_stats.append(statistic(values[idx]))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_stats, 100 * alpha)
    upper = np.percentile(boot_stats, 100 * (1 - alpha))
    
    return (float(lower), float(upper))


# =============================================================================
# 9. COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_sr(sr: np.ndarray, hr: np.ndarray, pan: Optional[np.ndarray] = None,
                task_model=None, task_metric=None,
                compute_all: bool = True) -> Dict[str, Any]:
    """
    Comprehensive SR evaluation following protocol taxonomy.
    
    Args:
        sr: Super-resolved image [C, H, W] in [0, 1]
        hr: High-resolution reference [C, H, W] in [0, 1]
        pan: PAN image [1, H, W] (optional)
        task_model: Fixed-backbone task model (optional)
        task_metric: Task metric function (optional)
        compute_all: If True, compute all metrics (slow)
    
    Returns:
        Dict with all metric categories
    """
    results = {}
    
    # 1. Reconstruction Quality
    results['reconstruction'] = {
        'psnr': psnr(sr, hr),
        'ssim': ssim(sr, hr),
    }
    if LPIPS_AVAILABLE and compute_all:
        try:
            results['reconstruction']['lpips'] = lpips_score(sr, hr)
        except Exception as e:
            results['reconstruction']['lpips'] = f'error: {e}'
    
    # 2. Spectral Fidelity
    results['spectral'] = {
        'sam': sam(sr, hr),
        'ergas': ergas(sr, hr, scale=4),
        'band_correlation_dev': band_correlation_deviation(sr, hr),
        'sci': spectral_consistency_index(sr, hr),
    }
    
    # 3. Structural Fidelity
    results['structural'] = {
        'wavelet_hf_energy': wavelet_highfreq_energy(sr).tolist(),
        'qnr': qnr(sr, hr, pan) if compute_all else 'skipped',
    }
    if compute_all:
        results['structural']['structural_hallucination'] = structural_hallucination_score(sr, hr)
    
    # 4. Hallucination Detection
    hall = hallucination_index(sr, hr, task_model=task_model, task_metric=task_metric)
    results['hallucination'] = hall
    
    # 5. Downstream Task (if model provided)
    if task_model is not None and task_metric is not None:
        results['downstream'] = {
            'task_utility': task_utility_score(
                task_metric(task_model(sr), hr),
                task_metric(task_model(hr), hr)
            ),
            'ratio': ratio_metric(
                task_utility_score(
                    task_metric(task_model(sr), hr),
                    task_metric(task_model(hr), hr)
                ),
                hall['composite']
            )
        }
    
    return results


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    C, H, W = 4, 256, 256
    
    hr = np.random.rand(C, H, W).astype(np.float32)
    sr = hr + 0.05 * np.random.randn(C, H, W)
    sr = np.clip(sr, 0, 1)
    
    print("Testing evaluation metrics...")
    print(f"PSNR: {psnr(sr, hr):.2f} dB")
    print(f"SSIM: {ssim(sr, hr):.4f}")
    print(f"SAM: {sam(sr, hr):.4f} rad")
    print(f"ERGAS: {ergas(sr, hr):.4f}")
    print(f"Band corr dev: {band_correlation_deviation(sr, hr):.4f}")
    print(f"SCI: {spectral_consistency_index(sr, hr):.4f}")
    print(f"Struct hallucination: {structural_hallucination_score(sr, hr):.4f}")
    print(f"QNR: {qnr(sr, hr):.4f}")
    
    hall = hallucination_index(sr, hr)
    print(f"Hallucination Index: {hall['composite']:.4f}")
    print("All metrics computed successfully!")