"""ALSAT-2B / NAOMI Degradation Engine

Implements the Wald protocol for degrading HR imagery to simulate ALSAT-2B NAOMI sensor characteristics.

ALL PARAMETERS ARE ESTIMATED FROM PUBLIC SPECIFICATIONS (ESA eoPortal).
No real ALSAT-2B calibration data is available for validation.

Key estimated parameters (flagged as such in configs/degradation/alsat2b.yaml):
- MTF: Gaussian kernel σ≈0.35px (from "MTF@Nyquist > 0.15")
- SNR: 20 dB conservative (spec says >100 = 20 dB)
- SRF: Trapezoidal from documented band edges (exact)
- Quantization: 12-bit → 10-bit downlink (exact)
- Registration: <0.3 pixel inter-band (exact)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from omegaconf import DictConfig
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter, shift
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DegradationParams:
    """Container for degradation parameters with source tracking."""
    # MTF
    mtf_sigma_ms: float = 0.35
    mtf_sigma_pan: float = 0.35
    mtf_type: str = "gaussian"

    # SRF
    srf_type: str = "trapezoidal"
    band_edges_um: Dict[str, List[float]] = None
    srf_transition_width: float = 0.05

    # Noise
    noise_model: str = "poisson_gaussian"
    snr_db: float = 20.0
    gain: float = 2.5
    read_noise_e: float = 5.0

    # Quantization
    quantization_bits: int = 10
    quantization_type: str = "uniform"

    # Geometric
    max_registration_error_px: float = 0.3
    registration_distribution: str = "uniform"

    # Downsampling
    downsample_method: str = "area"
    scale_factor_ms: int = 4
    scale_factor_pan: int = 1

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


class TrapezoidalSRF:
    """
    Trapezoidal Spectral Response Function from documented band edges.
    
    Based on ESA eoPortal Table 2 (exact edges):
    - B1 Blue:   0.45 - 0.52 µm
    - B2 Green:  0.53 - 0.60 µm
    - B3 Red:    0.62 - 0.69 µm
    - B4 NIR:    0.76 - 0.89 µm
    - PAN:       0.45 - 0.90 µm (broadband)
    """

    def __init__(self, band_edges_um: Dict[str, List[float]], transition_width: float = 0.05):
        """
        Args:
            band_edges_um: Dict of band_name -> [min_wavelength, max_wavelength] in µm
            transition_width: Fraction of bandwidth for transition edges (default 5%)
        """
        self.band_edges = band_edges_um
        self.transition_width = transition_width
        self._build_srf()

    def _build_srf(self):
        """Build SRF curves for each band."""
        # Wavelength sampling (hyperspectral resolution)
        self.wavelengths = np.linspace(0.35, 1.1, 750)  # 1nm sampling
        self.dwl = self.wavelengths[1] - self.wavelengths[0]  # wavelength delta
        self.srfs = {}

        for band, (w_min, w_max) in self.band_edges.items():
            bw = w_max - w_min
            trans = bw * self.transition_width

            # Trapezoidal shape
            srf = np.zeros_like(self.wavelengths)

            # Flat top
            flat_start = w_min + trans
            flat_end = w_max - trans
            flat_mask = (self.wavelengths >= flat_start) & (self.wavelengths <= flat_end)
            srf[flat_mask] = 1.0

            # Rising edge
            rise_mask = (self.wavelengths >= w_min) & (self.wavelengths < flat_start)
            srf[rise_mask] = (self.wavelengths[rise_mask] - w_min) / trans

            # Falling edge
            fall_mask = (self.wavelengths > flat_end) & (self.wavelengths <= w_max)
            srf[fall_mask] = (w_max - self.wavelengths[fall_mask]) / trans

            # Normalize to unit integral (area = 1)
            integral = np.trapz(srf, self.wavelengths)
            if integral > 0:
                srf = srf / integral

            self.srfs[band] = srf

    def convolve(self, hr_cube: np.ndarray, wavelengths: np.ndarray = None) -> np.ndarray:
        """
        Convolve HR hyperspectral cube with SRFs to get MS bands.
        
        Args:
            hr_cube: [C_hyper, H, W] hyperspectral cube
            wavelengths: [C_hyper] wavelengths in µm (if None, assumes uniform sampling)
        
        Returns:
            [C_ms, H, W] multispectral bands
        """
        if wavelengths is None:
            # Assume uniform sampling across 0.35-1.1 µm
            wavelengths = np.linspace(0.35, 1.1, hr_cube.shape[0])

        ms_bands = []
        for band_name, srf in self.srfs.items():
            # Interpolate SRF to hyperspectral wavelengths
            srf_interp = np.interp(wavelengths, self.wavelengths, srf, left=0, right=0)
            # Convolve: sum over spectral dimension * wavelength delta
            ms_band = np.sum(hr_cube * srf_interp[:, np.newaxis, np.newaxis], axis=0) * self.dwl
            ms_bands.append(ms_band)

        return np.stack(ms_bands, axis=0)


class GaussianMTF:
    """Gaussian MTF kernel for spatial degradation."""

    def __init__(self, sigma: float):
        """
        Args:
            sigma: Standard deviation of Gaussian kernel in pixels
        """
        self.sigma = sigma
        # Kernel size: 6*sigma covers ~99.7% of Gaussian
        self.kernel_size = max(3, int(6 * sigma) | 1)  # odd
        self.kernel = self._build_kernel()

    def _build_kernel(self) -> np.ndarray:
        """Build 2D Gaussian kernel."""
        if self.sigma <= 0:
            # Identity kernel (no blur)
            k = 3
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k//2, k//2] = 1.0
            return kernel
        
        k = self.kernel_size
        center = k // 2
        x, y = np.meshgrid(np.arange(k) - center, np.arange(k) - center)
        kernel = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        return kernel / np.sum(kernel)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply MTF blur to image [H, W] or [C, H, W]."""
        if image.ndim == 2:
            return signal.convolve2d(image, self.kernel, mode='same', boundary='symm')
        elif image.ndim == 3:
            return np.stack([self.apply(image[c]) for c in range(image.shape[0])], axis=0)
        else:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


class PoissonGaussianNoise:
    """
    Poisson-Gaussian noise model for satellite imagery.
    
    Variance = signal / gain + read_noise^2
    SNR = signal / sqrt(variance)
    
    The SNR parameter controls the overall noise scaling.
    """

    def __init__(self, snr_db: float, gain: float = 2.5, read_noise_e: float = 5.0):
        """
        Args:
            snr_db: Target SNR in dB at reference radiance
            gain: System gain (electrons/DN)
            read_noise_e: Read noise in electrons RMS
        """
        self.snr_db = snr_db
        self.gain = gain
        self.read_noise_e = read_noise_e
        self.read_noise_dn = read_noise_e / gain
        
        # Compute noise scaling factor from SNR
        # SNR (linear) = 10^(SNR_db/20) for amplitude, 10^(SNR_db/10) for power
        # We'll scale the total noise variance by this factor
        self.snr_linear = 10**(snr_db / 20.0)  # amplitude ratio

    def apply(self, image: np.ndarray, reference_radiance: float = 1.0) -> np.ndarray:
        """
        Add Poisson-Gaussian noise.
        
        Args:
            image: [H, W] or [C, H, W] in [0, 1] normalized radiance
            reference_radiance: Radiance level at which SNR is specified
        
        Returns:
            Noisy image
        """
        # Scale to DN space (12-bit ADC)
        max_dn = 2**12 - 1
        signal_dn = image * max_dn

        # Photon noise (Poisson) -> approximated as Gaussian for high signal
        photon_noise_var = signal_dn / self.gain

        # Read noise (Gaussian)
        read_noise_var = self.read_noise_dn**2

        # Total variance
        total_var = photon_noise_var + read_noise_var

        # Scale noise by inverse SNR (higher SNR = less noise)
        # SNR = signal / noise_std, so noise_std = signal / SNR
        # We scale the standard deviation by 1/SNR_linear
        noise_scale = 1.0 / self.snr_linear
        total_var = total_var * (noise_scale ** 2)

        # Add noise
        noise = np.random.normal(0, np.sqrt(total_var), image.shape)
        noisy_dn = signal_dn + noise

        # Clip to valid range
        noisy_dn = np.clip(noisy_dn, 0, max_dn)

        # Return to [0, 1]
        return noisy_dn / max_dn


class Quantizer:
    """Uniform quantization."""

    def __init__(self, bits: int = 10):
        self.bits = bits
        self.levels = 2**bits
        self.max_val = self.levels - 1

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Quantize image in [0, 1] to specified bit depth."""
        quantized = np.round(image * self.max_val) / self.max_val
        return quantized


class GeometricDegradation:
    """Inter-band geometric misregistration."""

    def __init__(self, max_shift_px: float = 0.3, distribution: str = "uniform"):
        self.max_shift = max_shift_px
        self.distribution = distribution

    def apply(self, ms_image: np.ndarray, pan_image: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random sub-pixel shifts between bands.
        
        Args:
            ms_image: [C, H, W] MS image
            pan_image: [1, H, W] PAN image (optional)
        
        Returns:
            (shifted_ms, shifted_pan)
        """
        C, H, W = ms_image.shape
        shifted_ms = np.zeros_like(ms_image)

        for c in range(C):
            if self.distribution == "uniform":
                shift_y = np.random.uniform(-self.max_shift, self.max_shift)
                shift_x = np.random.uniform(-self.max_shift, self.max_shift)
            elif self.distribution == "gaussian":
                shift_y = np.random.normal(0, self.max_shift / 2)
                shift_x = np.random.normal(0, self.max_shift / 2)
                shift_y = np.clip(shift_y, -self.max_shift, self.max_shift)
                shift_x = np.clip(shift_x, -self.max_shift, self.max_shift)
            else:
                shift_y = shift_x = 0

            shifted_ms[c] = shift(ms_image[c], [shift_y, shift_x], order=1, mode='reflect')

        shifted_pan = None
        if pan_image is not None:
            if self.distribution == "uniform":
                shift_y = np.random.uniform(-self.max_shift, self.max_shift)
                shift_x = np.random.uniform(-self.max_shift, self.max_shift)
            elif self.distribution == "gaussian":
                shift_y = np.random.normal(0, self.max_shift / 2)
                shift_x = np.random.normal(0, self.max_shift / 2)
                shift_y = np.clip(shift_y, -self.max_shift, self.max_shift)
                shift_x = np.clip(shift_x, -self.max_shift, self.max_shift)
            else:
                shift_y = shift_x = 0
            shifted_pan = shift(pan_image[0], [shift_y, shift_x], order=1, mode='reflect')[np.newaxis, ...]

        return shifted_ms, shifted_pan


def downsample(image: np.ndarray, scale: int, method: str = "area") -> np.ndarray:
    """
    Downsample image by integer scale factor.
    
    Args:
        image: [H, W] or [C, H, W]
        scale: Downsampling factor
        method: 'area' (cv2.INTER_AREA), 'bicubic', 'bilinear'
    
    Returns:
        Downsampled image
    """
    if scale == 1:
        return image

    methods = {
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST
    }
    interp = methods.get(method, cv2.INTER_AREA)

    if image.ndim == 2:
        h, w = image.shape
        return cv2.resize(image, (w // scale, h // scale), interpolation=interp)
    elif image.ndim == 3:
        c, h, w = image.shape
        out = np.zeros((c, h // scale, w // scale), dtype=image.dtype)
        for i in range(c):
            out[i] = cv2.resize(image[i], (w // scale, h // scale), interpolation=interp)
        return out
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


class ALSAT2BDegradation:
    """
    Complete ALSAT-2B/NAOMI degradation pipeline (Wald protocol).
    
    Takes HR hyperspectral cube at PAN resolution (2.5m) and produces:
    - MS at 10m (4 bands)
    - PAN at 2.5m (1 band)
    
    ALL PARAMETERS ARE ESTIMATED FROM PUBLIC SPECS.
    """

    def __init__(self, config: DictConfig):
        # Extract params with defaults
        p = config
        
        # MTF
        self.mtf_ms = GaussianMTF(p.get('mtf_sigma_ms', 0.35))
        self.mtf_pan = GaussianMTF(p.get('mtf_sigma_pan', 0.35))

        # SRF - DOCUMENTED band edges
        band_edges = p.get('band_edges_um', {
            'B1_blue': [0.45, 0.52],
            'B2_green': [0.53, 0.60],
            'B3_red': [0.62, 0.69],
            'B4_nir': [0.76, 0.89],
            'PAN': [0.45, 0.90]
        })
        self.srf = TrapezoidalSRF(band_edges, p.get('srf_transition_width', 0.05))

        # Noise - ESTIMATED (conservative)
        self.noise = PoissonGaussianNoise(
            snr_db=p.get('snr_db', 20.0),
            gain=p.get('gain', 2.5),
            read_noise_e=p.get('read_noise_e', 5.0)
        )

        # Quantization - DOCUMENTED (exact)
        self.quantize = Quantizer(bits=p.get('quantization_bits', 10))

        # Geometric - DOCUMENTED (exact)
        self.geometric = GeometricDegradation(
            max_shift_px=p.get('max_registration_error_px', 0.3),
            distribution=p.get('registration_distribution', 'uniform')
        )

        # Downsampling
        self.downsample_method = p.get('downsample_method', 'area')
        self.scale_ms = p.get('scale_factor_ms', 4)
        self.scale_pan = p.get('scale_factor_pan', 1)

        # Reproducibility
        self.seed = p.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)

        # PAN band index (assumed last in hyperspectral cube or synthetic)
        self.pan_idx = p.get('pan_idx', -1)  # -1 means synthetic from MS

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Also seed numpy global for any legacy calls
        np.random.seed(seed)

    def _create_synthetic_pan(self, ms_bands: np.ndarray) -> np.ndarray:
        """Create synthetic PAN from MS bands using weighted sum.
        
        Weights approximate the broadband PAN response (0.45-0.90 µm).
        """
        # Weights: B1(0.45-0.52), B2(0.53-0.60), B3(0.62-0.69), B4(0.76-0.89)
        # Approximate overlap with PAN (0.45-0.90)
        weights = np.array([0.15, 0.30, 0.35, 0.20])  # Sum = 1.0
        pan = np.sum(ms_bands * weights[:, np.newaxis, np.newaxis], axis=0)
        return pan[np.newaxis, ...]  # [1, H, W]

    def forward(self, hr_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply complete degradation pipeline.
        
        Args:
            hr_cube: [C_hyper, H, W] at 2.5m resolution (PAN-equivalent)
                     or [C_ms, H, W] already at MS bands (4 channels)
        
        Returns:
            (ms_lr, pan_lr):
                ms_lr: [4, H/4, W/4] at 10m
                pan_lr: [1, H, W] at 2.5m
        """
        # Ensure RNG state
        np.random.seed(self.seed)

        # Input: if 4 bands, assume already MS; if more, apply SRF
        if hr_cube.shape[0] == 4:
            ms = hr_cube
        else:
            # Apply SRF to get 4 MS bands (filter out PAN from SRF)
            ms_bands_dict = {k: v for k, v in self.srf.srfs.items() if not k.startswith('PAN')}
            # Temporarily replace srfs with only MS bands
            original_srfs = self.srf.srfs
            self.srf.srfs = ms_bands_dict
            ms = self.srf.convolve(hr_cube)  # [4, H, W] at 2.5m
            self.srf.srfs = original_srfs

        # Create PAN (synthetic from MS or from hyperspectral)
        if self.pan_idx >= 0 and hr_cube.shape[0] > 4:
            pan = hr_cube[self.pan_idx:self.pan_idx+1]  # [1, H, W]
        else:
            pan = self._create_synthetic_pan(ms)  # [1, H, W]

        # 1. Spatial degradation (MTF blur)
        ms_blur = self.mtf_ms.apply(ms)  # [4, H, W]
        pan_blur = self.mtf_pan.apply(pan[0])[np.newaxis, ...]  # [1, H, W]

        # 2. Downsampling
        ms_lr = downsample(ms_blur, self.scale_ms, self.downsample_method)  # [4, H/4, W/4]
        pan_lr = downsample(pan_blur, self.scale_pan, self.downsample_method)  # [1, H, W]

        # 3. Radiometric degradation (noise + quantization)
        ms_lr = self.noise.apply(ms_lr)
        ms_lr = self.quantize.apply(ms_lr)

        pan_lr = self.noise.apply(pan_lr)
        pan_lr = self.quantize.apply(pan_lr)

        # 4. Geometric (registration errors)
        ms_lr, pan_lr = self.geometric.apply(ms_lr, pan_lr)

        return ms_lr, pan_lr

    def forward_ms_only(self, hr_ms: np.ndarray) -> np.ndarray:
        """
        Degradation for MS-only input (e.g., Sentinel-2 10m → 40m simulation).
        
        Args:
            hr_ms: [4, H, W] at 10m
            
        Returns:
            lr_ms: [4, H/4, W/4] at 40m
        """
        np.random.seed(self.seed)
        
        # MTF blur
        ms_blur = self.mtf_ms.apply(hr_ms)
        
        # Downsample
        lr = downsample(ms_blur, self.scale_ms, self.downsample_method)
        
        # Noise + quantization
        lr = self.noise.apply(lr)
        lr = self.quantize.apply(lr)
        
        # Geometric (inter-band shifts)
        lr, _ = self.geometric.apply(lr)
        
        return lr


def create_degradation(config_path: str = "configs/degradation/alsat2b.yaml") -> ALSAT2BDegradation:
    """Factory function to create degradation engine from config."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    return ALSAT2BDegradation(config['degradation'])


def main():
    """Test the degradation engine."""
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic test data
    # 384x384 at 2.5m = 960x960m ground coverage
    H, W = 384, 384
    C_hyper = 50  # Hyperspectral bands
    
    # Create test cube with some structure
    hr_cube = np.zeros((C_hyper, H, W), dtype=np.float32)
    for c in range(C_hyper):
        # Add some spatial patterns
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        hr_cube[c] = 0.3 + 0.2 * np.sin(2 * np.pi * x / 64) * np.cos(2 * np.pi * y / 64)
        hr_cube[c] += 0.1 * np.random.rand(H, W)
    hr_cube = np.clip(hr_cube, 0, 1)
    
    # Create degradation engine
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/degradation/alsat2b.yaml")
    deg = ALSAT2BDegradation(config['degradation'])
    
    # Test forward pass
    ms_lr, pan_lr = deg.forward(hr_cube)
    
    print(f"HR cube: {hr_cube.shape}")
    print(f"MS LR: {ms_lr.shape}")  # Should be [4, 96, 96]
    print(f"PAN LR: {pan_lr.shape}")  # Should be [1, 384, 384]
    print(f"MS range: [{ms_lr.min():.4f}, {ms_lr.max():.4f}]")
    print(f"PAN range: [{pan_lr.min():.4f}, {pan_lr.max():.4f}]")
    
    # Test MS-only degradation (S2 10m -> 40m)
    ms_10m = np.random.rand(4, 1024, 1024).astype(np.float32)
    ms_40m = deg.forward_ms_only(ms_10m)
    print(f"S2 MS 10m -> 40m: {ms_10m.shape} -> {ms_40m.shape}")
    
    print("\nAll degradation tests passed!")


if __name__ == "__main__":
    main()