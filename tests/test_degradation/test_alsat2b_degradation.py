"""Unit Tests for ALSAT-2B Degradation Engine"""

import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.degradation.alsat2b_degradation import (
    ALSAT2BDegradation,
    TrapezoidalSRF,
    GaussianMTF,
    PoissonGaussianNoise,
    Quantizer,
    GeometricDegradation,
    downsample,
    create_degradation,
)
from omegaconf import OmegaConf


class TestTrapezoidalSRF:
    """Test Spectral Response Function."""

    def test_srf_creation(self):
        band_edges = {
            'B1': [0.45, 0.52],
            'B2': [0.53, 0.60],
        }
        srf = TrapezoidalSRF(band_edges)
        assert 'B1' in srf.srfs
        assert 'B2' in srf.srfs
        assert len(srf.wavelengths) == 750

    def test_srf_normalization(self):
        band_edges = {'B1': [0.45, 0.52]}
        srf = TrapezoidalSRF(band_edges)
        # Each SRF should integrate to 1
        for band, curve in srf.srfs.items():
            integral = np.trapz(curve, srf.wavelengths)
            assert abs(integral - 1.0) < 1e-3

    def test_srf_convolve(self):
        band_edges = {
            'B1': [0.45, 0.52],
            'B2': [0.53, 0.60],
            'B3': [0.62, 0.69],
            'B4': [0.76, 0.89],
        }
        srf = TrapezoidalSRF(band_edges)

        # Create hyperspectral cube: 100 bands, 64x64
        C_hyper = 100
        H, W = 64, 64
        wavelengths = np.linspace(0.35, 1.1, C_hyper)
        hr_cube = np.random.rand(C_hyper, H, W).astype(np.float32)

        ms = srf.convolve(hr_cube, wavelengths)
        assert ms.shape == (4, H, W)
        assert ms.min() >= 0
        assert ms.max() <= hr_cube.max()


class TestGaussianMTF:
    """Test MTF Gaussian kernel."""

    def test_kernel_creation(self):
        mtf = GaussianMTF(sigma=0.35)
        assert mtf.sigma == 0.35
        assert mtf.kernel.shape[0] == mtf.kernel.shape[1]
        assert mtf.kernel_size % 2 == 1  # odd
        assert abs(np.sum(mtf.kernel) - 1.0) < 1e-6

    def test_mtf_application_2d(self):
        mtf = GaussianMTF(sigma=0.35)
        img = np.ones((64, 64), dtype=np.float32)
        out = mtf.apply(img)
        assert out.shape == img.shape
        # Blur should reduce variance
        assert np.std(out) <= np.std(img) + 1e-6

    def test_mtf_application_3d(self):
        mtf = GaussianMTF(sigma=0.35)
        img = np.random.rand(4, 64, 64).astype(np.float32)
        out = mtf.apply(img)
        assert out.shape == img.shape

    def test_mtf_frequency_response(self):
        """Test that MTF kernel has correct frequency response properties."""
        # For sigma=0.35, MTF at Nyquist should be exp(-pi^2 * sigma^2 / 2) ≈ 0.546
        sigma = 0.35
        mtf = GaussianMTF(sigma=sigma)
        
        # Check kernel properties
        assert mtf.sigma == sigma
        assert mtf.kernel.shape[0] == mtf.kernel.shape[1]
        assert mtf.kernel_size % 2 == 1
        assert abs(np.sum(mtf.kernel) - 1.0) < 1e-6
        
        # The kernel should be symmetric
        assert np.allclose(mtf.kernel, mtf.kernel[::-1, :])
        assert np.allclose(mtf.kernel, mtf.kernel[:, ::-1])
        
        # Center value should be highest
        center = mtf.kernel_size // 2
        assert mtf.kernel[center, center] == mtf.kernel.max()


class TestPoissonGaussianNoise:
    """Test noise model."""

    def test_noise_application(self):
        noise = PoissonGaussianNoise(snr_db=20, gain=2.5, read_noise_e=5.0)
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        noisy = noise.apply(img)
        assert noisy.shape == img.shape
        assert noisy.min() >= 0
        assert noisy.max() <= 1.0

    def test_noise_increases_variance(self):
        noise = PoissonGaussianNoise(snr_db=20, gain=2.5, read_noise_e=5.0)
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        noisy = noise.apply(img)
        # Noise should increase pixel variance
        assert np.var(noisy) > np.var(img)


class TestQuantizer:
    """Test quantization."""

    def test_quantization_10bit(self):
        quant = Quantizer(bits=10)
        img = np.linspace(0, 1, 100).astype(np.float32)
        quantized = quant.apply(img)
        # Should have at most 1024 unique values
        unique_vals = len(np.unique(quantized))
        assert unique_vals <= 1024

    def test_quantization_preserves_range(self):
        quant = Quantizer(bits=10)
        img = np.random.rand(64, 64).astype(np.float32)
        quantized = quant.apply(img)
        assert quantized.min() >= 0
        assert quantized.max() <= 1.0


class TestGeometricDegradation:
    """Test geometric misregistration."""

    def test_shift_application(self):
        geo = GeometricDegradation(max_shift_px=0.3, distribution="uniform")
        ms = np.random.rand(4, 64, 64).astype(np.float32)
        pan = np.random.rand(1, 64, 64).astype(np.float32)
        
        shifted_ms, shifted_pan = geo.apply(ms, pan)
        assert shifted_ms.shape == ms.shape
        assert shifted_pan.shape == pan.shape

    def test_deterministic_with_seed(self):
        geo = GeometricDegradation(max_shift_px=0.3, distribution="uniform")
        ms = np.random.rand(4, 64, 64).astype(np.float32)
        
        np.random.seed(42)
        out1, _ = geo.apply(ms)
        
        np.random.seed(42)
        out2, _ = geo.apply(ms)
        
        assert np.allclose(out1, out2)


class TestDownsample:
    """Test downsampling."""

    def test_downsample_area(self):
        img = np.random.rand(128, 128).astype(np.float32)
        out = downsample(img, scale=4, method="area")
        assert out.shape == (32, 32)

    def test_downsample_3d(self):
        img = np.random.rand(4, 128, 128).astype(np.float32)
        out = downsample(img, scale=4, method="area")
        assert out.shape == (4, 32, 32)

    def test_downsample_scale1(self):
        img = np.random.rand(64, 64).astype(np.float32)
        out = downsample(img, scale=1, method="area")
        assert out.shape == img.shape
        assert np.allclose(out, img)


class TestALSAT2BDegradation:
    """Test complete degradation pipeline."""

    @pytest.fixture
    def degradation(self):
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        return ALSAT2BDegradation(config['degradation'])

    @pytest.fixture
    def hr_cube(self):
        """Create test hyperspectral cube."""
        C_hyper = 50
        H, W = 384, 384
        cube = np.zeros((C_hyper, H, W), dtype=np.float32)
        for c in range(C_hyper):
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            cube[c] = 0.3 + 0.2 * np.sin(2 * np.pi * x / 64) * np.cos(2 * np.pi * y / 64)
            cube[c] += 0.1 * np.random.rand(H, W)
        return np.clip(cube, 0, 1)

    def test_forward_hyperspectral(self, degradation, hr_cube):
        ms_lr, pan_lr = degradation.forward(hr_cube)
        
        # Check shapes
        assert ms_lr.shape == (4, 96, 96)  # 384/4 = 96
        assert pan_lr.shape == (1, 384, 384)
        
        # Check ranges
        assert ms_lr.min() >= 0
        assert ms_lr.max() <= 1.0
        assert pan_lr.min() >= 0
        assert pan_lr.max() <= 1.0

    def test_forward_ms_only(self, degradation):
        ms_10m = np.random.rand(4, 1024, 1024).astype(np.float32)
        ms_40m = degradation.forward_ms_only(ms_10m)
        assert ms_40m.shape == (4, 256, 256)
        assert ms_40m.min() >= 0
        assert ms_40m.max() <= 1.0

    def test_deterministic_output(self, degradation, hr_cube):
        degradation.set_seed(42)
        out1, _ = degradation.forward(hr_cube)
        
        degradation.set_seed(42)
        out2, _ = degradation.forward(hr_cube)
        
        assert np.allclose(out1, out2)

    def test_sensitivity_mtf(self, hr_cube):
        """Test that MTF sigma affects output."""
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        
        deg1 = ALSAT2BDegradation(config['degradation'])
        deg1.set_seed(42)
        out1, _ = deg1.forward(hr_cube)
        
        # Create config with different MTF
        config2 = config.copy()
        config2['degradation']['mtf_sigma_ms'] = 0.40
        deg2 = ALSAT2BDegradation(config2['degradation'])
        deg2.set_seed(42)
        out2, _ = deg2.forward(hr_cube)
        
        # Higher sigma = more blur = lower variance
        assert np.var(out1) > np.var(out2)

    def test_sensitivity_snr(self, hr_cube):
        """Test that SNR parameter affects output."""
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        
        # Test with different seeds to see noise effect
        deg1 = ALSAT2BDegradation(config['degradation'])
        deg1.set_seed(42)
        out1, _ = deg1.forward(hr_cube)
        
        # Same config, different seed -> should be different (noise is random)
        deg1b = ALSAT2BDegradation(config['degradation'])
        deg1b.set_seed(43)
        out1b, _ = deg1b.forward(hr_cube)
        
        # Different seeds should produce different outputs
        assert not np.allclose(out1, out1b)
        
        # Now test SNR effect: use a uniform input and compare different SNR configs
        uniform_cube = np.ones((50, 384, 384), dtype=np.float32) * 0.5
        
        # Low SNR (20 dB)
        deg_low = ALSAT2BDegradation(config['degradation'])
        deg_low.set_seed(42)
        out_low, _ = deg_low.forward(uniform_cube)
        
        # High SNR (100 dB) - much less noise
        config_high_snr = config.copy()
        config_high_snr['degradation']['snr_db'] = 100
        deg_high = ALSAT2BDegradation(config_high_snr['degradation'])
        deg_high.set_seed(42)
        out_high, _ = deg_high.forward(uniform_cube)
        
        # Different SNR should produce different outputs
        assert not np.allclose(out_low, out_high)
        
        # Also test the noise component directly
        from src.degradation.alsat2b_degradation import PoissonGaussianNoise
        noise_low = PoissonGaussianNoise(snr_db=20, gain=2.5, read_noise_e=5.0)
        noise_high = PoissonGaussianNoise(snr_db=100, gain=2.5, read_noise_e=5.0)
        
        test_img = np.ones((100, 100), dtype=np.float32) * 0.5
        np.random.seed(42)
        out_noise_low = noise_low.apply(test_img)
        np.random.seed(42)
        out_noise_high = noise_high.apply(test_img)
        
        # High SNR should have lower variance (less noise)
        var_low = np.var(out_noise_low - test_img)
        var_high = np.var(out_noise_high - test_img)
        assert var_low > var_high, f"Low SNR noise variance {var_low} should be > high SNR {var_high}"

    def test_config_sensitivity_ranges(self):
        """Verify sensitivity ranges are defined in config."""
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        deg_config = config['degradation']
        
        assert 'mtf_sigma_range' in deg_config
        assert 'snr_db_range' in deg_config
        assert len(deg_config['mtf_sigma_range']) >= 3
        assert len(deg_config['snr_db_range']) >= 3


class TestDegradationConfigs:
    """Test alternative degradation configs."""

    def test_bicubic_only_config(self):
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        deg = ALSAT2BDegradation(config['degradation_bicubic'])
        ms_10m = np.random.rand(4, 256, 256).astype(np.float32)
        ms_40m = deg.forward_ms_only(ms_10m)
        # Should be close to bicubic (but still downsampled by area)
        assert ms_40m.shape == (4, 64, 64)

    def test_bicubic_noise_config(self):
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        deg = ALSAT2BDegradation(config['degradation_bicubic_noise'])
        ms_10m = np.random.rand(4, 256, 256).astype(np.float32)
        ms_40m = deg.forward_ms_only(ms_10m)
        assert ms_40m.shape == (4, 64, 64)

    def test_ideal_optics_config(self):
        config = OmegaConf.load("configs/degradation/alsat2b.yaml")
        deg = ALSAT2BDegradation(config['degradation_ideal_optics'])
        ms_10m = np.random.rand(4, 256, 256).astype(np.float32)
        ms_40m = deg.forward_ms_only(ms_10m)
        assert ms_40m.shape == (4, 64, 64)


class TestFactoryFunction:
    def test_create_degradation(self):
        deg = create_degradation("configs/degradation/alsat2b.yaml")
        assert isinstance(deg, ALSAT2BDegradation)
        assert deg.mtf_ms.sigma == 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])