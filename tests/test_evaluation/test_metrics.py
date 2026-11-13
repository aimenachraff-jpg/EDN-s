"""
Unit tests for evaluation metrics.
All tests use synthetic data - no external dependencies.
"""

import numpy as np
import pytest

# Import our metrics module
import sys
sys.path.insert(0, 'src')
from evaluation.metrics import (
    psnr, mse, mae, ssim, sam, ergas,
    spectral_consistency_index,
    wavelet_hh_energy, glcm_entropy,
    task_utility_score, compute_all_metrics,
)


class TestBasicMetrics:
    """Tests for basic pixel-wise metrics."""

    def test_mse_identical(self):
        """MSE should be 0 for identical images."""
        img = np.random.rand(64, 64).astype(np.float32)
        assert mse(img, img) == 0.0

    def test_mse_known_values(self):
        """MSE should match known calculation."""
        hr = np.ones((10, 10), dtype=np.float32)
        sr = np.zeros((10, 10), dtype=np.float32)
        assert mse(hr, sr) == 1.0

    def test_mae_identical(self):
        """MAE should be 0 for identical images."""
        img = np.random.rand(32, 32).astype(np.float32)
        assert mae(img, img) == 0.0

    def test_mae_known_values(self):
        """MAE should match known calculation."""
        hr = np.ones((10, 10), dtype=np.float32)
        sr = np.zeros((10, 10), dtype=np.float32)
        assert mae(hr, sr) == 1.0

    def test_psnr_identical(self):
        """PSNR should be inf for identical images."""
        img = np.random.rand(64, 64).astype(np.float32)
        assert psnr(img, img) == float('inf')

    def test_psnr_known_values(self):
        """PSNR should match formula for known MSE."""
        hr = np.ones((10, 10), dtype=np.float32)
        sr = np.zeros((10, 10), dtype=np.float32)
        # MSE = 1, max_val = 1 => PSNR = 20*log10(1) - 10*log10(1) = 0
        expected = 0.0
        assert abs(psnr(hr, sr) - expected) < 1e-6

    def test_psnr_half_max(self):
        """PSNR when MSE = 0.25 * max^2 should be 6.02 dB."""
        hr = np.ones((10, 10), dtype=np.float32)
        sr = 0.5 * np.ones((10, 10), dtype=np.float32)
        # MSE = 0.25, max_val = 1 => PSNR = -10*log10(0.25) = 6.02
        expected = 10 * np.log10(4)  # ≈ 6.02
        assert abs(psnr(hr, sr) - expected) < 0.01


class TestSSIM:
    """Tests for Structural Similarity Index."""

    def test_ssim_identical(self):
        """SSIM should be 1.0 for identical images."""
        img = np.random.rand(64, 64).astype(np.float32)
        assert abs(ssim(img, img) - 1.0) < 1e-6

    def test_ssim_different(self):
        """SSIM should be < 1 for different images."""
        hr = np.ones((64, 64), dtype=np.float32)
        sr = np.zeros((64, 64), dtype=np.float32)
        val = ssim(hr, sr)
        assert 0 <= val < 1.0

    def test_ssim_multichannel(self):
        """SSIM should work on multi-channel images."""
        hr = np.random.rand(3, 64, 64).astype(np.float32)
        sr = hr.copy()
        assert abs(ssim(hr, sr, channel_axis=0) - 1.0) < 1e-6

    def test_ssim_batch(self):
        """SSIM should work on batched images."""
        hr = np.random.rand(2, 3, 32, 32).astype(np.float32)
        sr = hr.copy()
        assert abs(ssim(hr, sr) - 1.0) < 1e-6


class TestSAM:
    """Tests for Spectral Angle Mapper."""

    def test_sam_identical(self):
        """SAM should be ~0 for identical multi-band images (allowing small numerical error)."""
        img = np.random.rand(4, 32, 32).astype(np.float32)
        # Numerical precision gives ~0.00013 rad; use reasonable tolerance
        assert sam(img, img) < 1e-3

    def test_sam_orthogonal(self):
        """SAM should be pi/2 for orthogonal spectra."""
        hr = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.float32)  # [2, 2, 2]
        sr = np.array([[[0, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.float32)
        # At each pixel, spectra are orthogonal
        val = sam(hr, sr)
        # For 2-band orthogonal: angle = pi/2 ≈ 1.57
        assert abs(val - np.pi/2) < 0.1

    def test_sam_shape_mismatch(self):
        """SAM should raise on 2D inputs."""
        hr = np.random.rand(32, 32)
        sr = np.random.rand(32, 32)
        with pytest.raises(ValueError):
            sam(hr, sr)


class TestERGAS:
    """Tests for ERGAS metric."""

    def test_ergas_identical(self):
        """ERGAS should be 0 for identical images."""
        img = np.random.rand(4, 32, 32).astype(np.float32)
        assert ergas(img, img) < 1e-6

    def test_ergas_scale_invariance(self):
        """ERGAS should scale with 1/scale^2."""
        hr = np.random.rand(3, 32, 32).astype(np.float32)
        sr = hr + 0.1 * np.random.randn(*hr.shape).astype(np.float32)

        ergas_2 = ergas(hr, sr, scale=2)
        ergas_4 = ergas(hr, sr, scale=4)
        # ERGAS(4) = ERGAS(2) * (2/4)^2 = ERGAS(2) / 4
        assert abs(ergas_4 - ergas_2 / 4) < 1e-6


class TestSCI:
    """Tests for Spectral Consistency Index."""

    def test_sci_identical(self):
        """SCI should be 1.0 for identical images."""
        img = np.random.rand(4, 32, 32).astype(np.float32)
        assert abs(spectral_consistency_index(img, img) - 1.0) < 1e-6

    def test_sci_permutation(self):
        """SCI should detect band permutations."""
        hr = np.random.rand(4, 16, 16).astype(np.float32)
        sr = hr[[1, 0, 3, 2]]  # Swap bands
        sci = spectral_consistency_index(hr, sr)
        assert sci < 1.0  # Should detect spectral inconsistency


class TestTextureMetrics:
    """Tests for texture-based metrics (require optional dependencies)."""

    def test_wavelet_hh_energy(self):
        """Wavelet HH energy should be computable."""
        img = np.random.rand(32, 32).astype(np.float32)
        energy = wavelet_hh_energy(img)
        assert isinstance(energy, float)
        assert energy >= 0

    def test_glcm_entropy(self):
        """GLCM entropy should be computable."""
        img = np.random.rand(32, 32).astype(np.float32)
        entropy = glcm_entropy(img)
        assert isinstance(entropy, float)
        assert entropy >= 0


class TestTaskUtilityScore:
    """Tests for Task Utility Score."""

    def test_tus_perfect_recovery(self):
        """TUS should be 1 when SR matches HR."""
        hr_val = 30.0  # PSNR
        sr_val = 30.0
        baseline_val = 20.0
        assert task_utility_score(hr_val, sr_val, baseline_val) == 1.0

    def test_tus_baseline_recovery(self):
        """TUS should be 0 when SR matches baseline."""
        hr_val = 30.0
        sr_val = 20.0
        baseline_val = 20.0
        assert task_utility_score(hr_val, sr_val, baseline_val) == 0.0

    def test_tus_halfway(self):
        """TUS should be 0.5 at halfway point."""
        hr_val = 30.0
        sr_val = 25.0
        baseline_val = 20.0
        assert abs(task_utility_score(hr_val, sr_val, baseline_val) - 0.5) < 1e-6

    def test_tus_lower_is_better(self):
        """TUS should work for lower-is-better metrics."""
        hr_val = 0.1  # SAM (lower better)
        sr_val = 0.2
        baseline_val = 0.5
        tus = task_utility_score(hr_val, sr_val, baseline_val, higher_is_better=False)
        # SR is closer to HR than baseline: (0.5-0.2)/(0.5-0.1) = 0.75
        assert abs(tus - 0.75) < 1e-6

    def test_tus_clipping(self):
        """TUS should clip to [0, 1]."""
        hr_val = 30.0
        sr_val = 35.0  # Better than HR (impossible but test clipping)
        baseline_val = 20.0
        assert task_utility_score(hr_val, sr_val, baseline_val) == 1.0

        sr_val = 15.0  # Worse than baseline
        assert task_utility_score(hr_val, sr_val, baseline_val) == 0.0


class TestComputeAllMetrics:
    """Tests for the convenience function."""

    def test_compute_all_2d(self):
        """compute_all_metrics should work on 2D images."""
        hr = np.random.rand(32, 32).astype(np.float32)
        sr = hr.copy()
        results = compute_all_metrics(hr, sr)

        assert 'psnr' in results
        assert 'ssim' in results
        assert 'mse' in results
        assert 'mae' in results
        assert results['psnr'] == float('inf')
        assert abs(results['ssim'] - 1.0) < 1e-6

    def test_compute_all_3d(self):
        """compute_all_metrics should work on 3D multi-band images."""
        hr = np.random.rand(4, 32, 32).astype(np.float32)
        sr = hr.copy()
        results = compute_all_metrics(hr, sr)

        assert 'sam' in results
        assert 'sci' in results
        # Numerical precision gives ~0.0001 rad; use reasonable tolerance
        assert results['sam'] < 1e-3
        assert abs(results['sci'] - 1.0) < 1e-6

    def test_compute_all_with_baseline(self):
        """compute_all_metrics should compute TUS when baseline provided."""
        hr = np.random.rand(32, 32).astype(np.float32)
        sr = hr.copy()
        baseline = hr * 0.9 + 0.1  # Slightly different

        results = compute_all_metrics(hr, sr, baseline=baseline)

        assert 'tus_psnr' in results
        assert 'tus_ssim' in results
        # SR == HR so TUS should be 1
        assert abs(results['tus_psnr'] - 1.0) < 1e-6
        assert abs(results['tus_ssim'] - 1.0) < 1e-6


class TestTensorInputs:
    """Tests that metrics work with both numpy and torch tensors."""

    def test_psnr_torch(self):
        import torch
        hr = torch.rand(32, 32)
        sr = hr.clone()
        assert psnr(hr, sr) == float('inf')

    def test_ssim_torch(self):
        import torch
        hr = torch.rand(3, 32, 32)
        sr = hr.clone()
        assert abs(ssim(hr, sr, channel_axis=0) - 1.0) < 1e-6

    def test_mixed_inputs(self):
        """Mix of numpy and torch should work."""
        import torch
        hr_np = np.random.rand(32, 32).astype(np.float32)
        sr_torch = torch.from_numpy(hr_np.copy())
        assert psnr(hr_np, sr_torch) == float('inf')


class TestShapeValidation:
    """Tests for input validation."""

    def test_shape_mismatch_raises(self):
        hr = np.random.rand(32, 32)
        sr = np.random.rand(16, 16)
        with pytest.raises(ValueError):
            psnr(hr, sr)

    def test_sam_requires_3d(self):
        hr = np.random.rand(32, 32)
        sr = np.random.rand(32, 32)
        with pytest.raises(ValueError):
            sam(hr, sr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])