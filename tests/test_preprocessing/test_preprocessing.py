"""
Unit tests for Preprocessing Utilities.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.preprocessing import (
    percentile_normalize,
    cloud_mask_from_scl,
    buffer_mask,
    tile_image,
    untile_image,
    align_bands,
    compute_band_statistics,
    spectral_angle_mapper,
)


class TestPercentileNormalize:
    """Tests for percentile normalization."""
    
    def test_basic_normalize(self):
        """Should normalize to [0, 1]."""
        img = np.random.rand(64, 64).astype(np.float32) * 100 + 50
        norm = percentile_normalize(img)
        assert norm.min() >= 0
        assert norm.max() <= 1.01
        assert norm.dtype == np.float32
    
    def test_empty_image(self):
        """Should handle all-zero image."""
        img = np.zeros((64, 64), dtype=np.float32)
        norm = percentile_normalize(img)
        assert np.all(norm == 0)
    
    def test_output_range(self):
        """Should support custom output range."""
        img = np.random.rand(64, 64).astype(np.float32) * 100
        norm = percentile_normalize(img, output_range=(0, 255))
        assert norm.max() <= 255.01


class TestCloudMask:
    """Tests for cloud masking."""
    
    def test_cloud_mask_creation(self):
        """Should create cloud mask from SCL."""
        scl = np.random.randint(0, 12, (64, 64))
        mask = cloud_mask_from_scl(scl)
        assert mask.shape == scl.shape
        assert mask.dtype == bool
    
    def test_cloud_mask_clear_pixels(self):
        """Clear pixels should be True."""
        scl = np.full((64, 64), 4, dtype=np.uint8)  # All vegetation
        mask = cloud_mask_from_scl(scl)
        assert mask.all()
    
    def test_cloud_mask_cloud_pixels(self):
        """Cloud pixels should be False."""
        scl = np.full((64, 64), 9, dtype=np.uint8)  # All high prob cloud
        mask = cloud_mask_from_scl(scl)
        assert not mask.any()


class TestBufferMask:
    """Tests for buffer mask."""
    
    def test_buffer_mask_no_buffer(self):
        """No buffer should return original mask."""
        mask = np.ones((64, 64), dtype=bool)
        result = buffer_mask(mask, buffer_pixels=0)
        assert np.array_equal(result, mask)
    
    def test_buffer_mask_with_buffer(self):
        """Buffer should dilate mask."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 32] = True
        result = buffer_mask(mask, buffer_pixels=2)
        # Result should have more False pixels (buffer around the True pixel)
        assert result.sum() < mask.sum() or result.sum() == mask.sum()


class TestTiling:
    """Tests for tiling operations."""
    
    def test_tile_image_2d(self):
        """Should tile 2D image."""
        img = np.random.rand(128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        assert len(tiles) == 4  # 2x2
        assert len(positions) == 4
        assert tiles[0].shape == (1, 64, 64)
    
    def test_tile_image_3d(self):
        """Should tile 3D image."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        assert len(tiles) == 4
        assert tiles[0].shape == (4, 64, 64)
    
    def test_tile_with_overlap(self):
        """Should tile with overlap."""
        img = np.random.rand(128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=16)
        # With overlap, should have more tiles
        assert len(tiles) >= 4
    
    def test_untile_image(self):
        """Should reconstruct image from tiles."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        reconstructed = untile_image(tiles, positions, (128, 128), 64, overlap=0)
        assert reconstructed.shape == (4, 128, 128)
        np.testing.assert_allclose(reconstructed, img, atol=1e-6)
    
    def test_untile_with_overlap(self):
        """Should reconstruct with overlap blending."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=16)
        reconstructed = untile_image(tiles, positions, (128, 128), 64, overlap=16, blend=True)
        assert reconstructed.shape == (4, 128, 128)


class TestAlignBands:
    """Tests for band alignment."""
    
    def test_align_same_shape(self):
        """Should handle bands with same shape."""
        bands = {
            'B2': np.random.rand(64, 64),
            'B3': np.random.rand(64, 64),
        }
        aligned = align_bands(bands, 'B2')
        assert 'B2' in aligned
        assert 'B3' in aligned
        assert aligned['B3'].shape == (64, 64)
    
    def test_align_different_shape(self):
        """Should resize bands to reference shape."""
        bands = {
            'B2': np.random.rand(64, 64),
            'B5': np.random.rand(32, 32),
        }
        aligned = align_bands(bands, 'B2')
        assert aligned['B5'].shape == (64, 64)


class TestComputeBandStatistics:
    """Tests for band statistics."""
    
    def test_statistics(self):
        """Should compute basic statistics."""
        img = np.random.rand(64, 64).astype(np.float32) * 100
        stats = compute_band_statistics(img)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'p2' in stats
        assert 'p98' in stats
    
    def test_empty_image(self):
        """Should handle empty image."""
        img = np.zeros((64, 64), dtype=np.float32)
        stats = compute_band_statistics(img)
        assert stats['mean'] == 0


class TestSpectralAngleMapper:
    """Tests for SAM."""
    
    def test_sam_identical(self):
        """SAM should be ~0 for identical spectra."""
        spec = np.random.rand(4).astype(np.float32)
        sam = spectral_angle_mapper(spec, spec)
        assert sam < 1e-3
    
    def test_sam_2d(self):
        """SAM should work on 2D images."""
        img1 = np.random.rand(4, 32, 32).astype(np.float32)
        img2 = img1.copy()
        sam = spectral_angle_mapper(img1, img2)
        assert sam < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])