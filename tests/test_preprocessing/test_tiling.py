"""
Unit tests for tiling utilities.
Tests cover all tiling scenarios including edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.preprocessing import calculate_tiles, create_weight_kernel, tile_image, untile_image


class TestCalculateTiles:
    """Tests for calculate_tiles function."""
    
    def test_calculate_tiles_2d(self):
        """Should calculate tile grid for 2D image."""
        n_h, n_w, positions = calculate_tiles((128, 128), 64, 0)
        
        assert n_h == 2
        assert n_w == 2
        assert len(positions) == 4
        assert (0, 0) in positions
        assert (1, 1) in positions
    
    def test_calculate_tiles_with_overlap(self):
        """Should calculate tiles with overlap."""
        n_h, n_w, positions = calculate_tiles((100, 100), 60, 20)

        # step=40, so ceil(100/40)=3 tiles per dimension
        assert n_h == 3
        assert n_w == 3
        assert len(positions) == 9
    
    def test_calculate_tiles_3d(self):
        """Should calculate tiles for 3D image."""
        n_h, n_w, positions = calculate_tiles((4, 128, 128), 64, 0)
        
        assert n_h == 2
        assert n_w == 2
        assert len(positions) == 4
    
    def test_calculate_tiles_small_image(self):
        """Should handle small images."""
        n_h, n_w, positions = calculate_tiles((50, 50), 64, 0)
        
        assert n_h == 1
        assert n_w == 1
        assert len(positions) == 1


class TestCreateWeightKernel:
    """Tests for create_weight_kernel function."""
    
    def test_create_weight_kernel_basic(self):
        """Should create basic weight kernel."""
        kernel = create_weight_kernel(5, sigma=1.0)
        
        assert kernel.shape == (5, 5)
        assert np.all(kernel >= 0)
        assert np.allclose(kernel.sum(), 1.0, rtol=1e-5)
    
    def test_create_weight_kernel_default_sigma(self):
        """Should use default sigma when not provided."""
        kernel = create_weight_kernel(5)
        
        assert kernel.shape == (5, 5)
        assert np.allclose(kernel.sum(), 1.0, rtol=1e-5)
    
    def test_create_weight_kernel_center_peak(self):
        """Should have peak at center."""
        kernel = create_weight_kernel(5, sigma=1.0)

        # Center should be the maximum value
        assert kernel[2, 2] == kernel.max()
    
    def test_create_weight_kernel_symmetric(self):
        """Should be symmetric."""
        kernel = create_weight_kernel(7, sigma=1.5)
        
        assert np.allclose(kernel, kernel.T)  # Symmetric around main diagonal
        assert np.allclose(kernel, np.rot90(kernel, 2))  # 180-degree symmetry
    
    def test_create_weight_kernel_sigma_zero(self):
        """Should handle sigma close to zero."""
        kernel = create_weight_kernel(3, sigma=0.01)
        
        assert kernel.shape == (3, 3)
        # All mass at center with very small sigma
        assert kernel[1, 1] > 0.9


class TestTileImageIntegration:
    """Integration tests for tile_image using the primary implementation."""
    
    def test_tile_2d_image(self):
        """Should tile 2D image correctly."""
        img = np.random.rand(128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        
        assert len(tiles) == 4
        assert len(positions) == 4
        # Original preprocessing.py adds channel dim for 2D input
        assert tiles[0].shape == (1, 64, 64)
        assert positions[0] == (0, 0)
        assert positions[1] == (0, 64)
        assert positions[2] == (64, 0)
        assert positions[3] == (64, 64)
    
    def test_tile_3d_image(self):
        """Should tile 3D image correctly."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        
        assert len(tiles) == 4
        assert tiles[0].shape == (4, 64, 64)
        assert positions[0] == (0, 0)
    
    def test_tile_with_overlap(self):
        """Should create overlapping tiles."""
        img = np.random.rand(100, 100).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=60, overlap=20)
        
        # With 100px image, 60px tiles, 20px overlap -> stride=40
        # Tiles at: y=0,40; x=0,40 -> 2x2 = 4 tiles
        assert len(tiles) == 4
        assert tiles[0].shape == (1, 60, 60)
    
    def test_untile_simple(self):
        """Should reconstruct simple tiled image."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=0)
        
        reconstructed = untile_image(tiles, positions, (128, 128), 64, overlap=0)
        np.testing.assert_allclose(reconstructed, img, atol=1e-5)
    
    def test_untile_with_overlap(self):
        """Should reconstruct with overlap blending."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=16)
        
        reconstructed = untile_image(tiles, positions, (128, 128), 64, overlap=16, blend=True)
        assert reconstructed.shape == (4, 128, 128)
        assert np.all(reconstructed >= 0) and np.all(reconstructed <= 1)
    
    def test_untile_custom_weights(self):
        """Should use custom weight kernel."""
        img = np.random.rand(4, 128, 128).astype(np.float32)
        tiles, positions = tile_image(img, tile_size=64, overlap=16)
        
        weights = create_weight_kernel(64)
        reconstructed = untile_image(tiles, positions, (128, 128), 64, 
                                   overlap=16, blend_weights=weights)
        assert reconstructed.shape == (4, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
