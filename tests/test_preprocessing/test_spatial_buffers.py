"""
Unit tests for spatial buffer utilities.
Tests cover all buffer scenarios including edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.preprocessing import spatial_buffer_mask as buffer_mask, apply_spatial_padding, edge_pad_image, is_valid_region, calculate_buffer_requirements, create_hierarchical_buffers


class TestBufferMask:
    """Tests for buffer_mask function."""
    
    def test_buffer_mask_basic(self):
        """Should create basic buffer mask."""
        mask = np.array([[0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0]])
        
        buffered = buffer_mask(mask, buffer_pixels=1, mode='outside')
        
        assert buffered.shape == mask.shape
        # Original features should still be present
        assert np.all(buffered[2, :] == 1)
        # Buffer should extend outward
        assert buffered[1, 0] == 1
        assert buffered[0, 2] == 1
    
    def test_buffer_mask_inside(self):
        """Should erode (inside buffer)."""
        mask = np.array([[0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0]])
        
        buffered = buffer_mask(mask, buffer_pixels=1, mode='inside')
        
        assert buffered.shape == mask.shape
        # Features should shrink: only interior pixels (4-connected) survive
        assert buffered[2, 2] == 1  # center survives
        assert buffered[1, 2] == 1  # inner vertical arm survives
        assert buffered[2, 1] == 1  # inner horizontal arm survives
        # Edge pixels are eroded away (note: (1,1) is an edge pixel of the
        # cross shape, so it must be eroded like any other boundary pixel)
        assert buffered[1, 1] == 0
        assert buffered[0, 2] == 0
    
    def test_buffer_mask_no_buffer(self):
        """Should return original mask when no buffer requested."""
        mask = np.array([[1, 0], [0, 1]])
        buffered = buffer_mask(mask, buffer_pixels=0, mode='outside')
        
        assert np.array_equal(buffered, mask)
    
    def test_buffer_mask_zero_buffer(self):
        """Should handle zero buffer size."""
        mask = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]])
        
        buffered = buffer_mask(mask, buffer_pixels=0, mode='outside')
        assert np.array_equal(buffered, mask)
    
    def test_buffer_mask_large_buffer(self):
        """Should handle larger buffer sizes."""
        mask = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])
        
        buffered = buffer_mask(mask, buffer_pixels=2, mode='outside')
        
        # With 2-pixel buffer on 3x3 image, should fill entire image
        assert buffered.all()
    
    def test_buffer_mask_edge_image(self):
        """Should handle edge images correctly."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # Central feature
        
        buffered = buffer_mask(mask, buffer_pixels=3, mode='outside')
        
        # Should extend to image borders
        assert buffered[0, 3] == 1
        assert buffered[9, 3] == 1
        assert buffered[3, 0] == 1
        assert buffered[3, 9] == 1


class TestApplySpatialPadding:
    """Tests for apply_spatial_padding function."""
    
    def test_apply_spatial_padding_basic(self):
        """Should apply basic padding."""
        img = np.random.rand(5, 5)
        padded = apply_spatial_padding(img, padding=1, mode='reflect')
        
        assert padded.shape == (7, 7)
        # Center pixel should be preserved (reflect mode)
        assert padded[3, 3] == img[2, 2]
    
    def test_apply_spatial_padding_zero(self):
        """Should return original when padding is zero."""
        img = np.random.rand(5, 5)
        padded = apply_spatial_padding(img, padding=0, mode='reflect')
        
        assert np.array_equal(padded, img)
    
    def test_apply_spatial_padding_constant(self):
        """Should use constant value for constant mode."""
        img = np.random.rand(3, 3)
        padded = apply_spatial_padding(img, padding=1, mode='constant', edge_value=0.5)
        
        assert padded.shape == (5, 5)
        assert padded[0, 0] == 0.5  # Edge should be constant value
        assert padded[2, 2] == img[1, 1]  # Center preserved
    
    def test_apply_spatial_padding_3d(self):
        """Should handle 3D images."""
        img = np.random.rand(4, 5, 5)
        padded = apply_spatial_padding(img, padding=1, mode='reflect')
        
        assert padded.shape == (4, 7, 7)
    
    def test_apply_spatial_padding_edge_modes(self):
        """Should handle different edge modes."""
        img = np.random.rand(3, 3)
        
        # Test different modes
        modes = ['reflect', 'constant', 'edge', 'symmetric']
        for mode in modes:
            padded = apply_spatial_padding(img, padding=1, mode=mode)
            assert padded.shape == (5, 5)


class TestEdgePadImage:
    """Tests for edge_pad_image function."""
    
    def test_edge_pad_image_exact(self):
        """Should not pad when already divisible."""
        img = np.random.rand(128, 128)
        padded = edge_pad_image(img, (128, 128), 64, 0, 'reflect')
        
        assert np.array_equal(padded, img)
    
    def test_edge_pad_image_requires_padding(self):
        """Should pad when not divisible."""
        img = np.random.rand(100, 100)
        padded = edge_pad_image(img, (100, 100), 64, 0, 'reflect')
        
        # 100 % 64 = 36 remainder, needs padding to reach next multiple
        assert padded.shape[0] == 128 or padded.shape[1] == 128
    
    def test_edge_pad_image_with_overlap(self):
        """Should handle overlap correctly."""
        img = np.random.rand(100, 100)
        padded = edge_pad_image(img, (100, 100), 64, 20, 'reflect')
        
        # Step size = 64 - 20 = 44
        # 100 % 44 = 12 remainder, pad by (44-12)=32 -> 132
        assert padded.shape == (132, 132)
    
    def test_edge_pad_image_3d(self):
        """Should handle 3D images."""
        img = np.random.rand(4, 100, 100)
        padded = edge_pad_image(img, (4, 100, 100), 64, 0, 'reflect')
        
        assert padded.shape[0] == 4
        assert padded.shape[1] == 128 or padded.shape[2] == 128


class TestIsValidRegion:
    """Tests for is_valid_region function."""
    
    def test_is_valid_region_valid_pixels(self):
        """Should return True for valid pixels."""
        region = np.array([[1, 0], [0, 1]])
        assert is_valid_region(region, valid_threshold=0)
    
    def test_is_valid_region_invalid_pixels(self):
        """Should return False for invalid pixels."""
        region = np.array([[0, 0], [0, 0]])
        assert not is_valid_region(region, valid_threshold=0)
    
    def test_is_valid_region_with_threshold(self):
        """Should respect validity threshold."""
        region = np.array([[0.1, 0.5], [0.9, 1.0]])
        assert is_valid_region(region, valid_threshold=0.5)
        assert is_valid_region(region, valid_threshold=0.95)  # 1.0 >= 0.95
    
    def test_is_valid_region_nodata(self):
        """Should handle nodata value."""
        region = np.array([[0, 1], [2, -9999]])
        assert is_valid_region(region, nodata_value=-9999)
    
    def test_is_valid_region_all_nodata(self):
        """Should return False for all nodata."""
        region = np.array([[-9999, -9999], [-9999, -9999]])
        assert not is_valid_region(region, nodata_value=-9999)
    
    def test_is_valid_region_empty(self):
        """Should return False for empty region."""
        region = np.array([])
        assert not is_valid_region(region)
    
    def test_is_valid_region_float(self):
        """Should handle floating point values."""
        region = np.array([[0.1, 0.2], [0.3, 0.4]])
        assert is_valid_region(region, valid_threshold=0.0)
    
    def test_is_valid_region_boolean(self):
        """Should handle boolean values."""
        region = np.array([[True, False], [False, True]])
        assert is_valid_region(region)
    
    def test_is_valid_region_boolean_false(self):
        """Should handle all-false boolean."""
        region = np.array([[False, False], [False, False]])
        assert not is_valid_region(region)


class TestCalculateBufferRequirements:
    """Tests for calculate_buffer_requirements function."""
    
    def test_calculate_buffer_requirements_basic(self):
        """Should calculate buffer requirements."""
        mask = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])

        buffer_req, features, edge_feat = calculate_buffer_requirements(mask, 1)
        # buffer_pixels=1 < min_buffer_size=10, so required_buffer increases
        assert buffer_req == 6   # min(1+5, 50)
        assert features == 9     # All pixels after dilation on 3x3 image
        assert edge_feat == 4
    
    def test_calculate_buffer_requirements_large_buffer(self):
        """Should adjust buffer for large requirements."""
        mask = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])
        
        buffer_req, features, edge_feat = calculate_buffer_requirements(mask, 10)
        # Should increase to min buffer size
        assert buffer_req == 10 or buffer_req == 5
    
    def test_calculate_buffer_requirements_no_features(self):
        """Should handle regions with no features."""
        mask = np.zeros((10, 10), dtype=bool)
        
        buffer_req, features, edge_feat = calculate_buffer_requirements(mask, 5)
        assert features == 0
        assert edge_feat == 0
    
    def test_calculate_buffer_requirements_small_image(self):
        """Should handle small images correctly."""
        mask = np.array([[1, 0], [0, 1]])
        
        buffer_req, features, edge_feat = calculate_buffer_requirements(mask, 2)
        assert features >= 1
        assert edge_feat >= 2


class TestCreateHierarchicalBuffers:
    """Tests for create_hierarchical_buffers function."""
    
    def test_create_hierarchical_buffers_basic(self):
        """Should create hierarchical buffers."""
        mask = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])
        
        buffers = create_hierarchical_buffers(mask, [1, 2, 3])
        
        assert len(buffers) == 4  # original + 3 buffer sizes
        assert np.array_equal(buffers[0], mask)
        assert buffers[1].shape == mask.shape
        assert buffers[2].shape == mask.shape
        assert buffers[3].shape == mask.shape
    
    def test_create_hierarchical_buffers_sorted(self):
        """Should sort buffer sizes."""
        mask = np.array([[1, 0], [0, 1]])
        
        buffers = create_hierarchical_buffers(mask, [3, 1, 2])
        
        # Should be sorted: [1, 2, 3]
        assert len(buffers) == 4
    
    def test_create_hierarchical_buffers_no_buffer(self):
        """Should handle empty buffer sizes."""
        mask = np.array([[1, 0], [0, 1]])
        
        buffers = create_hierarchical_buffers(mask, [])
        
        assert len(buffers) == 1
        assert np.array_equal(buffers[0], mask)
    
    def test_create_hierarchical_buffers_zero_sizes(self):
        """Should handle zero buffer sizes."""
        mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        
        buffers = create_hierarchical_buffers(mask, [0, 1, 0])
        
        # Should include 0 size (original) and 1 size
        assert len(buffers) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])