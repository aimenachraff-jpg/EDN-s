"""
Unit tests for Inference Pipeline.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from src.inference import (
    create_model,
    tile_inference,
    batch_inference,
    bicubic_upscale,
    evaluate_on_pairs,
)


class TestCreateModel:
    """Tests for model creation in inference."""
    
    def test_create_edsr(self):
        """Should create EDSR model."""
        model = create_model('edsr', {'n_resblocks': 2, 'n_feats': 16, 'scale': 2, 'in_channels': 4, 'out_channels': 4})
        assert model is not None
    
    def test_create_swinir(self):
        """Should create SwinIR model."""
        model = create_model('swinir', {'upscale': 2, 'img_size': 32, 'window_size': 4, 'depths': [2,2,2,2,2,2], 'embed_dim': 60, 'num_heads': [2,2,2,2,2,2], 'mlp_ratio': 2, 'in_chans': 4, 'out_chans': 4})
        assert model is not None
    
    def test_create_realesrgan(self):
        """Should create RealESRGAN model."""
        model = create_model('realesrgan', {'net_g': {'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2}, 'scale': 2})
        assert model is not None
    
    def test_create_invalid(self):
        """Should raise on invalid model type."""
        with pytest.raises(ValueError):
            create_model('invalid_model', {})


class TestTileInference:
    """Tests for tiled inference."""
    
    def test_tile_inference_small(self):
        """Should run inference on small image."""
        model = create_model('edsr', {'n_resblocks': 2, 'n_feats': 16, 'scale': 2, 'in_channels': 4, 'out_channels': 4})
        model.eval()
        lr = np.random.rand(4, 64, 64).astype(np.float32)
        sr = tile_inference(model, lr, tile_size=32, tile_overlap=8, scale=2, device='cpu')
        assert sr.shape == (4, 128, 128)
    
    def test_tile_inference_large(self):
        """Should handle larger image with tiling."""
        model = create_model('edsr', {'n_resblocks': 2, 'n_feats': 16, 'scale': 2, 'in_channels': 4, 'out_channels': 4})
        model.eval()
        lr = np.random.rand(4, 128, 128).astype(np.float32)
        sr = tile_inference(model, lr, tile_size=32, tile_overlap=8, scale=2, device='cpu')
        assert sr.shape == (4, 256, 256)


class TestBatchInference:
    """Tests for batch inference."""
    
    def test_batch_inference(self):
        """Should run batched inference."""
        model = create_model('edsr', {'n_resblocks': 2, 'n_feats': 16, 'scale': 2, 'in_channels': 4, 'out_channels': 4})
        model.eval()
        lr_batch = np.random.rand(4, 4, 32, 32).astype(np.float32)
        sr_batch = batch_inference(model, lr_batch, device='cpu')
        assert sr_batch.shape == (4, 4, 64, 64)


class TestBicubicUpscale:
    """Tests for bicubic upscaling."""
    
    def test_bicubic_upscale(self):
        """Should upscale using bicubic."""
        lr = np.random.rand(4, 32, 32).astype(np.float32)
        sr = bicubic_upscale(lr, scale=2)
        assert sr.shape == (4, 64, 64)
    
    def test_bicubic_upscale_4x(self):
        """Should upscale 4x."""
        lr = np.random.rand(4, 32, 32).astype(np.float32)
        sr = bicubic_upscale(lr, scale=4)
        assert sr.shape == (4, 128, 128)


class TestEvaluateOnPairs:
    """Tests for pair evaluation."""
    
    def test_evaluate_on_pairs(self):
        """Should evaluate on LR-HR pairs."""
        model = create_model('edsr', {'n_resblocks': 2, 'n_feats': 16, 'scale': 2, 'in_channels': 4, 'out_channels': 4})
        model.eval()

        # Create temp files
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        lr_dir = os.path.join(tmpdir, 'lr')
        hr_dir = os.path.join(tmpdir, 'hr')
        os.makedirs(lr_dir)
        os.makedirs(hr_dir)

        lr_paths = []
        hr_paths = []
        for i in range(3):
            lr_path = os.path.join(lr_dir, f'{i}.npy')
            hr_path = os.path.join(hr_dir, f'{i}.npy')
            np.save(lr_path, np.random.rand(4, 32, 32).astype(np.float32))
            np.save(hr_path, np.random.rand(4, 64, 64).astype(np.float32))
            lr_paths.append(lr_path)
            hr_paths.append(hr_path)

        metrics = evaluate_on_pairs(model, lr_paths, hr_paths, device='cpu')
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        # Aggregated metrics are returned as dict with stats
        if isinstance(metrics['psnr'], dict):
            assert metrics['psnr']['mean'] > 0
        else:
            assert metrics['psnr'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])