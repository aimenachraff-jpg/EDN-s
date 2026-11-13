"""
Unit tests for Super-Resolution models.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from src.models.edsr import EDSR, EDSRPretrained
from src.models.swinir import SwinIR, TinySwinIR
from src.models.realesrgan import RealESRGAN, RRDBNet


class TestEDSR:
    """Tests for EDSR model."""
    
    def test_edsr_creation(self):
        """EDSR should create with default config."""
        model = EDSR()
        assert model is not None
        assert model.scale == 4
        assert model.in_channels == 4
    
    def test_edsr_forward_shape(self):
        """EDSR should produce correct output shape."""
        model = EDSR(n_resblocks=2, n_feats=16, scale=4, in_channels=4, out_channels=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 256, 256)
    
    def test_edsr_custom_scale(self):
        """EDSR should support different scales."""
        for scale in [2, 3, 4]:
            model = EDSR(n_resblocks=2, n_feats=16, scale=scale, in_channels=4, out_channels=4)
            model.eval()
            x = torch.randn(1, 4, 64, 64)
            with torch.no_grad():
                y = model(x)
            expected_h = 64 * scale
            assert y.shape == (1, 4, expected_h, expected_h)
    
    def test_edsr_parameter_count(self):
        """EDSR parameter count should match expectations."""
        model = EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4, out_channels=4)
        total_params = sum(p.numel() for p in model.parameters())
        # EDSR-small should have ~1.5M params
        assert total_params > 1_000_000
        assert total_params < 5_000_000
    
    def test_edsr_gradient_flow(self):
        """Gradients should flow through EDSR."""
        model = EDSR(n_resblocks=2, n_feats=16, scale=2, in_channels=4, out_channels=4)
        model.train()
        x = torch.randn(1, 4, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
    
    def test_edsr_pretrained_creation(self):
        """EDSRPretrained should create without path."""
        model = EDSRPretrained(n_resblocks=2, n_feats=16, scale=2, in_channels=4, out_channels=4)
        assert model is not None


class TestSwinIR:
    """Tests for SwinIR model."""
    
    def test_swinir_creation(self):
        """SwinIR should create with default config."""
        model = SwinIR()
        assert model is not None
        assert model.upscale == 4
    
    def test_swinir_forward_shape(self):
        """SwinIR should produce correct output shape."""
        model = SwinIR(upscale=4, img_size=64, window_size=8, depths=[2, 2, 2, 2, 2, 2],
                       embed_dim=60, num_heads=[2, 2, 2, 2, 2, 2], mlp_ratio=2,
                       in_chans=4, out_chans=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 256, 256)
    
    def test_tiny_swinir_creation(self):
        """TinySwinIR should create for CPU development."""
        model = TinySwinIR()
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_tiny_swinir_forward(self):
        """TinySwinIR should produce correct output shape."""
        model = TinySwinIR(upscale=4, img_size=64, in_chans=4, out_chans=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 256, 256)
    
    def test_swinir_gradient_flow(self):
        """Gradients should flow through SwinIR."""
        model = TinySwinIR(upscale=2, img_size=32, in_chans=4, out_chans=4)
        model.train()
        x = torch.randn(1, 4, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None


class TestRealESRGAN:
    """Tests for Real-ESRGAN model."""
    
    def test_realesrgan_creation(self):
        """RealESRGAN should create with config."""
        model = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2})
        assert model is not None
        assert model.scale == 4
    
    def test_realesrgan_forward_shape(self):
        """RealESRGAN should produce correct output shape."""
        model = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2})
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 256, 256)
    
    def test_rrdbnet_creation(self):
        """RRDBNet should create with config."""
        model = RRDBNet(num_in_ch=4, num_out_ch=4, num_feat=64, num_block=2, scale=4)
        assert model is not None
    
    def test_rrdbnet_forward(self):
        """RRDBNet should produce correct output shape."""
        model = RRDBNet(num_in_ch=4, num_out_ch=4, num_feat=64, num_block=2, scale=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 256, 256)
    
    def test_realesrgan_discriminator(self):
        """RealESRGAN should setup discriminator for training."""
        model = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2})
        model.setup_training()
        assert model.discriminator is not None
    
    def test_realesrgan_gradient_flow(self):
        """Gradients should flow through RealESRGAN."""
        model = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2})
        model.train()
        x = torch.randn(1, 4, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None


class TestModelComparison:
    """Tests comparing model behaviors."""
    
    def test_all_models_produce_same_output_shape(self):
        """All models should produce same output shape for same input."""
        x = torch.randn(1, 4, 64, 64)
        
        edsr = EDSR(n_resblocks=2, n_feats=16, scale=4, in_channels=4, out_channels=4)
        swinir = TinySwinIR(upscale=4, img_size=64, in_chans=4, out_chans=4)
        realesrgan = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2})
        
        for model in [edsr, swinir, realesrgan]:
            model.eval()
            with torch.no_grad():
                y = model(x)
            assert y.shape == (1, 4, 256, 256), f"{model.__class__.__name__} produced wrong shape"
    
    def test_models_have_parameters(self):
        """All models should have trainable parameters."""
        models = [
            EDSR(n_resblocks=2, n_feats=16, scale=4, in_channels=4, out_channels=4),
            TinySwinIR(upscale=4, img_size=64, in_chans=4, out_chans=4),
            RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 2}),
        ]
        for model in models:
            total_params = sum(p.numel() for p in model.parameters())
            assert total_params > 0, f"{model.__class__.__name__} has no parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])