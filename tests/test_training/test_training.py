"""
Unit tests for Training Pipeline.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import (
    SRDataset, TrainingConfig, get_model_config,
    CharbonnierLoss, SRLoss, SRTrainer,
    cpu_smoke_test, create_trainer
)


class TestTrainingConfig:
    """Tests for training configuration."""
    
    def test_default_config(self):
        """TrainingConfig should have defaults."""
        config = TrainingConfig()
        assert config.edsr_batch == 16
        assert config.edsr_patch == 48
        assert config.edsr_lr == 1e-4
        assert config.edsr_epochs == 300
    
    def test_get_model_config(self):
        """get_model_config should return model-specific config."""
        config = TrainingConfig()
        edsr_config = get_model_config('edsr', config)
        assert edsr_config['batch_size'] == 16
        assert edsr_config['patch_size'] == 48
        assert edsr_config['lr'] == 1e-4
        
        swinir_config = get_model_config('swinir', config)
        assert swinir_config['batch_size'] == 16
        assert swinir_config['patch_size'] == 64
        
        realesrgan_config = get_model_config('realesrgan', config)
        assert realesrgan_config['batch_size'] == 16
        assert realesrgan_config['patch_size'] == 128


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_charbonnier_loss(self):
        """CharbonnierLoss should compute."""
        loss_fn = CharbonnierLoss()
        x = torch.randn(2, 4, 32, 32)
        y = torch.randn(2, 4, 32, 32)
        loss = loss_fn(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_charbonnier_loss_identical(self):
        """CharbonnierLoss should be ~eps for identical inputs."""
        loss_fn = CharbonnierLoss()
        x = torch.randn(2, 4, 32, 32)
        loss = loss_fn(x, x)
        # Charbonnier loss for identical inputs = eps = 1e-3
        assert loss.item() < 1.1e-3
    
    def test_sr_loss_l1(self):
        """SRLoss with L1 should compute."""
        loss_fn = SRLoss(loss_type='l1')
        sr = torch.randn(2, 4, 32, 32)
        hr = torch.randn(2, 4, 32, 32)
        losses = loss_fn(sr, hr)
        assert 'total' in losses
        assert 'pixel' in losses
        assert losses['total'].item() > 0
    
    def test_sr_loss_with_lpips(self):
        """SRLoss with LPIPS should compute."""
        loss_fn = SRLoss(loss_type='l1+lpips', lpips_weight=0.1)
        sr = torch.randn(2, 4, 32, 32)
        hr = torch.randn(2, 4, 32, 32)
        losses = loss_fn(sr, hr)
        assert 'total' in losses
        # LPIPS may or may not be available
        if 'lpips' in losses:
            assert losses['lpips'].item() > 0
    
    def test_sr_loss_with_perceptual(self):
        """SRLoss with perceptual should compute."""
        from src.models.realesrgan import PerceptualLoss
        perceptual = PerceptualLoss()
        loss_fn = SRLoss(loss_type='l1', perceptual_weight=1.0, perceptual_loss=perceptual)
        sr = torch.randn(2, 3, 32, 32)  # 3 channels for VGG
        hr = torch.randn(2, 3, 32, 32)
        losses = loss_fn(sr, hr)
        assert 'total' in losses
        assert 'perceptual' in losses


class TestSRDataset:
    """Tests for SR dataset."""
    
    def test_dataset_with_temp_files(self, tmp_path):
        """SRDataset should load from directory."""
        # Create temp files
        lr_dir = tmp_path / 'lr'
        hr_dir = tmp_path / 'hr'
        lr_dir.mkdir()
        hr_dir.mkdir()
        
        for i in range(5):
            np.save(lr_dir / f'{i}.npy', np.random.rand(4, 32, 32).astype(np.float32))
            np.save(hr_dir / f'{i}.npy', np.random.rand(4, 128, 128).astype(np.float32))
        
        dataset = SRDataset(str(lr_dir), str(hr_dir), patch_size=16, scale=4, augment=True)
        assert len(dataset) == 5
        
        lr, hr = dataset[0]
        assert lr.shape == (4, 16, 16)
        assert hr.shape == (4, 64, 64)


class TestSRTrainer:
    """Tests for SR trainer."""
    
    def test_trainer_creation(self):
        """SRTrainer should create."""
        from src.models.edsr import EDSR
        model = EDSR(n_resblocks=2, n_feats=16, scale=2, in_channels=4, out_channels=4)
        
        # Create dummy data
        lr_data = torch.randn(10, 4, 32, 32)
        hr_data = torch.randn(10, 4, 64, 64)
        dataset = TensorDataset(lr_data, hr_data)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)
        
        config = {'lr': 1e-4, 'epochs': 1, 'seed': 42, 'deterministic': False, 'amp': False}
        trainer = SRTrainer(model, config, train_loader, val_loader, device='cpu', use_mlflow=False)
        assert trainer is not None
    
    def test_trainer_train_step(self):
        """Trainer should execute training step."""
        from src.models.edsr import EDSR
        model = EDSR(n_resblocks=2, n_feats=16, scale=2, in_channels=4, out_channels=4)
        
        lr_data = torch.randn(4, 4, 32, 32)
        hr_data = torch.randn(4, 4, 64, 64)
        dataset = TensorDataset(lr_data, hr_data)
        train_loader = DataLoader(dataset, batch_size=2)
        
        config = {'lr': 1e-4, 'epochs': 1, 'seed': 42, 'deterministic': False, 'amp': False,
                  'loss_type': 'l1', 'grad_clip': 1.0, 'val_interval': 1, 'checkpoint_interval': 100}
        trainer = SRTrainer(model, config, train_loader, None, device='cpu', use_mlflow=False)
        
        lr, hr = next(iter(train_loader))
        losses = trainer.train_step(lr, hr)
        assert 'total' in losses
        assert 'pixel' in losses
    
    def test_trainer_validate(self):
        """Trainer should validate."""
        from src.models.edsr import EDSR
        model = EDSR(n_resblocks=2, n_feats=16, scale=2, in_channels=4, out_channels=4)
        
        lr_data = torch.randn(4, 4, 32, 32)
        hr_data = torch.randn(4, 4, 64, 64)
        dataset = TensorDataset(lr_data, hr_data)
        val_loader = DataLoader(dataset, batch_size=2)
        
        config = {'lr': 1e-4, 'epochs': 1, 'seed': 42, 'deterministic': False, 'amp': False,
                  'loss_type': 'l1', 'grad_clip': 1.0, 'val_interval': 1, 'checkpoint_interval': 100}
        trainer = SRTrainer(model, config, None, val_loader, device='cpu', use_mlflow=False)
        
        metrics = trainer.validate()
        assert 'val_loss' in metrics
        assert 'val_psnr' in metrics
        assert 'val_ssim' in metrics


class TestCPUSmokeTest:
    """Tests for CPU smoke test."""
    
    def test_edsr_smoke(self):
        """EDSR smoke test should pass."""
        elapsed = cpu_smoke_test('edsr', None)
        assert elapsed < 30
    
    def test_swinir_smoke(self):
        """SwinIR smoke test should pass."""
        elapsed = cpu_smoke_test('swinir', None)
        assert elapsed < 30
    
    def test_realesrgan_smoke(self):
        """RealESRGAN smoke test should pass."""
        elapsed = cpu_smoke_test('realesrgan', None)
        assert elapsed < 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])