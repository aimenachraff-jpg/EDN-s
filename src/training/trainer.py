"""
Training Pipeline for Super-Resolution Models

Implements training recipes from Protocol Section 4.3:
- EDSR-small: batch=16, patch=48x48, lr=1e-4, 300 epochs, L1 loss, fp16
- SwinIR-base: batch=16, patch=64x64, lr=2e-4, 200 epochs, L1+0.1*LPIPS, fp16
- Real-ESRGAN: batch=16, patch=128x128, lr=1e-4, 400k iters, L1+1.0*Perceptual+0.1*GAN, fp16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from omegaconf import DictConfig, OmegaConf
import mlflow

from src.models.edsr import EDSR, EDSRPretrained
from src.models.swinir import SwinIR, TinySwinIR
from src.models.realesrgan import RealESRGAN, RealESRGANPretrained, PerceptualLoss, GANLoss

logger = logging.getLogger(__name__)


# =============================================================================
# DATASETS
# =============================================================================

class SRDataset(Dataset):
    """
    Super-Resolution Dataset for paired LR-HR training.
    
    Expects directory structure:
    root/
        lr/  (low resolution)
        hr/  (high resolution)
    """
    
    def __init__(self, lr_dir: str, hr_dir: str, patch_size: int = 48,
                 scale: int = 4, augment: bool = True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        
        # Find paired files
        self.lr_files = sorted(self.lr_dir.glob("*.npy"))
        self.hr_files = sorted(self.hr_dir.glob("*.npy"))
        
        # Verify pairing
        assert len(self.lr_files) == len(self.hr_files), \
            f"LR/HR count mismatch: {len(self.lr_files)} vs {len(self.hr_files)}"
        
        logger.info(f"Dataset: {len(self.lr_files)} pairs from {lr_dir}")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr = torch.from_numpy(np.load(self.lr_files[idx])).float()  # [C, H, W]
        hr = torch.from_numpy(np.load(self.hr_files[idx])).float()
        
        # Random crop
        if self.augment:
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)
        
        return lr, hr
    
    def _random_crop(self, lr, hr):
        """Random crop LR and corresponding HR patch."""
        C, h, w = lr.shape
        ph = self.patch_size
        pw = self.patch_size
        
        if h < ph or w < pw:
            # Pad if needed
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            lr = F.pad(lr, (0, pad_w, 0, pad_h), mode='reflect')
            hr = F.pad(hr, (0, pad_w*self.scale, 0, pad_h*self.scale), mode='reflect')
            h, w = lr.shape[1], lr.shape[2]
        
        top = torch.randint(0, h - ph + 1, (1,)).item()
        left = torch.randint(0, w - pw + 1, (1,)).item()
        
        lr_patch = lr[:, top:top+ph, left:left+pw]
        hr_patch = hr[:, top*self.scale:(top+ph)*self.scale, left*self.scale:(left+pw)*self.scale]
        
        return lr_patch, hr_patch
    
    def _augment(self, lr, hr):
        """Random flip and rotate."""
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
        if torch.rand(1) > 0.5:
            lr = lr.transpose(1, 2)
            hr = hr.transpose(1, 2)
        return lr, hr


# Add numpy import
import numpy as np


# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration from protocol."""
    
    # EDSR-small
    edsr_batch: int = 16
    edsr_patch: int = 48
    edsr_lr: float = 1e-4
    edsr_epochs: int = 300
    edsr_loss: str = "l1"
    edsr_amp: bool = True
    
    # SwinIR-base
    swinir_batch: int = 16
    swinir_patch: int = 64
    swinir_lr: float = 2e-4
    swinir_epochs: int = 200
    swinir_loss: str = "l1+lpips"
    swinir_lpips_weight: float = 0.1
    swinir_amp: bool = True
    
    # Real-ESRGAN
    realesrgan_batch: int = 16
    realesrgan_patch: int = 128
    realesrgan_lr: float = 1e-4
    realesrgan_iters: int = 400000
    realesrgan_loss: str = "l1+perceptual+gan"
    realesrgan_perceptual_weight: float = 1.0
    realesrgan_gan_weight: float = 0.1
    realesrgan_amp: bool = True
    
    # Common
    num_workers: int = 4
    seed: int = 42
    deterministic: bool = True
    val_interval: int = 1  # epochs
    checkpoint_interval: int = 50  # epochs
    grad_clip: float = 1.0


def get_model_config(model_name: str, config: TrainingConfig) -> Dict[str, Any]:
    """Get training config for specific model."""
    configs = {
        'edsr': {
            'batch_size': config.edsr_batch,
            'patch_size': config.edsr_patch,
            'lr': config.edsr_lr,
            'epochs': config.edsr_epochs,
            'loss_type': config.edsr_loss,
            'amp': config.edsr_amp,
        },
        'swinir': {
            'batch_size': config.swinir_batch,
            'patch_size': config.swinir_patch,
            'lr': config.swinir_lr,
            'epochs': config.swinir_epochs,
            'loss_type': config.swinir_loss,
            'lpips_weight': config.swinir_lpips_weight,
            'amp': config.swinir_amp,
        },
        'realesrgan': {
            'batch_size': config.realesrgan_batch,
            'patch_size': config.realesrgan_patch,
            'lr': config.realesrgan_lr,
            'max_iters': config.realesrgan_iters,
            'loss_type': config.realesrgan_loss,
            'perceptual_weight': config.realesrgan_perceptual_weight,
            'gan_weight': config.realesrgan_gan_weight,
            'amp': config.realesrgan_amp,
        }
    }
    return configs.get(model_name, {})


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)."""
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x - y) ** 2 + self.eps ** 2).mean()


class SRLoss(nn.Module):
    """Combined loss for SR training."""
    
    def __init__(self, loss_type: str = 'l1', lpips_weight: float = 0.0,
                 perceptual_weight: float = 0.0, gan_weight: float = 0.0,
                 perceptual_loss: Optional[nn.Module] = None,
                 gan_loss: Optional[nn.Module] = None):
        super().__init__()
        self.loss_type = loss_type
        self.lpips_weight = lpips_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.perceptual_loss = perceptual_loss
        self.gan_loss = gan_loss
        
        if 'l1' in loss_type:
            self.pixel_loss = nn.L1Loss()
        elif 'charbonnier' in loss_type:
            self.pixel_loss = CharbonnierLoss()
        else:
            self.pixel_loss = nn.MSELoss()
        
        self.lpips_fn = None
        if lpips_weight > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex')
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
            except ImportError:
                logger.warning("LPIPS not available, setting weight to 0")
                self.lpips_weight = 0
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor,
                disc_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            sr: Super-resolved output
            hr: Ground truth HR
            disc_pred: Discriminator prediction on SR (for GAN loss)
        
        Returns:
            Dict of loss components
        """
        losses = {}
        
        # Pixel loss
        losses['pixel'] = self.pixel_loss(sr, hr)
        total = losses['pixel']
        
        # LPIPS
        if self.lpips_weight > 0 and self.lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            sr_lpips = sr * 2 - 1
            hr_lpips = hr * 2 - 1
            losses['lpips'] = self.lpips_fn(sr_lpips, hr_lpips).mean()
            total = total + self.lpips_weight * losses['lpips']
        
        # Perceptual
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            losses['perceptual'] = self.perceptual_loss(sr, hr)
            total = total + self.perceptual_weight * losses['perceptual']
        
        # GAN
        if self.gan_weight > 0 and self.gan_loss is not None and disc_pred is not None:
            losses['gan'] = self.gan_loss(disc_pred, True)  # Generator wants real
            total = total + self.gan_weight * losses['gan']
        
        losses['total'] = total
        return losses


# =============================================================================
# TRAINER CLASS
# =============================================================================

class SRTrainer:
    """Generic SR Trainer with MLflow logging."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda', use_mlflow: bool = True,
                 experiment_name: str = 'sr_training'):
        
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        
        # Setup
        self._setup_reproducibility()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        self._setup_mlflow()
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_psnr = 0.0
    
    def _setup_reproducibility(self):
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_optimizer(self):
        lr = self.config.get('lr', 1e-4)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
    
    def _setup_scheduler(self):
        epochs = self.config.get('epochs', 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
    
    def _setup_loss(self):
        loss_config = self.config
        perceptual_loss = None
        gan_loss = None
        
        if loss_config.get('perceptual_weight', 0) > 0:
            perceptual_loss = PerceptualLoss().to(self.device)
        
        if loss_config.get('gan_weight', 0) > 0:
            gan_loss = GANLoss(gan_type='vanilla').to(self.device)
            # Setup discriminator
            if hasattr(self.model, 'discriminator') and self.model.discriminator is None:
                self.model.setup_training()
            self.discriminator = self.model.discriminator
            self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.config.get('lr', 1e-4), weight_decay=1e-4)
        
        self.criterion = SRLoss(
            loss_type=loss_config.get('loss_type', 'l1'),
            lpips_weight=loss_config.get('lpips_weight', 0),
            perceptual_weight=loss_config.get('perceptual_weight', 0),
            gan_weight=loss_config.get('gan_weight', 0),
            perceptual_loss=perceptual_loss,
            gan_loss=gan_loss
        ).to(self.device)
    
    def _setup_amp(self):
        self.use_amp = self.config.get('amp', False) and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
    
    def _setup_mlflow(self):
        if self.use_mlflow:
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run()
            # Log config
            mlflow.log_params(self.config)
            # Log model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            mlflow.log_param('total_params_m', total_params / 1e6)
    
    def train_step(self, lr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        lr, hr = lr.to(self.device), hr.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                sr = self.model(lr)
                
                # Discriminator forward if GAN
                disc_pred = None
                if hasattr(self, 'discriminator'):
                    with torch.no_grad():
                        disc_pred = self.discriminator(sr)
                
                losses = self.criterion(sr, hr, disc_pred)
            
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            sr = self.model(lr)
            
            disc_pred = None
            if hasattr(self, 'discriminator'):
                with torch.no_grad():
                    disc_pred = self.discriminator(sr)
            
            losses = self.criterion(sr, hr, disc_pred)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()
        
        # Discriminator step (if GAN)
        if hasattr(self, 'discriminator'):
            self._train_discriminator(lr, hr, sr.detach())
        
        self.global_step += 1
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def _train_discriminator(self, lr, hr, sr):
        """Train discriminator."""
        self.discriminator.train()
        self.disc_optimizer.zero_grad()
        
        # Real
        real_pred = self.discriminator(hr)
        real_loss = self.criterion.gan_loss(real_pred, True)
        
        # Fake
        fake_pred = self.discriminator(sr)
        fake_loss = self.criterion.gan_loss(fake_pred, False)
        
        d_loss = (real_loss + fake_loss) / 2
        
        if self.use_amp:
            with autocast():
                pass  # Already computed
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.disc_optimizer)
            self.scaler.update()
        else:
            d_loss.backward()
            self.disc_optimizer.step()
    
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []
        
        with torch.no_grad():
            for lr, hr in self.val_loader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                sr = self.model(lr)
                
                # Loss
                losses = self.criterion(sr, hr)
                val_losses.append(losses['total'].item())
                
                # PSNR/SSIM (convert to numpy)
                sr_np = sr.cpu().numpy()
                hr_np = hr.cpu().numpy()
                
                for i in range(sr_np.shape[0]):
                    from src.evaluation.metrics import psnr, ssim
                    val_psnrs.append(psnr(sr_np[i], hr_np[i]))
                    val_ssims.append(ssim(sr_np[i], hr_np[i]))
        
        return {
            'val_loss': np.mean(val_losses),
            'val_psnr': np.mean(val_psnrs),
            'val_ssim': np.mean(val_ssims)
        }
    
    def train(self, max_epochs: Optional[int] = None, max_iters: Optional[int] = None):
        """Main training loop."""
        epochs = max_epochs or self.config.get('epochs', 100)
        iters_limit = max_iters or self.config.get('max_iters', float('inf'))
        
        logger.info(f"Starting training: {epochs} epochs, max {iters_limit} iters")
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_losses = []
            
            for batch_idx, (lr, hr) in enumerate(self.train_loader):
                if self.global_step >= iters_limit:
                    break
                
                losses = self.train_step(lr, hr)
                epoch_losses.append(losses['total'])
                
                # Logging
                if self.global_step % 100 == 0:
                    self._log_metrics(losses, prefix='train')
            
            # Validation
            if epoch % self.config.get('val_interval', 1) == 0:
                val_metrics = self.validate()
                self._log_metrics(val_metrics, prefix='val')
                
                # Checkpoint
                if val_metrics.get('val_psnr', 0) > self.best_val_psnr:
                    self.best_val_psnr = val_metrics['val_psnr']
                    self.save_checkpoint('best')
            
            # Scheduler step
            self.scheduler.step()
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 50) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Log epoch summary
            avg_loss = np.mean(epoch_losses)
            self._log_metrics({'epoch_loss': avg_loss, 'lr': self.scheduler.get_last_lr()[0]}, prefix='epoch')
            
            if self.global_step >= iters_limit:
                break
        
        # Final checkpoint
        self.save_checkpoint('last')
        self._cleanup()
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        if self.use_mlflow:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}" if prefix else k, v, step=self.global_step)
        logger.info(f"Step {self.global_step}: {metrics}")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.get('checkpoint_dir', 'experiments/checkpoints'))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = ckpt_dir / f"{name}.pt"
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'config': self.config
        }, ckpt_path)
        
        if self.use_mlflow:
            mlflow.log_artifact(str(ckpt_path))
        
        logger.info(f"Saved checkpoint: {ckpt_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_psnr = ckpt['best_val_psnr']
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def _cleanup(self):
        if self.use_mlflow:
            mlflow.end_run()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_trainer(model_name: str, config: DictConfig, 
                   train_loader: DataLoader, val_loader: DataLoader,
                   device: str = 'cuda') -> SRTrainer:
    """Factory to create trainer for specific model."""
    
    # Create model
    if model_name == 'edsr':
        model = EDSR(
            n_resblocks=config.get('n_resblocks', 16),
            n_feats=config.get('n_feats', 64),
            scale=config.get('scale', 4),
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4)
        )
    elif model_name == 'swinir':
        model = SwinIR(
            upscale=config.get('upscale', 4),
            img_size=config.get('img_size', 64),
            window_size=config.get('window_size', 8),
            depths=config.get('depths', [6]*6),
            embed_dim=config.get('embed_dim', 180),
            num_heads=config.get('num_heads', [6]*6),
            mlp_ratio=config.get('mlp_ratio', 2),
            in_chans=config.get('in_chans', 4),
            out_chans=config.get('out_chans', 4)
        )
    elif model_name == 'realesrgan':
        model = RealESRGAN(
            net_g=config.get('net_g', {}),
            scale=config.get('scale', 4)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get training config
    train_config = OmegaConf.to_container(config.get('training', {}), resolve=True)
    
    return SRTrainer(model, train_config, train_loader, val_loader, device)


# =============================================================================
# CPU SMOKE TEST
# =============================================================================

def cpu_smoke_test(model_name: str, config: DictConfig):
    """Quick CPU inference test (Protocol Section 4.5)."""
    device = 'cpu'
    
    if model_name == 'edsr':
        model = EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4, out_channels=4)
    elif model_name == 'swinir':
        model = TinySwinIR()  # Use tiny for CPU test
    elif model_name == 'realesrgan':
        model = RealESRGAN(net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23}, scale=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(1, 4, 64, 64).to(device)
    
    import time
    start = time.time()
    with torch.no_grad():
        y = model(x)
    elapsed = time.time() - start
    
    print(f"{model_name} CPU inference: {elapsed:.2f}s for 64x64 -> {y.shape}")
    assert elapsed < 30, f"CPU inference too slow: {elapsed:.2f}s > 30s"
    
    return elapsed


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    print("Testing training pipeline...")
    
    # Test model creation
    edsr = EDSR()
    swinir = TinySwinIR()
    realesrgan = RealESRGAN(
        net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23},
        scale=4
    )
    
    x = torch.randn(1, 4, 64, 64)
    for name, model in [('EDSR', edsr), ('SwinIR', swinir), ('RealESRGAN', realesrgan)]:
        with torch.no_grad():
            y = model(x)
        print(f"{name}: {x.shape} -> {y.shape}")
    
    # Test loss
    loss_fn = SRLoss(loss_type='l1+lpips', lpips_weight=0.1)
    sr = torch.randn(1, 4, 256, 256)
    hr = torch.randn(1, 4, 256, 256)
    losses = loss_fn(sr, hr)
    print(f"Loss components: {losses}")
    
    print("Training pipeline test passed!")