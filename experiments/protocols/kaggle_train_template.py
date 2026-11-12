"""
Kaggle Training Template for Super-Resolution Models

This script is designed to run on Kaggle with 2x T4 GPUs (30h/week free).
Supports: EDSR-small, SwinIR-base, Real-ESRGAN

Usage on Kaggle:
1. Create new notebook (GPU T4 x2)
2. Add dataset: Sentinel-2 processed tiles (COG or NPY format)
3. Copy this script, adjust config paths
4. Run

Environment setup:
- Python 3.10+
- CUDA 11.8+
- Install requirements-cloud.txt
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# =============================================================================
# SETUP (Run once at notebook start)
# =============================================================================

def setup_kaggle_environment():
    """Install requirements and configure environment."""
    # Install cloud requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "-r", "/kaggle/working/requirements-cloud.txt"], check=True)
    
    # Install local package
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", 
                    "/kaggle/working/MASTER2-A1"], check=True)
    
    # Set seeds
    import torch
    import numpy as np
    import random
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return device


# =============================================================================
# DATA LOADING
# =============================================================================

def create_dataloaders(config, data_root):
    """Create train/val dataloaders from processed Sentinel-2 data."""
    from torch.utils.data import DataLoader
    from src.training.trainer import SRDataset
    
    train_lr = Path(data_root) / "train" / "lr"
    train_hr = Path(data_root) / "train" / "hr"
    val_lr = Path(data_root) / "val" / "lr"
    val_hr = Path(data_root) / "val" / "hr"
    
    train_dataset = SRDataset(
        str(train_lr), str(train_hr),
        patch_size=config['patch_size'],
        scale=config['scale'],
        augment=True
    )
    
    val_dataset = SRDataset(
        str(val_lr), str(val_hr),
        patch_size=config['patch_size'],
        scale=config['scale'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(model_name, config):
    """Create model from config."""
    if model_name == 'edsr':
        from src.models.edsr import EDSR
        return EDSR(
            n_resblocks=config.get('n_resblocks', 16),
            n_feats=config.get('n_feats', 64),
            scale=config.get('scale', 4),
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4)
        )
    elif model_name == 'swinir':
        from src.models.swinir import SwinIR
        return SwinIR(
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
        from src.models.realesrgan import RealESRGAN
        return RealESRGAN(
            net_g=config.get('net_g', {
                'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23
            }),
            scale=config.get('scale', 4)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(model_name, config, train_loader, val_loader, device):
    """Main training loop with MLflow logging."""
    import mlflow
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.cuda.amp import GradScaler, autocast
    from src.training.trainer import SR_Loss, PerceptualLoss, GANLoss
    from src.evaluation.metrics import psnr, ssim
    import numpy as np
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:/kaggle/working/mlruns")
    mlflow.set_experiment(f"sr_{model_name}")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log config
        mlflow.log_params(config)
        
        # Create model
        model = create_model(model_name, config).to(device)
        if device == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param('total_params_m', total_params / 1e6)
        print(f"Model parameters: {total_params / 1e6:.2f}M")
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        # Scheduler
        if 'epochs' in config:
            scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
            max_epochs = config['epochs']
            max_iters = float('inf')
        else:
            scheduler = None
            max_epochs = 1000
            max_iters = config.get('max_iters', 400000)
        
        # Loss
        perceptual_loss = None
        gan_loss = None
        discriminator = None
        disc_optimizer = None
        
        if config.get('perceptual_weight', 0) > 0:
            perceptual_loss = PerceptualLoss().to(device)
        
        if config.get('gan_weight', 0) > 0:
            gan_loss = GANLoss(gan_type='vanilla').to(device)
            if hasattr(model, 'discriminator') and model.discriminator is None:
                model.setup_training()
            discriminator = model.discriminator
            disc_optimizer = AdamW(discriminator.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        criterion = SR_Loss(
            loss_type=config.get('loss_type', 'l1'),
            lpips_weight=config.get('lpips_weight', 0),
            perceptual_weight=config.get('perceptual_weight', 0),
            gan_weight=config.get('gan_weight', 0),
            perceptual_loss=perceptual_loss,
            gan_loss=gan_loss
        ).to(device)
        
        # AMP
        use_amp = config.get('amp', True) and device == 'cuda'
        scaler = GradScaler() if use_amp else None
        
        # Training loop
        global_step = 0
        best_val_psnr = 0.0
        
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            
            for batch_idx, (lr, hr) in enumerate(train_loader):
                if global_step >= max_iters:
                    break
                
                lr, hr = lr.to(device), hr.to(device)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with autocast():
                        sr = model(lr)
                        
                        disc_pred = None
                        if discriminator is not None:
                            with torch.no_grad():
                                disc_pred = discriminator(sr)
                        
                        losses = criterion(sr, hr, disc_pred)
                    
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    sr = model(lr)
                    
                    disc_pred = None
                    if discriminator is not None:
                        with torch.no_grad():
                            disc_pred = discriminator(sr)
                    
                    losses = criterion(sr, hr, disc_pred)
                    losses['total'].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                    optimizer.step()
                
                # Discriminator step
                if discriminator is not None:
                    discriminator.train()
                    disc_optimizer.zero_grad()
                    
                    real_pred = discriminator(hr)
                    fake_pred = discriminator(sr.detach())
                    
                    real_loss = gan_loss(real_pred, True)
                    fake_loss = gan_loss(fake_pred, False)
                    d_loss = (real_loss + fake_loss) / 2
                    
                    if use_amp:
                        with autocast():
                            pass
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                
                global_step += 1
                epoch_losses.append(losses['total'].item())
                
                # Logging
                if global_step % 100 == 0:
                    log_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                               for k, v in losses.items()}
                    log_dict['lr'] = scheduler.get_last_lr()[0] if scheduler else config['lr']
                    mlflow.log_metrics(log_dict, step=global_step)
                
                # Validation
                if global_step % config.get('val_interval', 1000) == 0:
                    val_metrics = validate(model, val_loader, device, criterion)
                    mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()}, 
                                      step=global_step)
                    
                    if val_metrics.get('psnr', 0) > best_val_psnr:
                        best_val_psnr = val_metrics['psnr']
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                       best_val_psnr, config, run_id, 'best')
            
            # Epoch end
            if scheduler:
                scheduler.step()
            
            avg_loss = np.mean(epoch_losses)
            mlflow.log_metric('epoch_loss', avg_loss, step=global_step)
            mlflow.log_metric('epoch_lr', scheduler.get_last_lr()[0] if scheduler else config['lr'], 
                             step=global_step)
            
            print(f"Epoch {epoch+1}/{max_epochs}: loss={avg_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0] if scheduler else config['lr']:.2e}")
            
            # Periodic checkpoint
            if (epoch + 1) % config.get('checkpoint_interval', 50) == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                               best_val_psnr, config, run_id, f'epoch_{epoch+1}')
            
            if global_step >= max_iters:
                break
        
        # Final checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                       best_val_psnr, config, run_id, 'last')
        
        print(f"Training complete. Best val PSNR: {best_val_psnr:.2f} dB")
        print(f"Run ID: {run_id}")


def validate(model, val_loader, device, criterion):
    """Validation loop."""
    import torch
    from src.evaluation.metrics import psnr, ssim
    import numpy as np
    
    model.eval()
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            losses = criterion(sr, hr)
            val_losses.append(losses['total'].item())
            
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            
            for i in range(sr_np.shape[0]):
                val_psnrs.append(psnr(sr_np[i], hr_np[i]))
                val_ssims.append(ssim(sr_np[i], hr_np[i]))
    
    return {
        'loss': np.mean(val_losses),
        'psnr': np.mean(val_psnrs),
        'ssim': np.mean(val_ssims)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                   best_val_psnr, config, run_id, name):
    """Save checkpoint to Kaggle working directory."""
    import torch
    from pathlib import Path
    
    ckpt_dir = Path(f"/kaggle/working/checkpoints/{run_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    ckpt_path = ckpt_dir / f"{name}.pt"
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_psnr': best_val_psnr,
        'config': config,
        'run_id': run_id
    }, ckpt_path)
    
    print(f"Saved checkpoint: {ckpt_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This runs when executed as a script on Kaggle
    
    # Config (adjust for each model)
    MODEL_NAME = "edsr"  # or "swinir", "realesrgan"
    
    CONFIGS = {
        'edsr': {
            'n_resblocks': 16, 'n_feats': 64, 'scale': 4,
            'in_channels': 4, 'out_channels': 4,
            'batch_size': 16, 'patch_size': 48, 'lr': 1e-4,
            'epochs': 300, 'loss_type': 'l1', 'amp': True,
            'num_workers': 4, 'seed': 42, 'deterministic': True,
            'val_interval': 1, 'checkpoint_interval': 50, 'grad_clip': 1.0,
        },
        'swinir': {
            'upscale': 4, 'img_size': 64, 'window_size': 8,
            'depths': [6,6,6,6,6,6], 'embed_dim': 180, 'num_heads': [6,6,6,6,6,6],
            'mlp_ratio': 2, 'in_chans': 4, 'out_chans': 4,
            'batch_size': 16, 'patch_size': 64, 'lr': 2e-4,
            'epochs': 200, 'loss_type': 'l1+lpips', 'lpips_weight': 0.1, 'amp': True,
            'num_workers': 4, 'seed': 123, 'deterministic': True,
            'val_interval': 1, 'checkpoint_interval': 50, 'grad_clip': 1.0,
        },
        'realesrgan': {
            'net_g': {'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23},
            'scale': 4,
            'batch_size': 16, 'patch_size': 128, 'lr': 1e-4,
            'max_iters': 400000, 'loss_type': 'l1+perceptual+gan',
            'perceptual_weight': 1.0, 'gan_weight': 0.1, 'amp': True,
            'num_workers': 4, 'seed': 456, 'deterministic': True,
            'val_interval': 5000, 'checkpoint_interval': 50000, 'grad_clip': 1.0,
        }
    }
    
    config = CONFIGS[MODEL_NAME]
    
    # Setup
    device = setup_kaggle_environment()
    
    # Data path (adjust to your Kaggle dataset)
    DATA_ROOT = "/kaggle/input/sentinel2-sr-processed"  # Your dataset path
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, DATA_ROOT)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train
    train_model(MODEL_NAME, config, train_loader, val_loader, device)