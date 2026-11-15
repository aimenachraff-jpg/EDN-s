"""
Inference Pipeline for Super-Resolution Models

Implements inference utilities for evaluating trained models:
- Model loading from checkpoints
- Batched inference with tiling for large images
- Metrics computation
- Result saving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import numpy as np
import logging
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.edsr import EDSR, EDSRPretrained
from src.models.swinir import SwinIR, TinySwinIR
from src.models.realesrgan import RealESRGAN, RealESRGANPretrained
from src.evaluation.metrics import evaluate_sr

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, 
                               model_type: str,
                               config: Optional[DictConfig] = None,
                               device: str = 'cpu') -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        model_type: 'edsr', 'swinir', 'realesrgan'
        config: Model config (if None, uses defaults)
        device: Device to load on
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint if available.
    # NOTE: an explicitly passed `config` wins over checkpoint['config']:
    # the checkpoint stores the TRAINING config (epochs, lr, ...), which
    # lacks the model keys (n_resblocks, ...) needed to rebuild the right
    # architecture — silently loading a wrong-shaped model under
    # strict=False. A tiny-model checkpoint must be loadable with an
    # explicit model config (Phase 4 M3 smoke train hit exactly this).
    if config is not None:
        model_config = config
    elif 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = {}
    
    # Create model
    if model_type == 'edsr':
        model = EDSR(
            n_resblocks=model_config.get('n_resblocks', 16),
            n_feats=model_config.get('n_feats', 64),
            scale=model_config.get('scale', 4),
            in_channels=model_config.get('in_channels', 4),
            out_channels=model_config.get('out_channels', 4)
        )
    elif model_type == 'swinir':
        model = SwinIR(
            upscale=model_config.get('upscale', 4),
            img_size=model_config.get('img_size', 64),
            window_size=model_config.get('window_size', 8),
            depths=model_config.get('depths', [6]*6),
            embed_dim=model_config.get('embed_dim', 180),
            num_heads=model_config.get('num_heads', [6]*6),
            mlp_ratio=model_config.get('mlp_ratio', 2),
            in_chans=model_config.get('in_chans', 4),
            out_chans=model_config.get('out_chans', 4)
        )
    elif model_type == 'realesrgan':
        model = RealESRGAN(
            net_g=model_config.get('net_g', {
                'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23
            }),
            scale=model_config.get('scale', 4)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle pretrained models with 3->4 channel adaptation
    if 'pretrained_3ch' in model_config and model_config['pretrained_3ch']:
        # Adapt first conv
        if model_type == 'edsr' and 'head.weight' in state_dict:
            w = state_dict['head.weight']
            if w.shape[1] == 3:
                state_dict['head.weight'] = w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        elif model_type == 'swinir' and 'patch_embed.proj.weight' in state_dict:
            w = state_dict['patch_embed.proj.weight']
            if w.shape[1] == 3:
                state_dict['patch_embed.proj.weight'] = w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        elif model_type == 'realesrgan' and 'conv_first.weight' in state_dict:
            w = state_dict['conv_first.weight']
            if w.shape[1] == 3:
                state_dict['conv_first.weight'] = w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} from {checkpoint_path}")
    return model


def create_model(model_type: str, config: DictConfig, device: str = 'cpu') -> nn.Module:
    """Create model from config (for new training)."""
    if model_type == 'edsr':
        model = EDSR(
            n_resblocks=config.get('n_resblocks', 16),
            n_feats=config.get('n_feats', 64),
            scale=config.get('scale', 4),
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4)
        )
    elif model_type == 'swinir':
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
    elif model_type == 'realesrgan':
        model = RealESRGAN(
            net_g=config.get('net_g', {
                'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23
            }),
            scale=config.get('scale', 4)
        )
    elif model_type == 'bicubic':
        # Bicubic is handled separately (no model needed)
        return None
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


# =============================================================================
# TILED INFERENCE FOR LARGE IMAGES
# =============================================================================

def tile_inference(model: nn.Module, lr_image: np.ndarray, 
                   tile_size: int = 512, tile_overlap: int = 32,
                   scale: int = 4, device: str = 'cpu') -> np.ndarray:
    """
    Run inference on large image using tiling with overlap.
    
    Args:
        model: SR model in eval mode
        lr_image: LR image [C, H, W] in [0, 1]
        tile_size: Tile size in LR space
        tile_overlap: Overlap between tiles in LR space
        scale: Upscaling factor
        device: Device for inference
    
    Returns:
        SR image [C, H*scale, W*scale] in [0, 1]
    """
    C, H, W = lr_image.shape
    sr_H, sr_W = H * scale, W * scale
    
    # Output array
    sr_image = np.zeros((C, sr_H, sr_W), dtype=np.float32)
    weight_map = np.zeros((sr_H, sr_W), dtype=np.float32)
    
    # Gaussian weight for blending
    weight_kernel = gaussian_2d(tile_size * scale, sigma=tile_size * scale / 6)
    
    # Calculate tile positions
    step = tile_size - tile_overlap
    n_tiles_h = (H - tile_overlap + step - 1) // step
    n_tiles_w = (W - tile_overlap + step - 1) // step
    
    logger.info(f"Tiled inference: {n_tiles_h}x{n_tiles_w} tiles, "
                f"tile={tile_size}, overlap={tile_overlap}")
    
    model.eval()
    with torch.no_grad():
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # LR tile coordinates
                h_start = i * step
                w_start = j * step
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)
                
                # Extract tile
                lr_tile = lr_image[:, h_start:h_end, w_start:w_end]
                
                # Pad if needed
                pad_h = tile_size - lr_tile.shape[1]
                pad_w = tile_size - lr_tile.shape[2]
                if pad_h > 0 or pad_w > 0:
                    lr_tile = np.pad(lr_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                
                # Inference
                lr_tensor = torch.from_numpy(lr_tile).unsqueeze(0).float().to(device)
                sr_tile = model(lr_tensor).squeeze(0).cpu().numpy()
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    sr_tile = sr_tile[:, :sr_tile.shape[1]-pad_h*scale, :sr_tile.shape[2]-pad_w*scale]
                
                # SR tile coordinates
                sr_h_start = h_start * scale
                sr_w_start = w_start * scale
                sr_h_end = sr_h_start + sr_tile.shape[1]
                sr_w_end = sr_w_start + sr_tile.shape[2]
                
                # Blend with weight
                weight = weight_kernel[:sr_tile.shape[1], :sr_tile.shape[2]]
                
                sr_image[:, sr_h_start:sr_h_end, sr_w_start:sr_w_end] += sr_tile * weight
                weight_map[sr_h_start:sr_h_end, sr_w_start:sr_w_end] += weight
    
    # Normalize by weight map
    weight_map = np.maximum(weight_map, 1e-8)
    sr_image = sr_image / weight_map[np.newaxis, :, :]
    
    return np.clip(sr_image, 0, 1)


def gaussian_2d(size: int, sigma: float) -> np.ndarray:
    """Create 2D Gaussian kernel."""
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()


# =============================================================================
# BATCHED INFERENCE
# =============================================================================

def batch_inference(model: nn.Module, lr_batch: np.ndarray,
                    device: str = 'cpu') -> np.ndarray:
    """
    Run inference on batch of LR images.
    
    Args:
        model: SR model in eval mode
        lr_batch: Batch of LR images [B, C, H, W] in [0, 1]
        device: Device for inference
    
    Returns:
        Batch of SR images [B, C, H*scale, W*scale] in [0, 1]
    """
    model.eval()
    B = lr_batch.shape[0]
    results = []
    
    with torch.no_grad():
        for i in range(B):
            lr_tensor = torch.from_numpy(lr_batch[i]).unsqueeze(0).float().to(device)
            sr_tensor = model(lr_tensor)
            results.append(sr_tensor.squeeze(0).cpu().numpy())
    
    return np.stack(results, axis=0)


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_model(model: nn.Module, test_loader, 
                   device: str = 'cpu',
                   compute_hallucination: bool = False,
                   task_model: Optional[nn.Module] = None,
                   task_metric: Optional[callable] = None) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: SR model in eval mode
        test_loader: DataLoader with (lr, hr) pairs
        device: Device for inference
        compute_hallucination: Whether to compute hallucination metrics
        task_model: Fixed-backbone task model for semantic hallucination
        task_metric: Task metric function
    
    Returns:
        Aggregated metrics
    """
    from src.evaluation.metrics import evaluate_sr
    from src.hallucination.detection import HallucinationDetector
    
    model.eval()
    detector = HallucinationDetector() if compute_hallucination else None
    
    all_metrics = {
        'reconstruction': {'psnr': [], 'ssim': [], 'lpips': []},
        'spectral': {'sam': [], 'ergas': [], 'band_correlation_dev': [], 'sci': []},
        'structural': {'qnr': [], 'structural_hallucination': []},
        'hallucination': {'composite': [], 'sci_norm': [], 'structural_norm': [], 'semantic_norm': []},
    }
    
    if task_model is not None:
        all_metrics['downstream'] = {'task_utility': [], 'ratio': []}
    
    with torch.no_grad():
        for lr, hr in tqdm(test_loader, desc="Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # Convert to numpy
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            
            # Evaluate each sample in batch
            for i in range(sr_np.shape[0]):
                metrics = evaluate_sr(
                    sr_np[i], hr_np[i],
                    task_model=task_model,
                    task_metric=task_metric,
                    compute_all=True
                )
                
                # Aggregate
                for cat, vals in metrics.items():
                    if cat in all_metrics:
                        for k, v in vals.items():
                            if isinstance(v, (int, float)) and k in all_metrics[cat]:
                                all_metrics[cat][k].append(v)
    
    # Compute statistics
    results = {}
    for cat, metrics in all_metrics.items():
        results[cat] = {}
        for k, vals in metrics.items():
            if vals:
                results[cat][k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'median': float(np.median(vals)),
                }
    
    return results


def evaluate_on_pairs(model: nn.Module, lr_paths: List[str], hr_paths: List[str],
                      device: str = 'cpu') -> Dict[str, Any]:
    """
    Evaluate model on list of LR-HR pairs.
    
    Args:
        model: SR model
        lr_paths: List of LR image paths (.npy)
        hr_paths: List of HR image paths (.npy)
        device: Device for inference
    
    Returns:
        Aggregated metrics
    """
    from src.evaluation.metrics import evaluate_sr
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for lr_path, hr_path in zip(lr_paths, hr_paths):
            lr = np.load(lr_path)
            hr = np.load(hr_path)
            
            # Inference
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).float().to(device)
            sr_tensor = model(lr_tensor)
            sr = sr_tensor.squeeze(0).cpu().numpy()
            
            # Evaluate
            metrics = evaluate_sr(sr, hr, compute_all=True)
            all_metrics.append(metrics)
    
    # Aggregate
    return _aggregate_metrics(all_metrics)


def _aggregate_metrics(all_metrics: List[Dict]) -> Dict[str, Any]:
    """Aggregate metrics across samples.

    Handles flat format: {'psnr': value, 'ssim': value} - raw values from evaluate_sr
    Returns nested dict: {'metric_name': {'mean': ..., 'std': ...}}
    """
    if not all_metrics:
        return {}
    
    results = {}
    for metric in all_metrics[0].keys():
        vals = []
        for m in all_metrics:
            val = m.get(metric)
            if isinstance(val, (int, float)):
                vals.append(val)
        
        if vals:
            results[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
                'n': len(vals)
            }
    
    return results


# =============================================================================
# BICUBIC BASELINE
# =============================================================================

def bicubic_upscale(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Bicubic upsampling baseline.
    
    Args:
        lr: LR image [C, H, W] or [H, W]
        scale: Upscaling factor
    
    Returns:
        SR image [C, H*scale, W*scale]
    """
    import cv2
    
    if lr.ndim == 2:
        lr = lr[np.newaxis, ...]
    
    C, H, W = lr.shape
    sr = np.zeros((C, H * scale, W * scale), dtype=np.float32)
    
    for c in range(C):
        sr[c] = cv2.resize(lr[c], (W * scale, H * scale), 
                          interpolation=cv2.INTER_CUBIC)
    
    return np.clip(sr, 0, 1)


# =============================================================================
# MAIN INFERENCE SCRIPT
# =============================================================================

def run_inference(config: DictConfig):
    """
    Main inference entry point.
    
    Expected config:
    - model: model_type, checkpoint_path, model_config
    - data: test_lr_dir, test_hr_dir, output_dir
    - inference: tile_size, tile_overlap, batch_size, device
    """
    model_type = config.model.model_type
    checkpoint = config.model.checkpoint_path
    device = config.inference.get('device', 'cpu')
    
    # Load model
    model = load_model_from_checkpoint(checkpoint, model_type, config.model, device)
    
    # Get test data
    test_lr_dir = Path(config.data.test_lr_dir)
    test_hr_dir = Path(config.data.test_hr_dir)
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lr_files = sorted(test_lr_dir.glob("*.npy"))
    hr_files = sorted(test_hr_dir.glob("*.npy"))
    
    # Evaluate
    results = evaluate_on_pairs(model, 
                                [str(f) for f in lr_files],
                                [str(f) for f in hr_files],
                                device)
    
    # Save results
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save SR outputs
    tile_size = config.inference.get('tile_size', 512)
    tile_overlap = config.inference.get('tile_overlap', 32)
    
    for lr_path, hr_path in zip(lr_files, hr_files):
        lr = np.load(lr_path)
        
        if lr.shape[1] > tile_size or lr.shape[2] > tile_size:
            sr = tile_inference(model, lr, tile_size, tile_overlap, 
                               config.model.get('scale', 4), device)
        else:
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).float().to(device)
            with torch.no_grad():
                sr = model(lr_tensor).squeeze(0).cpu().numpy()
        
        # Save
        out_path = output_dir / lr_path.name
        np.save(out_path, sr)
    
    logger.info(f"Inference complete. Results saved to {output_dir}")
    return results


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    print("Testing inference pipeline...")
    
    # Create models
    edsr = EDSR()
    swinir = TinySwinIR()
    realesrgan = RealESRGAN(
        net_g={'num_in_ch': 4, 'num_out_ch': 4, 'num_feat': 64, 'num_block': 23},
        scale=4
    )
    
    models = {'EDSR': edsr, 'SwinIR': swinir, 'RealESRGAN': realesrgan}
    
    # Test bicubic
    lr = np.random.rand(4, 64, 64).astype(np.float32)
    sr_bicubic = bicubic_upscale(lr, 4)
    print(f"Bicubic: {lr.shape} -> {sr_bicubic.shape}")
    
    # Test model inference
    for name, model in models.items():
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        print(f"{name}: {x.shape} -> {y.shape}")
    
    # Test tiling
    large_lr = np.random.rand(4, 1024, 1024).astype(np.float32)
    sr_tiled = tile_inference(edsr, large_lr, tile_size=512, tile_overlap=32, scale=4)
    print(f"Tiled inference: {large_lr.shape} -> {sr_tiled.shape}")
    
    print("Inference pipeline test passed!")