#!/usr/bin/env python
"""
Test EDSR-small on a REAL Sentinel-2 raw band.
Uses data/raw/sentinel2/BR_01/22MFD_20190630/B4.tif (Red band, 10m).
"""
import sys, os, time
import numpy as np, torch, rasterio
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.models.edsr import EDSR
import yaml

# 1. Load real Sentinel-2 band
s2_path = "data/raw/sentinel2/BR_01/22MFD_20190630/22MFD_20190630_B4.tif"
print(f"Loading real Sentinel-2 band: {s2_path}")

with rasterio.open(s2_path) as src:
    crs = src.crs
    transform = src.transform
    full = src.read(1)  # Single band
    print(f"  Full resolution: {full.shape}, CRS: {crs}")

# 2. Handle NaN (Sentinel-2 often has NaN for clouds/shadows)
full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  After NaN fix: min={full.min():.4f}, max={full.max():.4f}")

# 3. Normalize using training stats (model expects z-score normalized input)
import json
with open('data/metadata/sentinel2/normalization_stats.json') as f:
    norm_stats = json.load(f)

# Band 4 = Red (B4)
b4_stats = norm_stats['bands']['B4']
mean = b4_stats['mean']
std = b4_stats['std']
print(f"  B4 normalization: mean={mean:.4f}, std={std:.4f}")

# Normalize: (x - mean) / std
full_norm = (full - mean) / std
print(f"  After normalization: min={full_norm.min():.4f}, max={full_norm.max():.4f}")

# 4. Downsample to 256x256 (simulating degraded LR input)
from scipy.ndimage import zoom
lr = zoom(full_norm, (256/full_norm.shape[0], 256/full_norm.shape[1]), order=3)
lr = lr.astype(np.float32)
print(f"  LR input: {lr.shape}, min={lr.min():.4f}, max={lr.max():.4f}")

# 5. Create 4-band array (duplicate B4 for all bands - simplified test)
lr_4band = np.stack([lr, lr, lr, lr], axis=0)  # (4, 256, 256)
print(f"  4-band LR: {lr_4band.shape}")

# 4. Load model
cfg = yaml.safe_load(open('configs/models/edsr_small.yaml'))['model']
model = EDSR(cfg['n_resblocks'], cfg['n_feats'], cfg['scale'], cfg['in_channels'], cfg['out_channels'])
ckpt = torch.load('experiments/checkpoints/edsr_small_phase5_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# 5. Inference
lr_tensor = torch.from_numpy(lr_4band).unsqueeze(0).float()
with torch.inference_mode():
    t0 = time.time()
    sr_tensor = model(lr_tensor)
    t1 = time.time()
sr = sr_tensor.squeeze(0).numpy()
print(f"SR done: {lr_4band.shape} -> {sr.shape} in {t1-t0:.2f}s")

# 6. Save GeoTIFF
sr_profile = {
    'driver': 'GTiff',
    'height': 1024, 'width': 1024, 'count': 4, 'dtype': 'float32',
    'crs': crs,
    'transform': rasterio.transform.from_origin(
        transform.c, transform.f, transform.a/4, transform.e/4
    )
}
with rasterio.open('real_s2_sr.tif', 'w', **sr_profile) as dst:
    dst.write(sr)
print("Saved: real_s2_sr.tif (1024x1024, x4 SR)")

# 7. Visual comparison (denormalize for display)
lr_vis = np.transpose(lr_4band[:3], (1, 2, 0))
sr_vis = np.transpose(sr[:3], (1, 2, 0))

# Denormalize: x * std + mean
lr_vis = lr_vis * std + mean
sr_vis = sr_vis * std + mean

# Clip to valid range [0, 1]
lr_vis = np.clip(lr_vis, 0, 1)
sr_vis = np.clip(sr_vis, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(lr_vis)
axes[0].set_title(f"LR Input (256x256)\nReal Sentinel-2 Band 4 (Red), 10m")
axes[0].axis('off')

axes[1].imshow(sr_vis)
axes[1].set_title(f"EDSR-small SR (1024x1024, x4)\n{t1-t0:.2f}s on CPU")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('real_s2_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: real_s2_comparison.png")
plt.show()

print("\nDone! Files created:")
print("  real_s2_sr.tif       - SR output GeoTIFF (1024x1024, 4 bands)")
print("  real_s2_comparison.png - Before/after visual")