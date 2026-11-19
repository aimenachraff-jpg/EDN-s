#!/usr/bin/env python
"""
Test EDSR-small on real Sentinel-2 with 3-panel comparison:
Original (HR) vs LR input vs EDSR SR output
"""
import sys, os, time, json
import numpy as np, torch, rasterio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.models.edsr import EDSR
import yaml

# Load normalization stats
with open('data/metadata/sentinel2/normalization_stats.json') as f:
    norm_stats = json.load(f)

# Load model
cfg = yaml.safe_load(open('configs/models/edsr_small.yaml'))['model']
model = EDSR(cfg['n_resblocks'], cfg['n_feats'], cfg['scale'], cfg['in_channels'], cfg['out_channels'])
ckpt = torch.load('experiments/checkpoints/edsr_small_phase5_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# Load all 4 bands - CENTER PATCH (high structure)
base_path = "data/raw/sentinel2/CL_01/19KCQ_20240704/"
bands_order = ['B2', 'B3', 'B4', 'B8']
patch_y, patch_x = 2048, 6656
hr_size = 512

print(f"\nLoading patch at ({patch_y}, {patch_x})...")
hr_bands = []
for bname in bands_order:
    fp = base_path + f'19KCQ_20240704_{bname}.tif'
    with rasterio.open(fp) as src:
        arr = src.read(1)
        patch = arr[patch_y:patch_y+hr_size, patch_x:patch_x+hr_size]
        patch = np.nan_to_num(patch, nan=0.0)
        hr_bands.append(patch)

hr_4band = np.stack(hr_bands, axis=0)  # (4, 512, 512) - HIGH RESOLUTION
print(f"HR input: {hr_4band.shape}")

# Normalize using training stats
normed_hr = np.zeros_like(hr_4band)
for i, bname in enumerate(bands_order):
    stats = norm_stats['bands'][bname]
    mean, std = stats['mean'], stats['std']
    normed_hr[i] = (hr_4band[i] - mean) / std

# Create LR by downsampling
lr_size = 256
lr_4band = np.zeros((4, lr_size, lr_size), dtype=np.float32)
for i in range(4):
    lr_4band[i] = zoom(normed_hr[i], (lr_size/hr_size, lr_size/hr_size), order=3)

print(f"LR input: {lr_4band.shape} (downsampled x2)")

# Run SR inference
lr_tensor = torch.from_numpy(lr_4band).unsqueeze(0).float()
with torch.inference_mode():
    t0 = time.time()
    sr_tensor = model(lr_tensor)
    t1 = time.time()
sr_4band = sr_tensor.squeeze(0).numpy()
print(f"SR output: {sr_4band.shape} in {t1-t0:.2f}s")

# Save SR output
out_profile = {
    'driver': 'GTiff',
    'height': 1024, 'width': 1024, 'count': 4, 'dtype': 'float32',
    'crs': 'EPSG:4326',
    'transform': rasterio.transform.from_origin(
        1700000 + patch_x * 10, -5000000 + patch_y * 10, 2.5, 2.5
    )
}
with rasterio.open('chile_3panel_sr.tif', 'w', **out_profile) as dst:
    dst.write(sr_4band)
print("Saved: chile_3panel_sr.tif")

# Function to convert band indices to RGB visualization
def bands_to_rgb(geo_tif_array, band_indices, title="RGB"):
    """Convert GeoTIFF bands to RGB for display."""
    vis = np.transpose(geo_tif_array[band_indices], (1, 2, 0))  # CHW -> HWC
    # Denormalize
    stats = [norm_stats['bands'][bands_order[i]] for i in band_indices]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    for c in range(3):
        vis[:,:,c] = vis[:,:,c] * stds[c] + means[c]
    vis = np.clip(vis, 0, 1)
    return vis

# Create 3 panels: HR vs LR vs SR
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: ORIGINAL (HR) - True RGB
hr_rgb = bands_to_rgb(hr_4band, [2, 1, 0])
axes[0].imshow(hr_rgb)
axes[0].set_title(f"ORIGINAL (High Resolution)\nSentinel-2 Band 4/3/2\n{hr_size}x{hr_size} pixels", fontsize=14, fontweight='bold')
axes[0].axis('off')

# Panel 2: LR INPUT - What the model received (downsampled)
lr_rgb = bands_to_rgb(lr_4band, [2, 1, 0])
axes[1].imshow(lr_rgb)
axes[1].set_title(f"LR INPUT (Degraded)\nDownsampled ×2 via bilinear\n{lr_size}x{lr_size} pixels", fontsize=14, fontweight='bold')
axes[1].axis('off')

# Panel 3: EDSR SR OUTPUT - Model reconstruction
sr_rgb = bands_to_rgb(sr_4band, [2, 1, 0])
axes[2].imshow(sr_rgb)
axes[2].set_title(f"EDSR-SR OUTPUT (Your Model)\nSuper-Resolution ×4 upscaled\n{sr_4band.shape[1]}x{sr_4band.shape[2]} pixels", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('chile_3panel_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: chile_3panel_comparison.png")
plt.show()

print("\n=== SUMMARY ===")
print(f"Location: Chile coast patch at ({patch_y}, {patch_x})")
print(f"Original HR: {hr_4band.shape} (512×512, ~10m resolution)")
print(f"LR Input:    {lr_4band.shape} (256×256, simulated ~40m)")
print(f"SR Output:   {sr_4band.shape} (1024×1024, your model's enhanced ~10m)")
print(f"Inference time: {t1-t0:.2f}s on CPU")