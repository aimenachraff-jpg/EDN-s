#!/usr/bin/env python
"""
Test EDSR-small on a real Sentinel-2 tile with STRUCTURE (city/vegetation).
Uses CL_01/19KCQ_20240704 - Chile scene with high B8 variance (vegetation/urban).
Creates a 4-band input and runs SR, then shows RGB and NIR visualization.
"""
import sys, os, time, json
import numpy as np, torch, rasterio
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

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

# Load all 4 bands from Chile scene (high structure area)
base_path = "data/raw/sentinel2/CL_01/19KCQ_20240704/"
bands_map = {'B2': 0, 'B3': 1, 'B4': 2, 'B8': 3}
files_order = ['B2', 'B3', 'B4', 'B8']
file_list = [base_path + f'19KCQ_20240704_{b}.tif' for b in files_order]

bands_data = []
for i, fp in enumerate(file_list):
    print(f"Loading {fp}...")
    with rasterio.open(fp) as src:
        arr = src.read(1)
        arr = np.nan_to_num(arr, nan=0.0)
        bands_data.append(arr)
        print(f"  Shape: {arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

full_4band = np.stack(bands_data, axis=0)  # (4, H, W)
print(f"\nFull 4-band array: {full_4band.shape}")

# Normalize each band
H, W = full_4band.shape[1], full_4band.shape[2]
normed = np.zeros_like(full_4band)
for i, bname in enumerate(files_order):
    stats = norm_stats['bands'][bname]
    mean, std = stats['mean'], stats['std']
    normed[i] = (full_4band[i] - mean) / std

# Downsample to 256x256 for LR input
lr_size = 256
lr_4band = np.zeros((4, lr_size, lr_size), dtype=np.float32)
for i in range(4):
    lr_4band[i] = zoom(normed[i], (lr_size/H, lr_size/W), order=3)

print(f"LR input: {lr_4band.shape}")

# Inference
lr_tensor = torch.from_numpy(lr_4band).unsqueeze(0).float()
with torch.inference_mode():
    t0 = time.time()
    sr_tensor = model(lr_tensor)
    t1 = time.time()
sr_4band = sr_tensor.squeeze(0).numpy()
print(f"SR done: {lr_4band.shape} -> {sr_4band.shape} in {t1-t0:.2f}s")

# Save SR output as GeoTIFF
out_profile = {
    'driver': 'GTiff',
    'height': 1024, 'width': 1024, 'count': 4, 'dtype': 'float32',
    'crs': 'EPSG:4326',
    'transform': rasterio.transform.from_origin(
        1700000, -5000000, 40/4, 40/4  # Approximate CRS for this tile
    )
}
with rasterio.open('chile_sr.tif', 'w', **out_profile) as dst:
    dst.write(sr_4band)
print("Saved: chile_sr.tif")

# Visualize: RGB composite (B4/R, B3/G, B2/B) and False color (B8/NIR, B4, B3)
def denormalize_and_visualize(band_indices, title, filename):
    vis = np.transpose(sr_4band[band_indices], (1, 2, 0))
    stats = [norm_stats['bands'][files_order[i]] for i in band_indices]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    # Denormalize
    for c in range(3):
        vis[:,:,c] = vis[:,:,c] * stds[c] + means[c]
    vis = np.clip(vis, 0, 1)
    return vis

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: True RGB (Red/Green/Blue)
rgb_vis = denormalize_and_visualize([2, 1, 0], "True RGB", "RGB")
axes[0].imshow(rgb_vis)
axes[0].set_title(f"True RGB\n(B4=Red, B3=Green, B2=Blue)\nChile coastal area")
axes[0].axis('off')

# Panel 2: False Color (NIR is veg indicator!)
fc_vis = denormalize_and_visualize([3, 2, 1], "False Color\n(NIR=Red, R=Green, G=Blue)", "FCI")
axes[1].imshow(fc_vis)
axes[1].set_title(f"False Color Infrared\n(B8=NIR, B4=Red, B3=Green)\nVegetation appears RED")
axes[1].axis('off')

# Panel 3: NIR alone (shows vegetation structure)
nir_vis = np.stack([sr_4band[3], sr_4band[3], sr_4band[3]], axis=-1)
stats_b8 = norm_stats['bands']['B8']
nir_vis = nir_vis * stats_b8['std'] + stats_b8['mean']
nir_vis = np.clip(nir_vis, 0, 1)
axes[2].imshow(nir_vis, cmap='Greens')
axes[2].set_title(f"NIR Band (B8)\nHigh values = Healthy Vegetation")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('chile_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: chile_comparison.png")
plt.show()

print("\n=== RESULTS ===")
print(f"Input: 4-band Sentinel-2 at ~40m resolution (simulated LR)")
print(f"Output: 4-band SR at ~10m resolution (actual SR x4)")
print(f"Scene: 19KCQ_20240704, Chile coast")
print(f"Bands used: B2(Blue), B3(Green), B4(Red), B8(NIR)")