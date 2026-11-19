#!/usr/bin/env python
"""
Quick test: Download image → Run EDSR-small SR → Show comparison
Usage: python test_model_quick.py [IMAGE_URL]
"""
import sys, os, time, io
import numpy as np, torch, rasterio, requests
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, os.getcwd())
from src.models.edsr import EDSR
import yaml

# 1. Download image
url = sys.argv[1] if len(sys.argv) > 1 else "https://picsum.photos/512/512"
print(f"Downloading {url}...")
resp = requests.get(url, timeout=60)
resp.raise_for_status()
Path("test_input_raw.png").write_bytes(resp.content)
img = Image.open("test_input_raw.png").convert("RGB").resize((256, 256))

# 2. Convert to 4-band array
arr = np.array(img).astype(np.float32) / 255.0
if arr.ndim == 2:
    arr = np.dstack([arr]*3)
if arr.shape[2] == 3:
    arr = np.dstack([arr, arr[:, :, 1:2]])  # Add pseudo-NIR
arr = np.transpose(arr, (2, 0, 1))  # CHW

# 3. Load model
cfg = yaml.safe_load(open('configs/models/edsr_small.yaml'))['model']
model = EDSR(cfg['n_resblocks'], cfg['n_feats'], cfg['scale'], cfg['in_channels'], cfg['out_channels'])
ckpt = torch.load('experiments/checkpoints/edsr_small_phase5_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# 4. Inference
lr_tensor = torch.from_numpy(arr).unsqueeze(0).float()
with torch.inference_mode():
    t0 = time.time()
    sr_tensor = model(lr_tensor)
    t1 = time.time()
sr = sr_tensor.squeeze(0).numpy()
print(f"SR done: {arr.shape} -> {sr.shape} in {t1-t0:.2f}s")

# 5. Visualize
lr_vis = np.transpose(arr[:3], (1, 2, 0))
sr_vis = np.transpose(sr[:3], (1, 2, 0))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(np.clip(lr_vis*1.5, 0, 1))
axes[0].set_title(f"Input (256×256)")
axes[0].axis('off')
axes[1].imshow(np.clip(sr_vis*1.5, 0, 1))
axes[1].set_title(f"EDSR SR (1024×1024, ×4) - {t1-t0:.2f}s")
axes[1].axis('off')
plt.tight_layout()
plt.savefig('test_comparison.png', dpi=150)
print("Saved: test_comparison.png")
plt.show()
