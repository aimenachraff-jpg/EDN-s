"""Generate before/after SR example images from a real test tile."""
import sys, os, time
import numpy as np
import torch, yaml, rasterio
from PIL import Image
from omegaconf import OmegaConf

ROOT = r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.models.edsr import EDSR
from src.inference.inference import bicubic_upscale
from src.degradation.alsat2b_degradation import ALSAT2BDegradation

# pick a clean test tile
import csv
rows = [r for r in csv.DictReader(open("data/metadata/sentinel2/full/BR_01_test_tile_index.csv"))
        if float(r["nan_fraction"]) == 0.0]
r = rows[42]  # pick one
hr_path = ("data/processed/sentinel2/" + r["aoi_id"] + "/test/"
           + r["mgrs_tile"] + "_" + r["date"] + "/"
           + "tile_r%05d_c%05d.tif" % (int(r["row"]), int(r["col"])))
print("tile:", r["tile_id"], hr_path)

with rasterio.open(hr_path) as src:
    hr = src.read().astype(np.float32)  # [4,1024,1024]

# degrade -> LR
deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open("configs/degradation/alsat2b.yaml"))["degradation"]))
np.random.seed(42)
lr = deg.forward_ms_only(hr).astype(np.float32)  # [4,256,256]

# EDSR SR
cfg = yaml.safe_load(open("configs/models/edsr_small.yaml"))
mcfg = OmegaConf.create(cfg["model"])
model = EDSR(n_resblocks=mcfg.n_resblocks, n_feats=mcfg.n_feats, scale=mcfg.scale,
             in_channels=mcfg.in_channels, out_channels=mcfg.out_channels)
ckpt = torch.load("experiments/checkpoints/edsr_small_phase5_best.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
with torch.no_grad():
    sr = model(torch.from_numpy(lr).unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
print("SR shape:", sr.shape)

# bicubic
bic = bicubic_upscale(lr, scale=4).astype(np.float32)

def to_rgb(arr, bands=(3, 2, 1)):
    """False-color composite: NIR, Red, Green -> RGB. Percentile stretch."""
    rgb = np.stack([arr[b] for b in bands], axis=-1)  # [H,W,3]
    # percentile stretch per channel
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)

# Save individual images
os.makedirs("tmp/examples", exist_ok=True)
Image.fromarray(to_rgb(lr)).save("tmp/examples/01_LR_input.png")
Image.fromarray(to_rgb(bic)).save("tmp/examples/02_bicubic_SR.png")
Image.fromarray(to_rgb(sr)).save("tmp/examples/03_EDSR_SR.png")
Image.fromarray(to_rgb(hr)).save("tmp/examples/04_HR_reference.png")

# Side-by-side comparison: LR (upscaled nearest) | bicubic | EDSR | HR
def resize_nearest(img, scale):
    return np.array(Image.fromarray(img).resize(
        (img.shape[1]*scale, img.shape[0]*scale), Image.NEAREST))
lr_up = resize_nearest(to_rgb(lr), 4)
bic_img = to_rgb(bic)
sr_img = to_rgb(sr)
hr_img = to_rgb(hr)

# downsample HR to manageable size for display (512x512)
def down(img, size=512):
    return np.array(Image.fromarray(img).resize((size, size), Image.LANCZOS))

panel = np.concatenate([down(lr_up), down(bic_img), down(sr_img), down(hr_img)], axis=1)
Image.fromarray(panel).save("tmp/examples/00_comparison_panel.png")

print("Saved:")
for f in sorted(os.listdir("tmp/examples")):
    print("  tmp/examples/" + f)