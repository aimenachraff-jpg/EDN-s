"""Benchmark CPU inference + verify test-tile path resolution."""
import csv, os, sys, time
import numpy as np
import torch, yaml, rasterio
from omegaconf import OmegaConf

ROOT = r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# test tile path resolution
p = "data/metadata/sentinel2/full/BR_01_test_tile_index.csv"
rows = [r for r in csv.DictReader(open(p)) if float(r["nan_fraction"]) == 0.0]
r = rows[0]
group = "BR_01_test"
hr_path = ("data/processed/sentinel2/" + group + "/" + r["aoi_id"] + "/test/"
           + r["mgrs_tile"] + "_" + r["date"] + "/"
           + "tile_r%05d_c%05d.tif" % (int(r["row"]), int(r["col"])))
print("exists:", os.path.exists(hr_path))
print("path:", hr_path)
with rasterio.open(hr_path) as src:
    hr = src.read().astype(np.float32)
print("shape:", hr.shape)

# CPU inference benchmark
from src.models.edsr import EDSR
cfg = yaml.safe_load(open("configs/models/edsr_small.yaml"))
mcfg = OmegaConf.create(cfg["model"])
model = EDSR(n_resblocks=mcfg.n_resblocks, n_feats=mcfg.n_feats, scale=mcfg.scale,
             in_channels=mcfg.in_channels, out_channels=mcfg.out_channels)
ckpt = torch.load("experiments/checkpoints/edsr_small_phase5_best.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
lr = np.random.rand(4, 256, 256).astype(np.float32)
t = torch.from_numpy(lr).unsqueeze(0)
t0 = time.time()
with torch.no_grad():
    for _ in range(3):
        _ = model(t)
print("CPU forward (256->1024) x3:", round((time.time() - t0) / 3, 3), "s each")
print("estimated 1823 clean tiles:", round((time.time() - t0) / 3 * 1823 / 3600, 2), "h")
