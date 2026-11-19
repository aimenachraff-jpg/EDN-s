# %% [cell 1] — SwinIR-base training (200 epochs, batch 16, patch 64, L1+LPIPS, AMP)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rasterio", "omegaconf", "mlflow", "lpips"], check=True)

import csv as csvmod, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch, yaml
from omegaconf import OmegaConf
import torch.nn as nn

T0 = time.time()
def banner(msg):
    print(f"\n===== [{(time.time()-T0)/60:.1f} min] {msg} =====", flush=True)

def find_input_dir(slug):
    hits = []
    for dp, dn, fn in os.walk("/kaggle/input"):
        if os.path.basename(dp) == slug:
            hits.append(dp)
    assert hits, f"dataset {slug} not mounted"
    return hits[0]

CODE = find_input_dir("master2-a1-code")
TILES = find_input_dir("s2-tiles-trainval")
sys.path.insert(0, CODE)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpus", torch.cuda.device_count(), flush=True)
assert torch.cuda.is_available(), "no GPU"
PAIRS = Path("/kaggle/working/pairs")
CKPT_DIR = "/kaggle/working/checkpoints_swinir"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- LPIPS 4-channel fix: patch SRLoss.forward to slice to 3 channels ----
import src.training.trainer as _tr
_orig_fwd = _tr.SRLoss.forward
def _patched_fwd(self, sr, hr, disc_pred=None):
    losses = {}
    losses['pixel'] = self.pixel_loss(sr, hr)
    total = losses['pixel']
    if self.lpips_weight > 0 and self.lpips_fn is not None:
        sr3 = (sr[:, :3] * 2 - 1).contiguous()
        hr3 = (hr[:, :3] * 2 - 1).contiguous()
        losses['lpips'] = self.lpips_fn(sr3, hr3).mean()
        total = total + self.lpips_weight * losses['lpips']
    if self.perceptual_weight > 0 and self.perceptual_loss is not None:
        losses['perceptual'] = self.perceptual_loss(sr, hr)
        total = total + self.perceptual_weight * losses['perceptual']
    if self.gan_weight > 0 and self.gan_loss is not None and disc_pred is not None:
        losses['gan'] = self.gan_loss(disc_pred, True)
        total = total + self.gan_weight * losses['gan']
    losses['total'] = total
    return losses
_tr.SRLoss.forward = _patched_fwd
print("SRLoss.forward patched for 4-channel LPIPS", flush=True)

# ---- layout discovery -----------------------------------------------------
csvs = sorted(glob.glob(f"{TILES}/*_tile_index.csv"))
assert len(csvs) == 16
from rasterio.windows import Window
import rasterio

def clean_rows(split):
    out = []
    for c in csvs:
        if not c.endswith(f"{split}_tile_index.csv"):
            continue
        group = os.path.basename(c).replace("_tile_index.csv", "")
        for r in csvmod.DictReader(open(c)):
            if float(r["nan_fraction"]) == 0.0:
                out.append((group, r))
    return out

# ---- pair precompute (reuse if already done) ------------------------------
import yaml
from omegaconf import OmegaConf
from src.degradation.alsat2b_degradation import ALSAT2BDegradation

deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open(f"{CODE}/configs/degradation/alsat2b.yaml"))["degradation"]))
np.random.seed(123)  # SwinIR seed

manifest = {}
if not (PAIRS / "s2_train" / "lr").exists() or len(list((PAIRS/"s2_train"/"lr").glob("*.npy"))) < 100:
    for split in ("train", "val"):
        lr_dir = PAIRS / f"s2_{split}" / "lr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        rows = clean_rows(split)
        print(f"{split}: {len(rows)} clean tiles", flush=True)
        saved = 0
        for group, r in rows:
            hr_path = (f"{TILES}/{group}/{r['aoi_id']}/{r['split']}/"
                       f"{r['mgrs_tile']}_{r['date']}/"
                       f"tile_r{int(r['row']):05d}_c{int(r['col']):05d}.tif")
            with rasterio.open(hr_path) as src:
                hr = src.read().astype(np.float32)
            if not np.isfinite(hr).all():
                continue
            lr = deg.forward_ms_only(hr).astype(np.float32)
            assert lr.shape == (4, 256, 256) and np.isfinite(lr).all()
            np.save(lr_dir / f"{r['tile_id']}.npy", lr)
            manifest[r["tile_id"]] = os.path.relpath(hr_path, TILES)
            saved += 1
            if saved % 500 == 0:
                print(f"  {split}: {saved}/{len(rows)}", flush=True)
        print(f"{split}: saved {saved}", flush=True)
    json.dump(manifest, open(PAIRS / "manifest.json", "w"))
else:
    manifest = json.load(open(PAIRS / "manifest.json"))
    print("reusing existing pairs:", len(list((PAIRS/"s2_train"/"lr").glob("*.npy"))), "train")

# ---- model ---------------------------------------------------------------
from src.models.swinir import SwinIR
mcfg = OmegaConf.create(yaml.safe_load(open(f"{CODE}/configs/models/swinir_base.yaml"))["model"])
train_cfg = dict(yaml.safe_load(open(f"{CODE}/configs/models/swinir_base.yaml"))["training"])
train_cfg["checkpoint_dir"] = CKPT_DIR
# SwinIR-base (12.87M params, 6-stage window attention) OOMs at batch 16 x patch 64 on a
# single 14.5 GiB GPU. Drop to batch 8; patch 64 kept for convergence parity with protocol.
train_cfg["batch_size"] = 8

model = SwinIR(upscale=mcfg.upscale, img_size=mcfg.img_size, window_size=mcfg.window_size,
               depths=list(mcfg.depths), embed_dim=mcfg.embed_dim,
               num_heads=list(mcfg.num_heads), mlp_ratio=mcfg.mlp_ratio,
               in_chans=mcfg.in_chans, out_chans=mcfg.out_chans)
print(f"SwinIR-base: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

# ---- dataset (LR from npy, HR windowed from tif) -------------------------
import torch
from torch.utils.data import DataLoader, Subset
from src.training.trainer import SRDataset, SRTrainer

class SRTileDataset(SRDataset):
    def __init__(self, lr_dir, manifest_path, patch_size=64, scale=4, augment=True):
        self.lr_dir, self.patch_size = Path(lr_dir), patch_size
        self.scale, self.augment = scale, augment
        self.lr_files = sorted(self.lr_dir.glob("*.npy"))
        assert self.lr_files
        man = json.load(open(manifest_path))
        TILES_G = find_input_dir("s2-tiles-trainval")
        self.hr_paths = [os.path.join(TILES_G, man[f.stem]) for f in self.lr_files]
    def __getitem__(self, idx):
        lr = torch.from_numpy(np.load(self.lr_files[idx])).float()
        _, h, w = lr.shape
        p = self.patch_size
        top = int(torch.randint(0, h - p + 1, (1,)))
        left = int(torch.randint(0, w - p + 1, (1,)))
        lr_p = lr[:, top:top+p, left:left+p]
        with rasterio.open(self.hr_paths[idx]) as src:
            win = Window(left*self.scale, top*self.scale, p*self.scale, p*self.scale)
            hr_p = torch.from_numpy(src.read(window=win).astype(np.float32))
        if self.augment:
            lr_p, hr_p = self._augment(lr_p, hr_p)
        return lr_p, hr_p

train_ds = SRTileDataset(f"{PAIRS}/s2_train/lr", f"{PAIRS}/manifest.json",
                         patch_size=train_cfg["patch_size"], scale=4, augment=True)
val_full = SRTileDataset(f"{PAIRS}/s2_val/lr", f"{PAIRS}/manifest.json",
                         patch_size=train_cfg["patch_size"], scale=4, augment=True)
val_ds = Subset(val_full, range(min(64, len(val_full))))
train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], num_workers=2, pin_memory=True)
print(f"train {len(train_ds)} / val-subset {len(val_ds)}", flush=True)

# ---- train ---------------------------------------------------------------
trainer = SRTrainer(model, train_cfg, train_loader, val_loader,
                    device="cuda", use_mlflow=False, experiment_name="phase5_swinir_base")
t0 = time.time()
trainer.train(max_epochs=train_cfg["epochs"])
elapsed = time.time() - t0
print(f"TRAINING DONE in {elapsed/3600:.2f} h | best_val_psnr={trainer.best_val_psnr:.3f}", flush=True)
json.dump({"model": "SwinIR-base", "epochs": train_cfg["epochs"],
           "train_pairs": len(train_ds), "elapsed_h": round(elapsed/3600, 3),
           "best_val_psnr_patches": trainer.best_val_psnr},
          open("/kaggle/working/swinir_train_summary.json", "w"), indent=2)
banner("SwinIR training done")
