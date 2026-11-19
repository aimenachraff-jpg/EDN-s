# %% [cell 1] — Real-ESRGAN training (max 50k iters, batch 16, patch 128, L1+perceptual+GAN, AMP)
# NOTE: protocol specifies 400k iters; capped to 50k here to fit Kaggle sessions (~2-3h).
# Increase max_iters below if more compute is available.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rasterio", "omegaconf", "mlflow"], check=True)

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

# ---- inline patch: Kaggle master2-a1-code dataset has a buggy trainer.py ----
# _setup_loss references `lr` which is only in scope inside _setup_optimizer.
# Since the dataset can't be versioned on this account, patch the mounted file in place.
import importlib
importlib.invalidate_caches()
import src.training.trainer as _tr  # noqa: E402
_tr_path = Path(_tr.__file__)
_s = _tr_path.read_text()
if "lr=lr" in _s:
    _s = _s.replace(
        "AdamW(self.discriminator.parameters(), lr=lr, weight_decay=1e-4)",
        "AdamW(self.discriminator.parameters(), lr=self.config.get('lr', 1e-4), weight_decay=1e-4)")
    _tr_path.write_text(_s)
    print("patched trainer.py _setup_loss (lr NameError fix)", flush=True)
else:
    print("trainer.py already patched", flush=True)
assert torch.cuda.is_available(), "no GPU"
PAIRS = Path("/kaggle/working/pairs")
CKPT_DIR = "/kaggle/working/checkpoints_realesrgan"
os.makedirs(CKPT_DIR, exist_ok=True)

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
from omegaconf import OmegaConf
from src.degradation.alsat2b_degradation import ALSAT2BDegradation

deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open(f"{CODE}/configs/degradation/alsat2b.yaml"))["degradation"]))
np.random.seed(456)  # Real-ESRGAN seed

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
from src.models.realesrgan import RealESRGAN, RRDBNet
mcfg = yaml.safe_load(open(f"{CODE}/configs/models/realesrgan.yaml"))
# RealESRGAN.__init__ builds the RRDBNet itself from the net_g config dict
model = RealESRGAN(net_g=mcfg["model"]["net_g"], scale=mcfg["model"]["scale"])
print(f"Real-ESRGAN: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

# ---- inline patch: trainer.py _setup_loss has NameError: 'lr' in disc_optimizer ----
import pathlib as _pl
_tr_path = _pl.Path(_tr.__file__)
_s = _tr_path.read_text()
_s = _s.replace(
    "AdamW(self.discriminator.parameters(), lr=lr, weight_decay=1e-4)",
    "AdamW(self.discriminator.parameters(), lr=self.config.get('lr', 1e-4), weight_decay=1e-4)")
_tr_path.write_text(_s)
print("patched trainer.py _setup_loss (lr NameError fix)", flush=True)

# ---- config ---------------------------------------------------------------
train_cfg = dict(mcfg["training"])
train_cfg["checkpoint_dir"] = CKPT_DIR
# cap iters to fit Kaggle session; override val/checkpoint intervals for monitoring
MAX_ITERS = 50000
train_cfg["max_iters"] = MAX_ITERS
train_cfg["val_interval"] = 50
train_cfg["checkpoint_interval"] = 200
train_cfg["loss_type"] = "l1+perceptual+gan"
train_cfg["perceptual_weight"] = 1.0
train_cfg["gan_weight"] = 0.1

# ---- dataset ---------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, Subset
from src.training.trainer import SRDataset, SRTrainer

class SRTileDataset(SRDataset):
    def __init__(self, lr_dir, manifest_path, patch_size=128, scale=4, augment=True):
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
            from rasterio.windows import Window
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
                    device="cuda", use_mlflow=False, experiment_name="phase5_realesrgan")
t0 = time.time()
trainer.train(max_epochs=2000, max_iters=MAX_ITERS)
elapsed = time.time() - t0
print(f"TRAINING DONE in {elapsed/3600:.2f} h | best_val_psnr={trainer.best_val_psnr:.3f}", flush=True)
json.dump({"model": "Real-ESRGAN (RRDBNet 23 blocks)", "max_iters": MAX_ITERS,
           "train_pairs": len(train_ds), "elapsed_h": round(elapsed/3600, 3),
           "best_val_psnr_patches": trainer.best_val_psnr},
          open("/kaggle/working/realesrgan_train_summary.json", "w"), indent=2)
banner("Real-ESRGAN training done")
