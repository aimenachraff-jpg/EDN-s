# %% [cell 1 v3] — environment + layout discovery + TIMING PROBE (mandatory, no training)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rasterio", "omegaconf", "mlflow"], check=True)

import csv as csvmod, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch, yaml
from omegaconf import OmegaConf

T0 = time.time()
def banner(msg):
    print(f"\n===== [{(time.time()-T0)/60:.1f} min] {msg} =====", flush=True)

def find_input_dir(slug):
    """Locate a mounted dataset under /kaggle/input regardless of layout
    (legacy: /kaggle/input/<slug>; new: /kaggle/input/datasets/<owner>/<slug>)."""
    hits = []
    for dp, dn, fn in os.walk("/kaggle/input"):
        if os.path.basename(dp) == slug:
            hits.append(dp)
    assert hits, f"dataset {slug} not mounted; /kaggle/input = {os.listdir('/kaggle/input')}"
    return hits[0]

CODE = find_input_dir("master2-a1-code")
TILES = find_input_dir("s2-tiles-trainval")
print("CODE dir:", CODE, flush=True)
print("TILES dir:", TILES, flush=True)
sys.path.insert(0, CODE)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpus", torch.cuda.device_count(), flush=True)
assert torch.cuda.is_available(), "no GPU — check kernel settings"

# ---- 1. layout verification ---------------------------------------------------
csvs = sorted(glob.glob(f"{TILES}/*_tile_index.csv"))
stats_hits = glob.glob(f"{TILES}/normalization_stats.json")
print("csvs found:", len(csvs), "(expect 16)", flush=True)
print("normalization_stats.json:", bool(stats_hits), flush=True)
assert len(csvs) == 16 and stats_hits, "unexpected dataset layout"
STATS = stats_hits[0]

def clean_rows(split):
    out = []
    for c in csvs:
        if not c.endswith(f"{split}_tile_index.csv"):
            continue
        for r in csvmod.DictReader(open(c)):
            if float(r["nan_fraction"]) == 0.0:
                out.append(r)
    return out

train_pool, val_pool = clean_rows("train"), clean_rows("val")
print(f"clean tiles (nan==0): train {len(train_pool)} / val {len(val_pool)}", flush=True)

# ---- 2. tile path resolver (layout: {group}/{aoi}/{split}/{mgrs}_{date}/...) --
def path_of(row, group=None):
    g = group or f"{row['aoi_id']}_{row['split']}"
    return (f"{TILES}/{g}/{row['aoi_id']}/{row['split']}/"
            f"{row['mgrs_tile']}_{row['date']}/"
            f"tile_r{int(row['row']):05d}_c{int(row['col']):05d}.tif")

import rasterio
p0 = path_of(train_pool[0], "BR_01_train")
with rasterio.open(p0) as src:
    hr = src.read().astype(np.float32)
assert hr.shape == (4, 1024, 1024) and np.isfinite(hr).all(), (hr.shape, "non-finite!")
print("sample tile OK:", hr.shape, hr.dtype,
      "range %.3f..%.3f" % (hr.min(), hr.max()), flush=True)

# ---- 3. degradation timing (CPU, per tile) ------------------------------------
from src.degradation.alsat2b_degradation import ALSAT2BDegradation
deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open(f"{CODE}/configs/degradation/alsat2b.yaml"))["degradation"]))
lr0 = deg.forward_ms_only(hr); assert lr0.shape == (4, 256, 256)
t0 = time.time()
for _ in range(3): deg.forward_ms_only(hr)
s_per_tile = (time.time() - t0) / 3
print(f"degradation: {s_per_tile:.2f} s/tile", flush=True)

# ---- 4. GPU training-step timing (EDSR-small, batch 16, patch 48) -------------
from src.models.edsr import EDSR
mcfg = OmegaConf.create(yaml.safe_load(open(f"{CODE}/configs/models/edsr_small.yaml"))["model"])
model = EDSR(n_resblocks=mcfg.n_resblocks, n_feats=mcfg.n_feats, scale=mcfg.scale,
             in_channels=mcfg.in_channels, out_channels=mcfg.out_channels).cuda(0)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(16, 4, 48, 48, device="cuda:0")
y = torch.randn(16, 4, 192, 192, device="cuda:0")
for _ in range(3):                      # warmup
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.l1_loss(model(x), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.l1_loss(model(x), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
torch.cuda.synchronize()
s_per_step = (time.time() - t0) / 10

steps_per_epoch = int(np.ceil(len(train_pool) / 16))
total_h = steps_per_epoch * s_per_step * 300 / 3600
print(f"EDSR-small: {n_params/1e6:.2f}M params | {s_per_step*1000:.0f} ms/step "
      f"-> 300 epochs ~= {total_h:.1f} h", flush=True)
json.dump({"s_per_step": s_per_step, "s_per_tile": s_per_tile,
           "steps_per_epoch": steps_per_epoch, "total_h": total_h,
           "n_params": n_params, "train_pool": len(train_pool),
           "val_pool": len(val_pool)},
          open("/kaggle/working/timing_probe.json", "w"), indent=2)
banner("timing probe done")

# %% [cell 2] — full-scale LR precompute (variant A): LR stored, HR streamed
PAIRS = Path("/kaggle/working/pairs")
np.random.seed(42)  # reproducible degradation noise (same seed as protocol)

summary, manifest = {}, {}
for split in ("train", "val"):
    lr_dir = PAIRS / f"s2_{split}" / "lr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    rows = [(os.path.basename(c).replace("_tile_index.csv", ""), r)
            for c in csvs if c.endswith(f"{split}_tile_index.csv")
            for r in csvmod.DictReader(open(c)) if float(r["nan_fraction"]) == 0.0]
    print(f"{split}: {len(rows)} clean tiles", flush=True)
    t0, saved = time.time(), 0
    for group, r in rows:
        with rasterio.open(path_of(r, group)) as src:
            hr = src.read().astype(np.float32)
        if not np.isfinite(hr).all():
            continue  # paranoia: index said clean, verify anyway (never fill voids)
        lr = deg.forward_ms_only(hr)
        assert lr.shape == (4, 256, 256) and np.isfinite(lr).all()
        np.save(lr_dir / f"{r['tile_id']}.npy", lr)
        manifest[r["tile_id"]] = os.path.relpath(path_of(r, group), TILES)
        saved += 1
        if saved % 400 == 0:
            dt = time.time() - t0
            print(f"  {split}: {saved}/{len(rows)} | {dt/saved:.2f} s/tile | "
                  f"eta {dt/saved*(len(rows)-saved)/60:.0f} min", flush=True)
    summary[split] = {"n_pairs": saved, "n_pool": len(rows)}
    print(f"{split}: saved {saved} LR tiles -> {lr_dir}", flush=True)

json.dump(manifest, open(PAIRS / "manifest.json", "w"))
json.dump({"dataset": "s2_sim_alsat2b_x4_full", "scale": 4,
           "degradation": "ALSAT2BDegradation.forward_ms_only (Wald, seed 42)",
           "lr_root": str(PAIRS), "hr_root": TILES, "summary": summary},
          open(PAIRS / "pairs_manifest.json", "w"), indent=2)
banner(f"precompute done: {summary}")

# %% [cell 3] — EDSR-small training: 300 epochs, batch 16, patch 48, L1, AMP
import logging
from rasterio.windows import Window
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, CODE)
logging.basicConfig(level=logging.INFO)

from src.training.trainer import SRDataset, SRTrainer

CKPT_DIR = "/kaggle/working/checkpoints"

class SRTileDataset(SRDataset):
    """LR from precomputed .npy; HR windowed-read from the dataset tif.
    Crop math is identical to SRDataset._random_crop (no-pad case: 256 >= 48)."""
    def __init__(self, lr_dir, manifest_path, patch_size=48, scale=4, augment=True):
        self.lr_dir, self.patch_size = Path(lr_dir), patch_size
        self.scale, self.augment = scale, augment
        self.lr_files = sorted(self.lr_dir.glob("*.npy"))
        assert self.lr_files, f"no LR tiles in {lr_dir}"
        man = json.load(open(manifest_path))
        self.hr_paths = [os.path.join(TILES, man[f.stem]) for f in self.lr_files]

    def __getitem__(self, idx):
        lr = torch.from_numpy(np.load(self.lr_files[idx])).float()   # [4,256,256]
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

cfg = yaml.safe_load(open(f"{CODE}/configs/models/edsr_small.yaml"))
model_cfg, train_cfg = OmegaConf.create(cfg["model"]), dict(cfg["training"])
train_cfg["checkpoint_dir"] = CKPT_DIR

train_ds = SRTileDataset(f"{PAIRS}/s2_train/lr", f"{PAIRS}/manifest.json",
                         patch_size=train_cfg["patch_size"], scale=4, augment=True)
val_full = SRTileDataset(f"{PAIRS}/s2_val/lr", f"{PAIRS}/manifest.json",
                         patch_size=train_cfg["patch_size"], scale=4, augment=True)
VAL_SUBSET = 64
val_ds = Subset(val_full, range(min(VAL_SUBSET, len(val_full))))
train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                        num_workers=2, pin_memory=True)
print(f"train {len(train_ds)} / val-subset {len(val_ds)} pairs", flush=True)

model = EDSR(n_resblocks=model_cfg["n_resblocks"], n_feats=model_cfg["n_feats"],
             scale=model_cfg["scale"], in_channels=model_cfg["in_channels"],
             out_channels=model_cfg["out_channels"])
print(f"EDSR-small: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

trainer = SRTrainer(model, train_cfg, train_loader, val_loader,
                    device="cuda", use_mlflow=False,
                    experiment_name="phase5_edsr_small")
# note: device must be exactly "cuda" — SRTrainer gates AMP on that string;
# model lands on cuda:0, GPU 1 stays idle (approved plan).

t0 = time.time()
trainer.train(max_epochs=train_cfg["epochs"])   # saves best.pt, epoch_50/100/..., last.pt
elapsed = time.time() - t0
print(f"TRAINING DONE in {elapsed/3600:.2f} h | best_val_psnr={trainer.best_val_psnr:.3f}",
      flush=True)
json.dump({"model": "EDSR-small 16res/64feat x4", "epochs": train_cfg["epochs"],
           "train_pairs": len(train_ds), "val_subset": len(val_ds),
           "elapsed_h": round(elapsed/3600, 3),
           "best_val_psnr_patches": trainer.best_val_psnr},
          open("/kaggle/working/train_summary.json", "w"), indent=2)
banner("training done")

# %% [cell 4] — final eval: EDSR-small vs bicubic on the SAME val pairs
from src.inference.inference import (
    load_model_from_checkpoint, bicubic_upscale, _aggregate_metrics)
from src.evaluation.metrics import evaluate_sr

N_PAIRS = 30          # same protocol as eval_bicubic_baseline.py (Phase 4 M2)
ckpt = f"{CKPT_DIR}/best.pt"
assert os.path.exists(ckpt), "best.pt missing after training"

model = load_model_from_checkpoint(ckpt, "edsr", config=model_cfg, device="cuda")
model.eval()

man = json.load(open(f"{PAIRS}/manifest.json"))
lr_files = sorted(glob.glob(f"{PAIRS}/s2_val/lr/*.npy"))[:N_PAIRS]
assert len(lr_files) == N_PAIRS, f"only {len(lr_files)} val LR tiles"
print(f"evaluating on {N_PAIRS} full-tile val pairs (LR 256 -> SR 1024)", flush=True)

def hr_of(lr_file):
    return os.path.join(TILES, man[os.path.basename(lr_file)[:-4]])

def run(model_fn, name):
    all_m, t0 = [], time.time()
    for i, lf in enumerate(lr_files):
        lr = np.load(lf)                                   # [4,256,256]
        with rasterio.open(hr_of(lf)) as src:
            hr = src.read().astype(np.float32)             # [4,1024,1024]
        sr = model_fn(lr)
        all_m.append(evaluate_sr(sr, hr, compute_all=True))
        if (i + 1) % 10 == 0:
            print(f"  {name}: {i+1}/{N_PAIRS}", flush=True)
    agg = _aggregate_metrics(all_m)
    mean = {k: v["mean"] for k, v in agg.items()}
    print(f"{name} done in {time.time()-t0:.0f}s", flush=True)
    for k, v in mean.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    return {"metrics_mean": mean, "metrics": agg, "n_pairs": N_PAIRS}

def sr_fn_edsr(lr):
    with torch.no_grad():
        t = torch.from_numpy(lr).unsqueeze(0).cuda()
        return model(t).squeeze(0).cpu().numpy().astype(np.float32)

edsr_res = run(sr_fn_edsr, "EDSR-small")
bic_res = run(lambda lr: bicubic_upscale(lr, scale=4), "bicubic")

d_psnr = edsr_res["metrics_mean"]["psnr"] - bic_res["metrics_mean"]["psnr"]
beats = edsr_res["metrics_mean"]["psnr"] > bic_res["metrics_mean"]["psnr"]
json.dump({
    "phase": "5 — EDSR-small on Kaggle 2xT4 (GPU 0)",
    "model_ckpt": "best.pt (best val-patch PSNR during training)",
    "note": "bicubic recomputed here on the SAME 30 val pairs for fairness; "
            "the Phase 4 37.59 dB reference used the smoke-subset pairs",
    "edsr": edsr_res, "bicubic": bic_res,
    "delta_psnr_db": round(d_psnr, 3), "beats_bicubic": bool(beats),
}, open("/kaggle/working/phase5_eval.json", "w"), indent=2)
print(f"\nVERDICT: EDSR-small {edsr_res['metrics_mean']['psnr']:.2f} dB vs "
      f"bicubic {bic_res['metrics_mean']['psnr']:.2f} dB -> "
      f"{'BEATS baseline' if beats else 'DOES NOT beat baseline'} ({d_psnr:+.2f} dB)",
      flush=True)

banner("ALL DONE — outputs in /kaggle/working")
for root, dirs, files in os.walk("/kaggle/working"):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p} ({os.path.getsize(p)/1e6:.2f} MB)")
