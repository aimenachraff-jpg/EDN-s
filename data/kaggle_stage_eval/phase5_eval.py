# Eval-only kernel: EDSR-small vs bicubic on the same 30 val pairs.
# Training already completed (best.pt from the v2 run); this script only
# loads the checkpoint (weights_only=False, trusted source) and runs Cell 4.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rasterio", "omegaconf"], check=True)

import csv as csvmod, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch, yaml, rasterio
from rasterio.windows import Window
from omegaconf import OmegaConf

T0 = time.time()
def banner(msg):
    print(f"\n===== [{(time.time()-T0)/60:.1f} min] {msg} =====", flush=True)

def find_input_dir(slug):
    hits = []
    for dp, dn, fn in os.walk("/kaggle/input"):
        if os.path.basename(dp) == slug:
            hits.append(dp)
    assert hits, f"dataset {slug} not mounted; /kaggle/input = {os.listdir('/kaggle/input')}"
    return hits[0]

CODE = find_input_dir("master2-a1-code")
TILES = find_input_dir("s2-tiles-trainval")
CKPT_DIR_IN = find_input_dir("phase5-best-ckpt")
sys.path.insert(0, CODE)
print("CODE:", CODE, flush=True)
print("TILES:", TILES, flush=True)
print("CKPT:", CKPT_DIR_IN, flush=True)

import torch
from src.models.edsr import EDSR
from src.inference.inference import bicubic_upscale, _aggregate_metrics
from src.evaluation.metrics import evaluate_sr

# ---- regenerate the 30 val LR pairs we eval on (seed 42, same order) -------
# (avoids needing the 995 stored .npy; deterministic given seed + CSV order)
csvs = sorted(glob.glob(f"{TILES}/*_tile_index.csv"))
assert len(csvs) == 16

from src.degradation.alsat2b_degradation import ALSAT2BDegradation
deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open(f"{CODE}/configs/degradation/alsat2b.yaml"))["degradation"]))
np.random.seed(42)

OUT = Path("/kaggle/working/pairs")
val_lr_dir = OUT / "s2_val" / "lr"
val_lr_dir.mkdir(parents=True, exist_ok=True)
manifest = {}
rows = []
for c in csvs:
    if not c.endswith("_val_tile_index.csv"):
        continue
    group = os.path.basename(c).replace("_tile_index.csv", "")
    for r in csvmod.DictReader(open(c)):
        if float(r["nan_fraction"]) == 0.0:
            rows.append((group, r))

saved = 0
for group, r in rows:
    hr_path = (f"{TILES}/{group}/{r['aoi_id']}/{r['split']}/"
               f"{r['mgrs_tile']}_{r['date']}/"
               f"tile_r{int(r['row']):05d}_c{int(r['col']):05d}.tif")
    with rasterio.open(hr_path) as src:
        hr = src.read().astype(np.float32)
    if not np.isfinite(hr).all():
        continue
    lr = deg.forward_ms_only(hr)
    assert lr.shape == (4, 256, 256) and np.isfinite(lr).all()
    np.save(val_lr_dir / f"{r['tile_id']}.npy", lr.astype(np.float32))
    manifest[r["tile_id"]] = os.path.relpath(hr_path, TILES)
    saved += 1
print(f"regenerated {saved} val LR pairs", flush=True)
json.dump(manifest, open(OUT / "manifest.json", "w"))

# ---- load checkpoint (weights_only=False: trusted source, our own file) ----
ckpt = os.path.join(CKPT_DIR_IN, "best.pt")
assert os.path.exists(ckpt), f"best.pt not found in {CKPT_DIR_IN}"
cfg = yaml.safe_load(open(f"{CODE}/configs/models/edsr_small.yaml"))
model_cfg = OmegaConf.create(cfg["model"])
checkpoint = torch.load(ckpt, map_location="cuda", weights_only=False)
model = EDSR(n_resblocks=model_cfg["n_resblocks"], n_feats=model_cfg["n_feats"],
             scale=model_cfg["scale"], in_channels=model_cfg["in_channels"],
             out_channels=model_cfg["out_channels"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.cuda().eval()
print(f"loaded best.pt (epoch {checkpoint.get('epoch')}, "
      f"best_val_psnr {checkpoint.get('best_val_psnr')})", flush=True)

# ---- Cell 4 eval on the first 30 sorted val pairs ---------------------------
N_PAIRS = 30
man = json.load(open(f"{OUT}/manifest.json"))
lr_files = sorted(glob.glob(f"{OUT}/s2_val/lr/*.npy"))[:N_PAIRS]
assert len(lr_files) == N_PAIRS, f"only {len(lr_files)} val LR tiles"
print(f"evaluating on {N_PAIRS} full-tile val pairs (LR 256 -> SR 1024)", flush=True)

def hr_of(lr_file):
    return os.path.join(TILES, man[os.path.basename(lr_file)[:-4]])

def run(model_fn, name):
    all_m, t0 = [], time.time()
    for i, lf in enumerate(lr_files):
        lr = np.load(lf).astype(np.float32)
        with rasterio.open(hr_of(lf)) as src:
            hr = src.read().astype(np.float32)
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
        t = torch.from_numpy(lr).float().unsqueeze(0).cuda()
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
    "training_summary": {
        "epochs": 300, "best_val_psnr_patches": checkpoint.get("best_val_psnr"),
        "elapsed_h": 6.82,
    },
}, open("/kaggle/working/phase5_eval.json", "w"), indent=2)
print(f"\nVERDICT: EDSR-small {edsr_res['metrics_mean']['psnr']:.2f} dB vs "
      f"bicubic {bic_res['metrics_mean']['psnr']:.2f} dB -> "
      f"{'BEATS baseline' if beats else 'DOES NOT beat baseline'} ({d_psnr:+.2f} dB)",
      flush=True)
banner("EVAL DONE")

# also write a tiny summary
json.dump({"best_val_psnr_patches": checkpoint.get("best_val_psnr"),
           "epochs": 300, "elapsed_h": 6.82},
          open("/kaggle/working/train_summary.json", "w"), indent=2)
print("saved -> /kaggle/working/phase5_eval.json, train_summary.json")
for root, dirs, files in os.walk("/kaggle/working"):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p} ({os.path.getsize(p)/1e6:.2f} MB)")
