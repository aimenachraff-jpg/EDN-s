"""
Phase 6: Hallucination detection on the test set.

Runs EDSR-small (best.pt) and bicubic x4 over every CLEAN test tile, then
applies src/hallucination/detection.py (spectral / structural / textural /
semantic) to each SR vs its HR reference. Aggregates per-category hallucination
rates and confidence distributions for EDSR vs bicubic.

Semantic detection is skipped (no task model) - reported as 'no_task_model'.
Resume-safe: per-tile results are appended to a JSONL file; re-running skips
already-processed tile_ids.

Run:  python scripts/phase6_hallucination.py [max_tiles]
"""
import csv, glob, json, os, sys, time
import numpy as np
import torch, yaml, rasterio
from pathlib import Path
from omegaconf import OmegaConf

ROOT = r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.models.edsr import EDSR
from src.inference.inference import bicubic_upscale
from src.hallucination.detection import HallucinationDetector, HallucinationType

MAX_TILES = int(sys.argv[1]) if len(sys.argv) > 1 else None
DATA_ROOT = Path("data/processed/sentinel2")
CKPT = "experiments/checkpoints/edsr_small_phase5_best.pt"
OUT_DIR = Path("results/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL = OUT_DIR / "phase6_per_tile.jsonl"
SUMMARY = OUT_DIR / "phase6_hallucination.json"

# ---- model ---------------------------------------------------------------
cfg = yaml.safe_load(open("configs/models/edsr_small.yaml"))
mcfg = OmegaConf.create(cfg["model"])
model = EDSR(n_resblocks=mcfg.n_resblocks, n_feats=mcfg.n_feats, scale=mcfg.scale,
             in_channels=mcfg.in_channels, out_channels=mcfg.out_channels)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---- degradation (deterministic, seed 42) --------------------------------
from src.degradation.alsat2b_degradation import ALSAT2BDegradation
deg = ALSAT2BDegradation(OmegaConf.create(
    yaml.safe_load(open("configs/degradation/alsat2b.yaml"))["degradation"]))
np.random.seed(42)

detector = HallucinationDetector()

# ---- enumerate clean test tiles ------------------------------------------
rows_all = []
for csv_path in sorted(glob.glob("data/metadata/sentinel2/full/*_test_tile_index.csv")):
    group = os.path.basename(csv_path).replace("_tile_index.csv", "")
    aoi = group.split("_")[0] + "_" + group.split("_")[1]
    for r in csv.DictReader(open(csv_path)):
        if float(r["nan_fraction"]) == 0.0:
            rows_all.append((group, r))
print(f"clean test tiles: {len(rows_all)}")
if MAX_TILES:
    rows_all = rows_all[:MAX_TILES]
    print(f"  (capped to {MAX_TILES})")

# ---- resume: skip already-processed tile_ids -----------------------------
done_ids = set()
if JSONL.exists():
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["tile_id"])
print(f"already processed: {len(done_ids)}")

# ---- per-tile processing --------------------------------------------------
t0 = time.time()
n_done = len(done_ids)
n_new = 0
for group, r in rows_all:
    tile_id = r["tile_id"]
    if tile_id in done_ids:
        continue
    hr_path = ("data/processed/sentinel2/" + r["aoi_id"] + "/test/"
               + r["mgrs_tile"] + "_" + r["date"] + "/"
               + "tile_r%05d_c%05d.tif" % (int(r["row"]), int(r["col"])))
    if not os.path.exists(hr_path):
        print("MISSING:", hr_path, flush=True)
        continue
    with rasterio.open(hr_path) as src:
        hr = src.read().astype(np.float32)
    if not np.isfinite(hr).all():
        continue  # never fill voids

    lr = deg.forward_ms_only(hr).astype(np.float32)  # [4,256,256] (degradation may emit float64)
    assert lr.shape == (4, 256, 256) and np.isfinite(lr).all()

    # SR
    with torch.no_grad():
        sr_edsr = model(torch.from_numpy(lr).unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
    sr_bic = bicubic_upscale(lr, scale=4).astype(np.float32)

    # detection (no task model -> semantic skipped)
    res_edsr = detector.detect_all(sr_edsr, hr)
    res_bic = detector.detect_all(sr_bic, hr)

    record = {
        "tile_id": tile_id,
        "aoi": r["aoi_id"],
        "group": group,
        "edsr": {
            "classification": res_edsr["classification"],
            "detected_types": res_edsr["detected_types"],
            "confidences": res_edsr["confidences"],
            "spectral": {k: v for k, v in res_edsr["details"]["spectral"].items()
                         if k in ("detected", "mean_sam", "ndvi_violation",
                                  "correlation_deviation", "confidence")},
            "structural": {k: v for k, v in res_edsr["details"]["structural"].items()
                           if k in ("detected", "confidence", "relative_energy",
                                    "local_anomaly_ratio")},
            "textural": {k: v for k, v in res_edsr["details"]["textural"].items()
                         if k in ("detected", "confidence", "entropy_diff",
                                  "entropy_sr", "entropy_hr")},
        },
        "bicubic": {
            "classification": res_bic["classification"],
            "detected_types": res_bic["detected_types"],
            "confidences": res_bic["confidences"],
            "spectral": {k: v for k, v in res_bic["details"]["spectral"].items()
                         if k in ("detected", "mean_sam", "ndvi_violation",
                                  "correlation_deviation", "confidence")},
            "structural": {k: v for k, v in res_bic["details"]["structural"].items()
                           if k in ("detected", "confidence", "relative_energy",
                                    "local_anomaly_ratio")},
            "textural": {k: v for k, v in res_bic["details"]["textural"].items()
                         if k in ("detected", "confidence", "entropy_diff",
                                  "entropy_sr", "entropy_hr")},
        },
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    done_ids.add(tile_id)
    n_new += 1
    n_done += 1
    if n_done % 50 == 0 or n_new == 1:
        dt = time.time() - t0
        rate = n_new / dt if dt > 0 else 0
        eta = (len(rows_all) - n_done) / rate / 60 if rate > 0 else 0
        print(f"  {n_done}/{len(rows_all)} done | {rate:.2f} tiles/s | "
              f"ETA {eta:.0f} min", flush=True)

# ---- aggregate ------------------------------------------------------------
print("Aggregating...", flush=True)
records = [json.loads(l) for l in open(JSONL) if l.strip()]
print(f"total records: {len(records)}")

def agg(model_key):
    out = {"n": len(records)}
    for cat in ("spectral", "structural", "textural"):
        det = [1 if r[model_key][cat]["detected"] else 0 for r in records]
        conf = [r[model_key][cat].get("confidence", 0) for r in records]
        out[cat] = {
            "detection_rate": float(np.mean(det)),
            "n_detected": int(sum(det)),
            "mean_confidence": float(np.mean(conf)),
        }
    # composite classification counts
    from collections import Counter
    cls = Counter(r[model_key]["classification"] for r in records)
    out["classification_counts"] = dict(cls)
    out["any_hallucination_rate"] = float(np.mean(
        [1 if r[model_key]["classification"] != "none" else 0 for r in records]))
    return out

summary = {
    "phase": "6 - hallucination detection on clean test tiles",
    "n_tiles": len(records),
    "models": {"edsr": agg("edsr"), "bicubic": agg("bicubic")},
    "note": "semantic detection skipped (no task model provided); "
            "detect_all called with task_model=None.",
}
json.dump(summary, open(SUMMARY, "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\nSaved per-tile -> {JSONL}")
print(f"Saved summary  -> {SUMMARY}")
print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
