"""
Phase 4 M1: Prepare LR/HR training pairs from real data
=======================================================
S2 Sim-ALSAT-2B track (methodology §7.2 "Sim ALSAT-2B, ×4, degradation model"):
HR = 4-band 10m tile (B2/B3/B4/B8), LR = ALSAT2BDegradation.forward_ms_only
(Wald protocol: MTF σ=0.35 + area ×4 downsample + Poisson-Gaussian noise
SNR 20dB + 10-bit quantization) -> [4, 256, 256] at 40m.

Pairs are saved FULL-TILE (no cropping): src/training SRDataset does its own
random patch cropping at load time. One pair per source tile, float32 .npy.

Tile selection (verified, honest):
- only tiles with nan_fraction == 0.0 (strictly clean; NaN pixels cannot
  enter the degradation pipeline — voids are excluded, never filled)
- deterministic subsample with seed 42 (smoke scale: default 120 train /
  30 val tiles; full-scale pair generation for Phase 5 runs on Kaggle)

The S2 ×2 native track (20m B5–B12 → 10m) requires the 20m bands, which were
not acquired (approval-gated download) — deferred, documented.

Run:  python scripts/prepare_pairs.py [n_train] [n_val]
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation.alsat2b_degradation import ALSAT2BDegradation  # noqa: E402

META_DIR = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "full"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "pairs"
MAX_NAN_FRACTION = 0.0  # strictly clean tiles only

DEGRADATION_CFG = OmegaConf.create(
    yaml.safe_load((PROJECT_ROOT / "configs" / "degradation" / "alsat2b.yaml").read_text())["degradation"]
)


def collect_pool(split: str):
    """All clean (nan==0) tiles of a split across AOIs, from tile indexes."""
    pool = []
    excluded_nan = 0
    for csv in sorted(META_DIR.glob(f"*_{split}_tile_index.csv")):
        with open(csv, newline="") as f:
            rows = list(__import__("csv").DictReader(f))
        for r in rows:
            nan = float(r["nan_fraction"])
            if nan > MAX_NAN_FRACTION:
                excluded_nan += 1
                continue
            pool.append({
                "tile_id": r["tile_id"], "aoi": r["aoi_id"],
                "path": str(PROJECT_ROOT / r["path"]), "nan": nan,
            })
    return pool, excluded_nan


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = np.random.default_rng(42)

    degradation = ALSAT2BDegradation(DEGRADATION_CFG)
    summary = {}
    for split, n_pick in (("train", n_train), ("val", n_val)):
        pool, excluded = collect_pool(split)
        print(f"{split}: {len(pool)} clean tiles available ({excluded} excluded nan>{MAX_NAN_FRACTION})")
        if len(pool) < n_pick:
            n_pick = len(pool)
            print(f"  -> capping to {n_pick}")
        picked = [pool[i] for i in rng.choice(len(pool), size=n_pick, replace=False)]

        out_lr = OUT_DIR / f"s2_{split}" / "lr"
        out_hr = OUT_DIR / f"s2_{split}" / "hr"
        out_lr.mkdir(parents=True, exist_ok=True)
        out_hr.mkdir(parents=True, exist_ok=True)

        saved = []
        for p in picked:
            with rasterio.open(p["path"]) as src:
                hr = src.read().astype(np.float32)          # [4, 1024, 1024]
            if not np.isfinite(hr).all():
                continue  # paranoia: index said clean, verify anyway
            lr = degradation.forward_ms_only(hr)            # [4, 256, 256]
            np.save(out_lr / f"{p['tile_id']}.npy", lr)
            np.save(out_hr / f"{p['tile_id']}.npy", hr)
            saved.append(p["tile_id"])
            if len(saved) % 20 == 0:
                print(f"  {split}: {len(saved)}/{len(picked)} pairs")

        summary[split] = {
            "n_pairs": len(saved), "n_pool": len(pool), "excluded_nan": excluded,
            "lr_shape": [4, 256, 256], "hr_shape": [4, 1024, 1024],
        }
        print(f"{split}: saved {len(saved)} pairs -> {out_lr.parent}")

    manifest = {
        "dataset": "s2_sim_alsat2b_x4",
        "scale": 4,
        "degradation": "ALSAT2BDegradation forward_ms_only (configs/degradation/alsat2b.yaml, Wald protocol)",
        "tile_selection": "nan_fraction == 0.0, seeded rng(42) subsample",
        "summary": summary,
    }
    out = OUT_DIR / "s2_pairs_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
