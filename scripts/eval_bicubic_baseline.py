"""
Phase 4 M2: Bicubic baseline evaluation on Sim-ALSAT-2B val pairs
================================================================
Zero-training baseline: bicubic ×4 upscale of the degraded LR, evaluated
against the HR tiles with the full metric suite (PSNR, SSIM, SAM, ERGAS,
SCI, wavelet HH energy, GLCM entropy, TUS).

Input:  data/interim/pairs/s2_val/{lr,hr}/*.npy   (from prepare_pairs.py)
Output: results/metrics/baseline_bicubic.json

Run:  python scripts/eval_bicubic_baseline.py [n_pairs]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import (  # noqa: E402
    _aggregate_metrics,
    bicubic_upscale,
)
from src.evaluation.metrics import evaluate_sr  # noqa: E402

PAIRS_DIR = PROJECT_ROOT / "data" / "interim" / "pairs" / "s2_val"
OUT = PROJECT_ROOT / "results" / "metrics" / "baseline_bicubic.json"


def main():
    n_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    lr_files = sorted((PAIRS_DIR / "lr").glob("*.npy"))[:n_cap]
    hr_files = sorted((PAIRS_DIR / "hr").glob("*.npy"))[:n_cap]
    assert len(lr_files) == len(hr_files) and lr_files, "no val pairs found"
    print(f"evaluating bicubic baseline on {len(lr_files)} val pairs", flush=True)

    t0 = time.time()
    all_metrics = []
    for i, (lr_f, hr_f) in enumerate(zip(lr_files, hr_files)):
        lr = np.load(lr_f)   # [4, 256, 256]
        hr = np.load(hr_f)   # [4, 1024, 1024]
        sr = bicubic_upscale(lr, scale=4)  # [4, 1024, 1024]
        m = evaluate_sr(sr, hr, compute_all=True)
        all_metrics.append(m)
        print(f"  pair {i + 1}/{len(lr_files)} done", flush=True)

    agg = _aggregate_metrics(all_metrics)
    result = {
        "baseline": "bicubic_x4",
        "dataset": "s2_sim_alsat2b_x4 (val pairs, nan==0 tiles)",
        "n_pairs": len(lr_files),
        "scale": 4,
        "elapsed_s": round(time.time() - t0, 1),
        "metrics_mean": {k: v["mean"] for k, v in agg.items()},
        "metrics": agg,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"saved -> {OUT}", flush=True)
    for k, v in result["metrics_mean"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}",
              flush=True)


if __name__ == "__main__":
    main()
