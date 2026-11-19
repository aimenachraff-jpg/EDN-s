"""
Phase 4 M3: EDSR-tiny CPU smoke training
========================================
NOT a thesis experiment. Proves the full pipeline end-to-end on CPU in
<30 min (methodology §9.1): pairs -> SRDataset -> SRTrainer -> checkpoint
saved -> checkpoint reloaded -> evaluate_on_pairs roundtrip.

Model: EDSR-tiny (configs/models/edsr_tiny.yaml: 4 resblocks / 32 feats),
2 epochs over the 119-pair smoke set. Real training (EDSR-small, SwinIR,
Real-ESRGAN, 300 epochs) is Phase 5 on Kaggle 2x T4.

Run:  python scripts/train_smoke_edsr_tiny.py
"""

import json
import time
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).parent.parent

import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.edsr import EDSR                                     # noqa: E402
from src.inference.inference import evaluate_on_pairs, load_model_from_checkpoint  # noqa: E402
from src.training.trainer import SRDataset, SRTrainer                # noqa: E402

PAIRS = PROJECT_ROOT / "data" / "interim" / "pairs"
CKPT_DIR = "experiments/checkpoints/edsr_tiny_smoke"
N_VAL_EVAL = 3          # pairs for the post-training eval roundtrip
VAL_SUBSET = 8          # val samples during training


def main():
    t0 = time.time()
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "models" / "edsr_tiny.yaml").read_text())
    model_cfg, train_cfg = OmegaConf.create(cfg["model"]), cfg["training"]

    train_ds = SRDataset(str(PAIRS / "s2_train" / "lr"), str(PAIRS / "s2_train" / "hr"),
                         patch_size=train_cfg["patch_size"], scale=4, augment=True)
    val_ds = SRDataset(str(PAIRS / "s2_val" / "lr"), str(PAIRS / "s2_val" / "hr"),
                       patch_size=train_cfg["patch_size"], scale=4, augment=True)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, range(min(VAL_SUBSET, len(val_ds)))),
                            batch_size=train_cfg["batch_size"], num_workers=0)

    model = EDSR(
        n_resblocks=model_cfg["n_resblocks"], n_feats=model_cfg["n_feats"],
        scale=model_cfg["scale"], in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EDSR-tiny: {n_params/1e6:.3f}M params, "
          f"{len(train_ds)} train / {len(val_ds)} val pairs", flush=True)

    tc = dict(train_cfg)
    tc["checkpoint_dir"] = CKPT_DIR
    trainer = SRTrainer(model, tc, train_loader, val_loader,
                        device="cpu", use_mlflow=False,
                        experiment_name="phase4_m3_smoke")
    trainer.train(max_epochs=train_cfg["epochs"])
    print(f"training done in {time.time()-t0:.0f}s, "
          f"best_val_psnr={trainer.best_val_psnr:.3f}", flush=True)

    # ---- Roundtrip proof: reload from checkpoint and evaluate ----
    ckpt_path = PROJECT_ROOT / CKPT_DIR / "last.pt"
    assert ckpt_path.exists(), f"checkpoint missing: {ckpt_path}"
    reloaded = load_model_from_checkpoint(str(ckpt_path), "edsr",
                                          config=model_cfg, device="cpu")

    lr_files = sorted((PAIRS / "s2_val" / "lr").glob("*.npy"))[:N_VAL_EVAL]
    hr_files = sorted((PAIRS / "s2_val" / "hr").glob("*.npy"))[:N_VAL_EVAL]
    metrics = evaluate_on_pairs(reloaded, [str(p) for p in lr_files],
                                [str(p) for p in hr_files], device="cpu")
    mean_psnr = metrics["psnr"]["mean"]
    print(f"reload+eval roundtrip OK on {N_VAL_EVAL} full val pairs: "
          f"psnr={mean_psnr:.3f} dB", flush=True)

    # Full-tile inference on a trained model must produce finite output
    assert metrics["psnr"]["n"] == N_VAL_EVAL
    assert mean_psnr == mean_psnr and mean_psnr > 0  # finite sanity

    result = {
        "smoke_test": "edsr_tiny CPU (Phase 4 M3) — NOT a thesis experiment",
        "model": "EDSR 4 resblocks / 32 feats / scale 4",
        "n_params_m": round(n_params / 1e6, 3),
        "epochs": train_cfg["epochs"],
        "n_train_pairs": len(train_ds),
        "best_val_psnr_patches": trainer.best_val_psnr,
        "roundtrip_psnr_fulltiles": mean_psnr,
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "elapsed_s": round(time.time() - t0, 1),
        "roundtrip_metrics": metrics,
    }
    out = PROJECT_ROOT / "results" / "metrics" / "smoke_edsr_tiny.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
