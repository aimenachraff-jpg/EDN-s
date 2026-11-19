# Phase 5 — Kaggle Notebook Draft (for approval; NOT yet run)
#
# Purpose: full-scale training on Kaggle 2× T4. This file is a cell-by-cell
# draft (paste each `# %% [cell]` block into a Kaggle notebook cell).
# Real training needs USER APPROVAL per AGENTS.md gates (>30 min runs,
# large uploads, cloud compute).
#
# Author: MASTER2-A1 Phase 4 M4 draft, 2026-11-15.

# %% [cell 0] — DATA UPLOAD (browser, once; not code)
#
# Kaggle private dataset "s2-tiles":
#   upload data/processed/sentinel2/          # train+val tiles only (~70 GB)
#   upload data/metadata/sentinel2/full/*_tile_index.csv
#   upload data/metadata/sentinel2/normalization_stats.json
# Kaggle private dataset "master2-a1-code":
#   upload src/, configs/, scripts/prepare_pairs.py, scripts/train_edsr.py
#
# Upload notes: train+val = 7,968+2,712 tiles ≈ 70 GB; test tiles (4,068)
# are NOT needed until Phase 6 evaluation. Kaggle dataset limit ~100 GB.

# %% [cell 1] — environment
# Kaggle preinstalls torch (CUDA), opencv, scipy. Needed extras:
# !pip install -q rasterio omegaconf
# from google.protobuf... (none; keep minimal)
import os, sys, json, time, glob
import numpy as np
import torch
sys.path.insert(0, "/kaggle/working")   # repo copied to /kaggle/working
print(torch.__version__, torch.cuda.device_count())  # expect 2 GPUs

# %% [cell 2] — full-scale pair precompute (CPU, one-off, cached)
# Reuses scripts/prepare_pairs.py logic at FULL scale:
#   python scripts/prepare_pairs.py 4212 1060   # full clean pools
# Estimated: ~120 tiles in ~4 min locally -> 5,272 tiles ≈ 3-4 h CPU on
# Kaggle (4 vCPU). Estimate to be REPLACED by the measured first-batch
# timing in cell 2b; if >4 h, degrade on-the-fly in the DataLoader
# instead (variant B below). Output cached as a Kaggle working dataset.
#
# Variant B (if precompute too slow): SRDataset wrapper that runs
# forward_ms_only inside __getitem__ (4 DataLoader workers hide the
# ~1-2 s/tile CPU cost behind GPU steps).

# %% [cell 2b] — timing probe (MANDATORY before committing to 300 epochs)
# Time 1 batch on 1 GPU; log it. Total training time estimate =
#   steps/epoch × epoch_time × epochs. If the EDSR-small 300-epoch
#   estimate exceeds a Kaggle 12 h session, cut epochs or split sessions
#   (resume from checkpoint). Record the estimate in the session log
#   BEFORE launching the full run; report it to the user.

# %% [cell 3] — training, 2 GPUs
# torch.multiprocessing / one process per GPU (DataParallel is slower
# for SR models). GPU 0: EDSR-small (edsr_small.yaml: 16 blocks/64 feats,
# 300 epochs, L1, patch 48). GPU 1: second family after EDSR is validated
# (SwinIR-base or Real-ESRGAN — user chooses at approval).
# Checkpoints every N epochs to /kaggle/working, downloadable at session
# end; MLflow replaced by Kaggle-native logging (CSV + JSON metrics) to
# keep the notebook self-contained (methodology §9.3 alternative).

# %% [cell 4] — in-session validation
# evaluate_on_pairs on the 30-pair local val set subset + bicubic baseline
# for immediate comparison (expect > 37.59 dB PSNR for a trained model —
# HYPOTHESIS to verify, not a promise).

# %% [cell 5] — outputs
# Save: best checkpoint per model, metrics JSON, timing log.
# Kaggle output is capped (~20 GB) — checkpoints only, no tiles.

# %% [open questions for the approval request]
# 1. Approve uploading train+val tiles (~70 GB) to a private Kaggle dataset?
# 2. Train EDSR-small first (single model, validate pipeline end-to-end)?
# 3. Second model on GPU 1 during the same session: SwinIR-base or
#    Real-ESRGAN (or leave idle)?
# 4. S2 ×2 native track needs 20m bands (B5-B12) — download now
#    (approval-gated, ~30-50 GB) or defer until after the ×4 results?
