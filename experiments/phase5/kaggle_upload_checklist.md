# Phase 5 — Kaggle Upload Checklist (approved 2026-11-15)

Approved decisions (user, 2026-11-15):
1. ✅ Upload train+val tiles (~72 GB) to a **private** Kaggle dataset
2. ✅ EDSR-small trains first (GPU 0); GPU 1 idle for now
3. ✅ S2 20m bands (B5–B12) deferred until ×4 results exist
4. Cell draft: `phase5_notebook_draft.py` (paste into Kaggle cells)

## What to upload

| Local path | Kaggle dataset name | Size (verified) |
|---|---|---|
| `data/processed/sentinel2/` (train+val tiles only) | `s2-tiles-trainval` | ~72.2 GB (7,968 + 2,712 tiles) |
| `data/metadata/sentinel2/full/*_tile_index.csv` (12 train + 12 val) | same dataset, `metadata/` folder | a few MB |
| `data/metadata/sentinel2/normalization_stats.json` | same dataset | ~1 KB |
| `src/`, `configs/`, `scripts/prepare_pairs.py` | `master2-a1-code` | ~1 MB |

Test tiles (4,068) are NOT uploaded (Phase 6 evaluation needs them, later,
from local disk or a second dataset).

## How to upload (browser, recommended — CLI not installed, no API token)

1. kaggle.com → Create → New Dataset → drag & drop
   `data\processed\sentinel2\` (browser upload resumes on interruption)
2. Add the CSVs and `normalization_stats.json` under `metadata/`
3. Keep visibility **Private**, name `s2-tiles-trainval`
4. Second dataset `master2-a1-code` from the repo files (~1 MB)

Realistic transfer time: at 20 Mbps up ≈ 8–9 h; 50 Mbps ≈ 3.5 h.
Start it before an overnight break.

## Alternative: kaggle CLI (optional, faster for retries)

```
pip install kaggle
# kaggle.com → Settings → Create New API Token → save to %USERPROFILE%\.kaggle\kaggle.json
# then from MASTER2-A1:
kaggle datasets init -p data/processed/sentinel2   # edit dataset-metadata.json: id "s2-tiles-trainval", license CC BY 4.0
kaggle datasets create -p data/processed/sentinel2 -r skip
# -r skip = upload files as-is, do NOT zip (disk has no room for a 72 GB zip copy)
```

NOTE: the CLI upload of ~10,700 files is slow; browser drag&drop is
usually faster. Either way the notebook reads the dataset at
`/kaggle/input/s2-tiles-trainval/`.

## After upload: session plan (per `phase5_notebook_draft.py`)

1. Timing probe cell FIRST (mandatory; records steps/s and total estimate)
2. Full-scale pair precompute (5,272 tiles) or on-the-fly variant B
3. EDSR-small on GPU 0 (300 epochs per protocol; split across sessions
   with checkpoint resume if >12 h)
4. Metrics JSON + best checkpoint to Kaggle output; download locally
5. In-session comparison against the verified bicubic baseline
   (PSNR 37.59 dB — trained model must beat it for the pipeline to be sane)

OpenCode progress widget: the upload happens in the browser, not a shell —
the widget will not track it; check the Kaggle page for upload progress.
