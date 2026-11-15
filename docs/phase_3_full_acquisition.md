# Phase 3 — Full-Scale Data Acquisition Report

**Date:** 2026-11-15
**Status:** COMPLETE — acquisition, tiling, and train-only normalization all verified
**Commits:** `bde23c9` (PROBA-V), `c39ccac` (normalization+tiling), `11234a2` (acquisition hardening), `42edbca` (tiling parse fix), void-fix commit (this batch)

---

## 1. Executive Summary

Full-scale acquisition of both thesis datasets, approved by the supervisor on
2026-11-14 ("ALL APPROVED"):

| Dataset | Result | Integrity evidence |
|---|---|---|
| PROBA-V Super-Resolution | 1,450 imgsets (1,160 train / 290 test), 58,202 PNGs, ~698 MB | MD5 `7101101c7c91eeb5979d472861c1b784` MATCH (Zenodo-recorded value) |
| Sentinel-2 L2A (8 AOIs) | 87/88 scenes, 67.86 GB, 24 manifests | Per-file MD5 in manifests; sanity scan 89/89 OK, 0 flags |

All pipeline bugs discovered during acquisition were **root-caused with direct
evidence** (no guessing) and fixed; every fix is committed with its evidence in
the message.

---

## 2. PROBA-V (Verified Facts)

- Source: Zenodo record 6327426, `probav_data.zip`, 708,416,603 bytes.
- Downloaded via `scripts/download_probav.py`; checksum verified before
  extraction; extraction produced 58,202 PNGs + `norm.csv` (1,450 lines).
- **Layout finding (important, corrected earlier assumption):** imgset IDs are
  **partitioned by band**, not duplicated: train/RED = 0000–0593 (594),
  train/NIR = 0594–1159 (566), test/RED = 1160–1305 (146),
  test/NIR = 1306–1449 (144). Totals match the official spec exactly
  (1,160 train + 290 test).
- Train imgsets: `HR.png + LR###.png + QM###.png`; test imgsets:
  `LR###.png + QM###.png + SM.png` (HR withheld per challenge).
- Loader (`src/data/probav_loader.py`) rewritten to **single-band semantics**
  (TDD, 12 tests against the real archive, all passing):
  - QM/SM masks normalized to {0,1} (archive stores clear=255)
  - test scenes surface SM as the evaluation mask, no HR
  - `get_scene_stack` returns `[N,1,128,128]` LR + `[1,384,384]` HR

### Limitation (documented)
No official region→imgset mapping ships with the data (verified against the
ESA Kelvins data page and arXiv:2101.10200 — proba-v-ref identifies the
*reference image*, not regions). Train/val split is therefore by scene ID
with seed 42; documented as a potential-leakage limitation.

---

## 3. Sentinel-2 Full Acquisition (Verified Facts)

- Source: Microsoft Planetary Computer STAC API (`sentinel-2-l2a`), no auth.
- Design: 8 AOIs (DZ_01–04 Algeria, FR_01 Paris, US_01 Iowa, CL_01 Chile,
  BR_01 Amazon) × (6 train / 2 val / 3 test) = 88 scenes target.
- Bands B2/B3/B4/B8 at 10 m, output EPSG:4326 at 0.0001° (~10 m at mid-lat).
- Selection: distinct dates evenly spread across each split's period
  (temporal stratification), distinct MGRS tiles preferred within a date.

### Final inventory (all 24 manifests verified)

| AOI | train | val | test | GB |
|---|---|---|---|---|
| DZ_01 | 5 | 2 | 3 | 6.95 |
| DZ_02 | 6 | 2 | 3 | 8.59 |
| DZ_03 | 6 | 2 | 3 | 7.98 |
| DZ_04 | 6 | 2 | 3 | 8.96 |
| FR_01 | 6 | 2 | 3 | 9.63 |
| US_01 | 6 | 2 | 3 | 10.32 |
| CL_01 | 6 | 2 | 3 | 8.92 |
| BR_01 | 6 | 2 | 3 | 6.52 |
| **Total** | **47** | **16** | **24** | **67.86** |

- **DZ_01 train = 5/6 (documented shortfall):** its 2020-12-23 date had only
  one candidate granule, which was rejected as degenerate (see §4.4) and
  recorded in `data/metadata/sentinel2/full/rejected_items.json` so no future
  run re-selects it.

### Quality audit (independent of download code)

`scan_all_scenes.py` checked **every** downloaded scene (subsampled reads):
nonzero fraction, unique-value count, value range, per-band sizes.
**Result: 89/89 scene dirs OK, 0 flags** (89 = 87 manifest scenes + 2 healthy
pilot-validation scenes kept outside the manifests).

---

## 4. Failure Modes Found & Fixed (each verified, not guessed)

### 4.1 Partial granules pass the cloud filter
- **Evidence:** `31SEB_20201228` had `eo:cloud_cover`=3.4% (filter passed) but
  `s2:nodata_pixel_percentage`=87.1%; SCL showed 87% nodata class — the
  granule covered only ~13% of the tile.
- **Fix:** selection filter `s2:nodata_pixel_percentage ≤ 50`.

### 4.2 Raw-DN saturation for pre-baseline-04.00 scenes
- **Evidence:** PC assets for baseline < 04.00 lack `raster:bands` scale
  metadata; odc-stac returned raw DN (0–10000); `clip(0,1)` saturated files to
  constant 1.0 (0.9 MB files, e.g. `31SEA_20211225` was 98.7% value 1.0).
- **Fix:** value-range detection (`max > 2.0` → rescale `/10000`).

### 4.3 SAS token expiry mid-run (HTTP 403 crash)
- **Evidence:** FR_01/val crash; failed href carried `se=...17:41:37Z` against
  a read at 17:41:55Z.
- **Fix:** `_resign()` re-signs asset hrefs before **every band load**; band
  reads moved into isolated try/except (odc-stac is lazy — the previous
  load-scoped try never caught the read failure).

### 4.4 Degenerate (defective) granule
- **Evidence:** `31SDB_20201223` passed cloud + nodata filters but is 96–99%
  raw DN=0/1 (nodata/saturated) → ~0.0001 reflectance after rescale;
  value-distribution check confirmed on all 4 bands.
- **Fix:** post-download check (>90% of finite pixels ≤ 0.001 → reject,
  delete dir, persist item ID to `rejected_items.json`).

### 4.5 Manifest builder crash on skipped scenes
- **Evidence:** `TypeError: string indices must be integers` — skip tuple
  reached the checksum loop.
- **Fix:** skip branch reuses existing manifest entries (rich metadata
  preserved across re-runs) or backfills shape/bounds/transform from GeoTIFF
  headers.

### Rulings recorded
1. Test split excludes dates whose best candidate is ≥95% cloud (config says
   "no filter"; a 100%-cloud scene has zero usable surface pixels).
2. Tile-preference within a date never overrides usability (same 95% rule).
3. EPSG:4326 uniform output retained (consistent with pilot + manifests);
   per-tile UTM reprojection deferred to the tiling module (documented).
4. Degenerate date dropped rather than force-filled (honest shortfall).

---

## 5. Tiling & Normalization (COMPLETE)

- `scripts/generate_tiles.py`: 1024×1024 tiles, 128 px overlap (per
  `configs/data/sentinel2.yaml`), per-scene 4-band stacks, per-AOI/split tile
  index CSV with NaN fraction per tile; resumable.
- Normalization statistics: **train split only** (leakage policy), computed
  per config method — percentile 2–98 clip to [0,1] — plus mean/std for
  reference. Implemented in `src/preprocessing/normalization.py`
  (16 tests, TDD; caught a real variance bug before any real data was
  processed). Output: `data/metadata/sentinel2/normalization_stats.json`.

### Tiling results (verified on disk)

| Item | Value |
|---|---|
| Tiles | **14,748** (7,968 train / 2,712 val / 4,068 test), one 4-band float32 GeoTIFF each |
| Size on disk | 100.15 GB under `data/processed/sentinel2/{AOI}/{split}/{mgrs}_{date}/` |
| Index CSVs | 24 files, row counts match per-scene logs exactly |
| Train stats from | 7,968 train tiles, all valid pixels streamed (n_pixels ≈ 7.63e9 per band); percentiles from a seeded 2,000-pixel/tile subsample (15,468,142 → after void fix 14,364,742 samples) |

### Void-fill bug found and fixed (verified)

The raw GeoTIFFs carry **no nodata tag**: the EPSG:4326 reprojection fills the
voids around the rotated granule with exact 0.0. First-pass tiles preserved
these zeros, so fully-void tiles were recorded with `nan_fraction ≈ 0`
(indistinguishable from clean data) and the first-pass train statistics were
computed partly on fake zeros.

- **Scan** (`void_scan.json`): 14/87 scenes have >5% void (BR_01 ×4 up to 43%,
  DZ_01 ×2, DZ_02 ×2 up to 44%, DZ_03 ×2, FR_01 ×3 up to 27%, US_01 ×1 41%);
  **all 14 are train scenes** — val/test splits are void-free.
- **Fix**: a pixel that is exactly 0.0 in all four bands simultaneously is
  fill, not surface; converted to NaN at scene load. Void-affected scenes
  re-tiled from the untouched raw data; indexes and stats regenerated.
- **Verification**: the previously all-zero tile (BR_01 `22MFD_20190630`
  r0c0) now reads `nan_fraction = 1.0`; regenerated stats (void pixels
  excluded) show every band mean rising (B2 0.1469→0.1608, B3 0.1757→0.1924,
  B4 0.2076→0.2273, B8 0.2906→0.3183) and p2 > 0 — a real dark-pixel floor
  instead of void-fill zeros.

### Final normalization statistics (train-only, NaN-excluded)

| Band | p2 | p98 | mean | std |
|---|---|---|---|---|
| B2 | 0.0040 | 0.9216 | 0.1608 | 0.2112 |
| B3 | 0.0010 | 0.9216 | 0.1924 | 0.2102 |
| B4 | 0.0001 | 0.9384 | 0.2273 | 0.2294 |
| B8 | 0.0001 | 0.9384 | 0.3183 | 0.2240 |

### Limitations (documented)

1. SCL (scene classification) bands were not downloaded; cloud masking at
   tile level is not yet applied. Tiles record NaN fraction so loaders can
   filter high-cloud tiles later. Candidate mitigation: re-download SCL for
   masked tiling in Phase 4, or accept cloud content as realistic degradation.
2. Void-tile filtering (tiles that are mostly NaN) happens at loader time;
   no tiles were deleted — loaders choose their own NaN-fraction threshold
   from the tile index.

---

## 6. Test Suite

245 tests passing, including 12 PROBA-V loader tests against the real
archive, 16 normalization tests, and 82 preprocessing tests.

## 7. Reproducibility

```text
PROBA-V:    python scripts/download_probav.py            # checksum-verified
Sentinel-2: python scripts/sentinel2_full.py [AOI...]    # resumable
Sanity:     python scan_all_scenes.py                    # 89/89 OK
Tiling:     python scripts/generate_tiles.py             # resumable
Stats only: python scripts/stats_only.py                 # stats from existing indexes
```

Manifests: `data/metadata/probav/probav_manifest.json`,
`data/metadata/sentinel2/full/{AOI}_{split}_manifest.json`,
`.../{AOI}_{split}_tile_index.csv` (24 files),
`data/metadata/sentinel2/normalization_stats.json` (train-only),
`data/metadata/sentinel2/void_scan.json` (void audit), and
`data/metadata/sentinel2/sanity_scan.json` (89/89 OK).
