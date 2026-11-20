# Dataset

## Overview

EDN-S uses Sentinel-2 L2A imagery as a substitute for ALSAT-2B. Real ALSAT-2B scenes
were not available, so the ALSAT-2B setting is simulated through a degradation model.
No result in this repository is a measurement on real ALSAT-2B data.

| Property | Value |
|---|---|
| Source | Sentinel-2 L2A (Copernicus) |
| Bands | B2, B3, B4, B8 |
| Native resolution | 10 m |
| AOIs | 8 |
| Scenes accepted | 87 |
| Tiles | 14,748 at 1024×1024, float32 |
| Split | 7,968 train / 2,712 val / 4,068 test |
| Tile overlap | 128 px |
| Normalisation | Train-only z-score, NaN excluded |

## Bands

| Channel | Band | Name | Approx. wavelength | Resolution |
|---|---|---|---|---|
| 1 | B2 | Blue | 490 nm | 10 m |
| 2 | B3 | Green | 560 nm | 10 m |
| 3 | B4 | Red | 665 nm | 10 m |
| 4 | B8 | NIR | 842 nm | 10 m |

Band order is fixed throughout the code. Do not reorder channels without updating the
normalisation statistics and the checkpoint.

## Directory layout

```
data/
├── raw/                     Downloaded scenes
├── interim/
│   └── pairs/
│       ├── train/lr         LR .npy arrays (256×256)
│       ├── train/hr         HR references (1024×1024)
│       └── val/             Validation pairs
├── processed/
│   └── sentinel2/<aoi>/<split>/   Tiled GeoTIFFs
├── metadata/
│   └── sentinel2/
│       ├── normalization_stats.json
│       └── <aoi>_<split>_tile_index.csv
└── splits/
```

## Tiling

- Tile size: 1024×1024.
- Overlap: 128 px.
- Only tiles with `nan_fraction == 0.0` are used for pairs and evaluation.
- Voids in rotated granules are converted to NaN before tiling. Without this step they
  would appear as exact zeros.

## Normalisation

Statistics are computed on the training split only, excluding NaN pixels:

- Per-band mean and standard deviation.
- Per-band 2nd and 98th percentiles.
- Stored in `data/metadata/sentinel2/normalization_stats.json`.
- A copy is bundled with the package at `src/edns/data/normalization_stats.json`.

The CLI applies z-score normalisation by default. The model was trained on z-score data;
using a different normalisation without retraining will degrade results.

## Degradation

LR images are generated with the Wald protocol:

1. Trapezoidal spectral response.
2. Gaussian MTF (σ ≈ 0.35 px).
3. Poisson–Gaussian noise (SNR ≈ 20 dB).
4. 12→10-bit quantisation.
5. Sub-pixel shift (< 0.3 px).
6. Area downsampling ×4.

Parameters are in [`configs/degradation/alsat2b.yaml`](../configs/degradation/alsat2b.yaml).
They approximate the ALSAT-2B acquisition chain and are not a calibrated sensor model.

## Pair generation

```bash
python scripts/prepare_pairs.py
```

The script reads clean tiles, applies `ALSAT2BDegradation.forward_ms_only`, and writes LR
`.npy` arrays. HR tiles are streamed during training through windowed rasterio reads.

For the reported run, 4,154 train and 995 validation LR arrays were produced; 123 tiles
were excluded because they were non-finite on read even though the index reported
`nan_fraction == 0.0`.

## PROBA-V

PROBA-V was acquired and verified (1,450 imgsets, 58,202 PNGs, MD5 checked against
Zenodo) but was not used for the main results. MISR pair generation was not performed.
