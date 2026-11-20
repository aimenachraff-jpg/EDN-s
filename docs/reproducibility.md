# Reproducibility

This document records the environment, data, configuration and commands behind the
reported EDN-S results. Every number in the README traces to a file under
`results/metrics/`.

---

## 1. Environment

### Local development and inference

| Component | Value |
|---|---|
| OS | Windows 10/11 |
| Python | 3.10.11 |
| PyTorch | 2.3.0 (CPU) |
| NumPy | 1.26.4 |
| SciPy | 1.13.0 |
| scikit-image | 0.22.0 |
| rasterio | 1.4.0 |
| GPU | None (Intel Iris Xe integrated graphics) |

### Training

| Component | Value |
|---|---|
| Platform | Kaggle |
| GPU | 2× NVIDIA T4 16 GB (GPU 0 used) |
| PyTorch | 2.x with CUDA |
| Mixed precision | fp16 AMP |

Exact package pins for local development are in [`requirements.txt`](../requirements.txt);
cloud extras are in [`requirements-cloud.txt`](../requirements-cloud.txt).

---

## 2. Determinism

| Component | Setting |
|---|---|
| Global seed | 42 |
| Data splits | Fixed CSVs under `data/metadata/sentinel2/` |
| Degradation noise | `numpy.random.default_rng(42)` / legacy seed 42 |
| Training shuffle | Seeded DataLoader |
| cuDNN | `deterministic=True`, `benchmark=False` |
| Checkpoint selection | Best validation PSNR |

Determinism is enforced where the stack allows it. GPU kernels can still produce small
non-deterministic differences depending on the CUDA/cuDNN build.

---

## 3. Data provenance

### Sentinel-2 L2A

| Attribute | Value |
|---|---|
| Source | Copernicus Sentinel-2 L2A |
| Bands | B2, B3, B4, B8 |
| Native resolution | 10 m |
| AOIs | 8 |
| Scenes accepted | 87 (one degenerate granule rejected) |
| Tiles | 14,748 at 1024×1024, float32 |
| Splits | 7,968 train / 2,712 val / 4,068 test |
| Tile overlap | 128 px |
| Normalisation | Train-only, NaN-excluded, percentile 2–98 and mean/std |
| Stats file | `data/metadata/sentinel2/normalization_stats.json` |

Void handling: raw GeoTIFFs carry no nodata tag, so rotated-granule voids appeared as
exact zeros. All-four-band-zero pixels were converted to NaN before tiling, and the
affected scenes were re-tiled.

### PROBA-V

PROBA-V was acquired (1,450 imgsets, 58,202 PNGs, MD5-verified against Zenodo) but was
not used for the main EDSR-small results. MISR pair generation was not performed.

### ALSAT-2B

No real ALSAT-2B imagery was obtained. The ALSAT-2B setting is simulated from Sentinel-2
with the degradation model in `src/degradation/alsat2b_degradation.py` and
`configs/degradation/alsat2b.yaml`.

---

## 4. Degradation model

The Wald-style pipeline applies:

| Stage | Parameter | Provenance |
|---|---|---|
| Spectral response | Trapezoidal band edges | Documented approximation |
| MTF | Gaussian, σ ≈ 0.35 px | Estimated |
| Noise | Poisson–Gaussian, SNR ≈ 20 dB | Conservative estimate |
| Quantisation | 12-bit → 10-bit | Documented |
| Sub-pixel shift | < 0.3 px | Documented |
| Downsampling | Area average ×4 | Wald protocol |

These parameters approximate the ALSAT-2B acquisition chain; they are not a calibrated
sensor model.

---

## 5. Training configuration

| Setting | Value |
|---|---|
| Model | EDSR-small |
| Residual blocks | 16 |
| Feature channels | 64 |
| Scale | ×4 |
| Input / output channels | 4 / 4 |
| Parameters | 1,518,764 |
| Batch size | 16 |
| Patch size | 48×48 |
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Schedule | Cosine annealing to 1e-6 |
| Loss | L1 |
| Epochs | 300 |
| Steps | 78,020 |
| Mixed precision | fp16 AMP |
| Gradient clipping | 1.0 |
| Training time | 6.82 h |
| Best validation PSNR | 25.169 dB (patch-level, epoch 239) |

Config: [`configs/models/edsr_small.yaml`](../configs/models/edsr_small.yaml).

### Pair construction

- Only tiles with `nan_fraction == 0.0` are used.
- 4,154 train and 995 val LR arrays were precomputed; 123 tiles were excluded because
  they were non-finite on read despite the index value.
- HR is streamed at training time with windowed rasterio reads.

---

## 6. Evaluation protocol

### Quality

- 30 validation pairs.
- Full-tile evaluation: LR 256×256 → SR 1024×1024.
- Bicubic ×4 recomputed on the same pairs.
- Metrics: PSNR, MSE, MAE, SSIM, SAM, ERGAS, SCI, wavelet HH energy, GLCM entropy.
- Output: `results/metrics/phase5_edsr_small_eval.json`.

### Hallucination

- 1,742 clean test tiles (`nan_fraction == 0.0`).
- HR read from `data/processed/sentinel2/<aoi>/test/`.
- LR generated with `ALSAT2BDegradation.forward_ms_only`, seed 42.
- SR produced by EDSR-small (CPU) and bicubic ×4.
- `HallucinationDetector.detect_all(sr, hr)` for both methods.
- Semantic category skipped (no task model).
- Output: `results/metrics/phase6_hallucination.json` and
  `results/metrics/phase6_per_tile.jsonl`.

### Statistical analysis

- PSNR difference and Cohen’s d from summary statistics.
- McNemar tests per hallucination category.
- 95% confidence intervals per category.
- Output: `results/metrics/phase8_statistical_analysis.json`.

> The PSNR test is approximate because per-pair PSNR values were not saved. This is
> stated in the result file and in the README.

---

## 7. Reproduction commands

```bash
# Install
pip install -e .

# Run the test suite
python -m pytest tests/ -q

# Basic reconstruction
edns-sr input.tif -o output.tif --ckpt experiments/checkpoints/edsr_small_phase5_best.pt

# Reconstruction with hallucination report
edns-sr input.tif -o output.tif \
  --ckpt experiments/checkpoints/edsr_small_phase5_best.pt \
  --hallucination --hr reference_hr.tif --report report.json

# Rebuild the documentation figures
python scripts/make_banner.py
python scripts/make_results_gallery.py
python scripts/make_architecture_diagram.py
python scripts/make_metrics_chart.py
```

Dataset acquisition and tiling scripts are in `scripts/`:

| Script | Purpose |
|---|---|
| `sentinel2_full.py` | Scene download |
| `generate_tiles.py` | Tiling |
| `prepare_pairs.py` | LR/HR pair generation |
| `eval_bicubic_baseline.py` | Bicubic baseline |
| `phase6_hallucination.py` | Hallucination evaluation |
| `phase8_statistical_analysis.py` | Statistical tests |

---

## 8. Known reproducibility limits

| Limit | Impact |
|---|---|
| No real ALSAT-2B data | Results describe the simulated setting only |
| Single seed | Run-to-run variance is not measured |
| 30 validation pairs | Wide confidence intervals |
| Per-pair PSNR not saved | Paired test is approximated |
| GPU nondeterminism | Small numerical differences possible |
| Semantic detection not run | No semantic hallucination results |
| Downstream tasks not trained | No task-driven evaluation results |
