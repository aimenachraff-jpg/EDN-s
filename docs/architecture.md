# Architecture

This document describes the implemented modules, their responsibilities and the data
flow. It does not propose a new architecture: the network is EDSR-small.

## Two trees

| Tree | Purpose | Shipped on PyPI |
|---|---|---|
| `src/edns/` | Installable package: inference, data I/O, metrics, hallucination detection, CLI | Yes |
| `src/models/`, `src/training/`, `src/preprocessing/`, `src/degradation/`, `src/downstream/`, `src/evaluation/`, `src/inference/` | Research and training code used to produce the experiments | No |

The packaged tree is a curated subset. Keep behaviour consistent when editing shared
model logic.

## Package modules (`src/edns/`)

### `edns.models`

- `EDSR` — the network.
- `EDSRPretrained` — loads BasicSR-style checkpoints and adapts 3-channel weights to
  4-channel input.
- `create_edsr(config)` — factory from a config object.
- `ResBlock` — conv–ReLU–conv with a residual connection.
- `Upsampler` — PixelShuffle stages.

### `edns.inference`

- `SRInference` — high-level inference object.
  - `preprocess(arr)` — band handling and z-score normalisation.
  - `run(arr)` — single-pass or tiled inference.
  - `run_tiled(arr)` — 256 px tiles, 32 px overlap, weighted blending.
  - `run_from_file(path)` — read a GeoTIFF, return the SR array and profile.
  - `save(sr, path, profile)` — write a GeoTIFF.
- `load_model(ckpt_path, config_path)` — build and load the model.
- `run_sr(...)` — convenience function used by the CLI and the public API.
- `main()` — `edns-sr` entry point.

### `edns.data`

- `io.py` — `load_sentinel2_band`, `load_sentinel2_stack`, `save_geotiff`,
  `find_sentinel2_bands`.
- `normalization.py` — `normalize_band`, `denormalize_band`, `normalize_stack`,
  `denormalize_stack`, `load_normalization_stats`.
- `normalization_stats.json` — bundled training statistics.

### `edns.evaluation`

- `compute_psnr`, `compute_mse`, `compute_mae`, `compute_ssim`, `compute_sam`,
  `compute_ergas`, `compute_spectral_consistency_index`, `compute_wavelet_hh_energy`,
  `compute_glcm_entropy`, `compute_task_utility_score`, `compute_all_metrics`.

### `edns.hallucination`

- `HallucinationType` — enum of categories.
- `HallucinationThresholds` — configurable thresholds.
- `HallucinationDetector` — `detect_spectral`, `detect_structural`, `detect_textural`,
  `detect_semantic`, `detect_all`.
- `AnnotationSample`, `AnnotationProtocol` — human annotation and Fleiss’ κ.
- `validate_hallucination_metrics` — metric validation helper.

### `edns.utils`

- `setup_logging`, `count_parameters`, `format_time`, `format_bytes`,
  `tensor_to_numpy`, `numpy_to_tensor`, `ensure_dir`, `Timer`.

## Research modules (`src/`)

| Module | Responsibility |
|---|---|
| `models/` | EDSR, SwinIR, Real-ESRGAN, base classes |
| `training/` | `SRDataset`, `SRTrainer`, losses, AMP, MLflow |
| `preprocessing/` | Tiling, spatial buffers, normalisation |
| `degradation/` | ALSAT-2B Wald simulation |
| `evaluation/` | Research metric implementation |
| `inference/` | Bicubic baseline and inference helpers |
| `downstream/` | Task-model interfaces (classification, detection, segmentation, change detection) |
| `data/` | PROBA-V loader, Sentinel-2 downloader, validation |

## Data flow

### Training

```
Sentinel-2 tiles (HR, 1024×1024)
        │
        ├── Wald degradation ──► LR (256×256)
        │
        └── paired patches (48×48 LR / 192×192 HR)
                    │
                EDSR-small
                    │
                L1 loss
```

### Inference

```
LR GeoTIFF (4 bands)
        │
   z-score normalisation
        │
   reflect padding to tile multiple
        │
   256 px tiles + 32 px overlap
        │
      EDSR-small ×4
        │
   weighted blending + crop
        │
SR GeoTIFF (4 bands, ×4 size, adjusted affine)
```

### Evaluation

```
SR array ──┬── quality metrics ◄── HR reference
           │
           └── hallucination detector ◄── HR reference
                    │
          spectral / structural / textural
          semantic (requires task model)
```

## Design decisions

| Decision | Reason |
|---|---|
| EDSR-small instead of a larger model | CPU inference is practical; the parameter budget matches the available hardware |
| Four-channel input | Preserves NIR, which matters for vegetation and spectral indices |
| No batch normalisation | Follows the EDSR reference design |
| Tiled inference | Keeps memory bounded for large GeoTIFFs |
| Separate packaged and research trees | The package stays small and installable; training keeps its full dependencies |
| Reference-based hallucination | A reference-free detector cannot distinguish invention from uncertainty |
| Semantic detection behind an interface | A task model is a separate training problem; the interface keeps the option open |

## Extension points

- **New model** — implement the same `forward(x) -> y` contract and register it in
  `src/models/__init__.py`; add a config under `configs/models/`.
- **New metric** — add a function in `src/edns/evaluation/metrics.py` and export it from
  `edns.evaluation`.
- **New hallucination signal** — add a `detect_*` method to `HallucinationDetector` and
  include it in `detect_all`.
- **New task model** — implement `TaskModel` in `src/downstream/tasks.py` and pass it to
  `detect_semantic`.
