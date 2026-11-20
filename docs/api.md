# API reference

The public API is the `edns` package. Everything documented here is importable from the
package root or from the submodule named below.

```python
import edns
print(edns.__version__)  # 1.0.2
```

---

## `edns` (package root)

| Name | Kind | Source |
|---|---|---|
| `EDSR` | class | `edns.models.edsr` |
| `EDSRPretrained` | class | `edns.models.edsr` |
| `create_edsr` | function | `edns.models.edsr` |
| `ResBlock` | class | `edns.models.edsr` |
| `Upsampler` | class | `edns.models.edsr` |
| `run_sr` | function | `edns.inference.app` |
| `load_model` | function | `edns.inference.app` |
| `SRInference` | class | `edns.inference.app` |
| `load_sentinel2_band` | function | `edns.data.io` |
| `normalize_band` | function | `edns.data.normalization` |
| `denormalize_band` | function | `edns.data.normalization` |
| `compute_psnr` | function | `edns.evaluation.metrics` |
| `compute_ssim` | function | `edns.evaluation.metrics` |
| `compute_sam` | function | `edns.evaluation.metrics` |
| `compute_ergas` | function | `edns.evaluation.metrics` |
| `compute_all_metrics` | function | `edns.evaluation.metrics` |

Hallucination classes are exported from `edns.hallucination`, not from the package root.

---

## Models

### `EDSR`

```python
EDSR(
    n_resblocks: int = 16,
    n_feats: int = 64,
    scale: int = 4,
    in_channels: int = 4,
    out_channels: int = 4,
    res_scale: float = 1.0,
    rgb_range: float = 1.0,
)
```

EDSR-small network. `forward(x)` maps `[N, in_channels, H, W]` to
`[N, out_channels, scale*H, scale*W]`.

```python
import torch
from edns.models import EDSR

model = EDSR()
x = torch.randn(1, 4, 64, 64)
with torch.no_grad():
    y = model(x)
print(y.shape)  # torch.Size([1, 4, 256, 256])
print(sum(p.numel() for p in model.parameters()))  # 1518764
```

The module also contains identity-initialised `sub_mean` and `add_mean` 1×1 convolutions
inherited from the EDSR reference implementation. They are included in the parameter count.

### `EDSRPretrained`

```python
EDSRPretrained(*args, pretrained_path: str | None = None, **kwargs)
```

Subclass of `EDSR` that loads a checkpoint at construction time and adapts 3-channel
pretrained weights to a 4-channel input when needed.

### `create_edsr`

```python
create_edsr(config: DictConfig) -> EDSR
```

Builds a model from an OmegaConf node with `n_resblocks`, `n_feats`, `scale`,
`in_channels` and `out_channels` keys.

### `ResBlock` / `Upsampler`

Building blocks used by `EDSR`: a conv–ReLU–conv residual block and a PixelShuffle
upsampling stage.

---

## Inference

### `run_sr`

```python
run_sr(
    input_path,
    output_path,
    ckpt_path: str | None = None,
    config_path: str | None = None,
    compute_hallucination: bool = False,
    hr_reference_path: str | Path | None = None,
    hallucination_report_path: str | Path | None = None,
) -> dict
```

Reads a GeoTIFF, runs tiled ×4 super-resolution, writes the output and optionally runs
hallucination detection.

Returns:

```python
{
    "output_path": "output_sr.tif",
    # present only when compute_hallucination=True
    "hallucination_report": {
        "input_path": ...,
        "output_path": ...,
        "hr_reference_path": ...,
        "hallucination": {
            "classification": "none" | "spectral" | "structural" | "textural" | "semantic" | "multiple",
            "detected_types": [...],
            "confidences": {...},
            "details": {...},
        },
        "summary": {
            "any_hallucination": bool,
            "detected_categories": [...],
            "max_confidence": float,
        },
    },
}
```

`hr_reference_path` is required when `compute_hallucination=True`; otherwise a
`ValueError` is raised. If the reference shape differs from the SR output, the reference
is resized per channel before comparison.

### `SRInference`

```python
SRInference(ckpt_path: str | None = None, config_path: str | None = None)
```

Holds the loaded model and the tiling settings (`tile_size=256`, `overlap=32`).

| Method | Signature | Returns |
|---|---|---|
| `preprocess` | `preprocess(arr)` | Normalised `[C, H, W]` array |
| `run` | `run(input_arr)` | SR array; uses a single pass for small inputs, tiled otherwise |
| `run_tiled` | `run_tiled(arr)` | SR array built from overlapping tiles with weighted blending |
| `run_from_file` | `run_from_file(input_path)` | `(sr_array, profile)` |
| `save` | `save(sr, output_path, profile)` | Writes the GeoTIFF; returns `None` |

```python
from edns.inference import SRInference

inf = SRInference("experiments/checkpoints/edsr_small_phase5_best.pt")
sr, profile = inf.run_from_file("input.tif")
inf.save(sr, "output_sr.tif", profile)
```

### `load_model`

```python
load_model(ckpt_path: str | None = None, config_path: str | None = None) -> EDSR
```

Builds the model, loads the checkpoint (accepting either a full checkpoint dict with
`model_state_dict` or a bare state dict), moves it to the selected device and sets
`eval()`. The model is cached in a module-level singleton guarded by a lock.

Checkpoint resolution order:

1. `ckpt_path` argument
2. `EDNS_CKPT` (the CLI passes this)
3. `experiments/checkpoints/edsr_small_phase5_best.pt`
4. `checkpoints/edsr_small_phase5_best.pt`
5. `<sys.prefix>/share/edns/edsr_small_phase5_best.pt`

### CLI

```bash
edns-sr INPUT [-o OUTPUT] [--ckpt PATH] [--config PATH]
              [--hallucination --hr REFERENCE] [--report REPORT] [-v]
```

| Argument | Meaning |
|---|---|
| `input` | Input GeoTIFF (4-band Sentinel-2) |
| `-o`, `--output` | Output path; defaults to `<input>_sr.tif` |
| `--ckpt` | Checkpoint path; defaults to `EDNS_CKPT` |
| `--config` | Model config path; defaults to `EDNS_CONFIG` |
| `--hallucination` | Enable hallucination detection |
| `--hr` | HR reference GeoTIFF (required with `--hallucination`) |
| `--report` | Write the hallucination report as JSON |
| `-v`, `--verbose` | Verbose logging |

---

## Data

### `load_sentinel2_band`

```python
load_sentinel2_band(filepath, band_idx: int = 1) -> (np.ndarray, dict)
```

Returns a `[H, W]` float32 array and the rasterio profile.

### `load_sentinel2_stack`

```python
load_sentinel2_stack(band_paths, band_order=["B2", "B3", "B4", "B8"]) -> (np.ndarray, dict)
```

Stacks bands into `[C, H, W]`. Raises `ValueError` if the path count does not match
`band_order`.

### `save_geotiff`

```python
save_geotiff(array, output_path, profile, dtype=None)
```

Writes an array with the supplied rasterio profile.

### `find_sentinel2_bands`

```python
find_sentinel2_bands(scene_dir, bands=["B2", "B3", "B4", "B8"]) -> dict
```

Locates band files inside a Sentinel-2 scene directory.

### `normalize_band` / `denormalize_band`

```python
normalize_band(band, band_name, stats=None, method="zscore")
denormalize_band(band, band_name, stats=None, method="zscore")
```

`band_name` must be one of `B2`, `B3`, `B4`, `B8`. `method` is `"zscore"` or
`"percentile"`. NaN pixels are preserved. `stats` defaults to the bundled training
statistics.

### `normalize_stack` / `denormalize_stack`

```python
normalize_stack(stack, band_names=["B2", "B3", "B4", "B8"], stats=None, method="zscore")
denormalize_stack(stack, band_names=["B2", "B3", "B4", "B8"], stats=None, method="zscore")
```

Apply the per-band transform to a `[C, H, W]` stack.

---

## Evaluation

### `compute_all_metrics`

```python
compute_all_metrics(sr, hr, baseline=None, scale=4) -> dict
```

**Argument order is SR first, HR second.** Accepts `[H, W]` or `[C, H, W]`.

Returned keys:

| Key | Present for | Notes |
|---|---|---|
| `psnr` | always | dB |
| `mse` | always | |
| `mae` | always | |
| `ssim` | always | |
| `ergas` | always (`None` for 2D) | |
| `sam` | always (`None` for 2D) | radians |
| `sci` | always (`None` for 2D) | |
| `wavelet_hh_energy` | always | diagnostic |
| `glcm_entropy` | always | diagnostic |
| `tus_psnr`, `tus_ssim` | when `baseline` is given | heuristic; see note |

> The TUS keys are a composite heuristic. The formula expects a bounded task metric; for
> PSNR the HR-vs-HR reference is infinite, so these values are diagnostic only and are not
> used in any reported result.

### Individual metrics

```python
compute_psnr(hr, sr, max_val=1.0) -> float
compute_mse(hr, sr) -> float
compute_mae(hr, sr) -> float
compute_ssim(hr, sr, max_val=1.0, window_size=11, sigma=1.5, channel_axis=None) -> float
compute_sam(hr, sr, eps=1e-8) -> float          # radians
compute_ergas(hr, sr, scale=4) -> float
compute_spectral_consistency_index(hr, sr) -> float
compute_wavelet_hh_energy(img, wavelet="db1", level=1) -> float
compute_glcm_entropy(img) -> float
compute_task_utility_score(hr_metric, sr_metric, baseline_metric, higher_is_better=True) -> float
```

The metric functions accept NumPy arrays or torch tensors. See
[Evaluation](evaluation.md) for interpretation guidance.

---

## Hallucination detection

Import from `edns.hallucination`.

### `HallucinationDetector`

```python
HallucinationDetector(thresholds: HallucinationThresholds | None = None)
```

| Method | Returns |
|---|---|
| `detect_spectral(sr, hr, band_names=None)` | Per-category dict |
| `detect_structural(sr, hr)` | Per-category dict |
| `detect_textural(sr, hr)` | Per-category dict |
| `detect_semantic(sr, hr, task_model, task_metric)` | Per-category dict |
| `detect_all(sr, hr, task_model=None, task_metric=None, band_names=None)` | Composite dict |

`detect_all` returns:

```python
{
    "classification": "none" | "spectral" | "structural" | "textural" | "semantic" | "multiple",
    "detected_types": ["textural"],
    "confidences": {"spectral": 0.01, "structural": 0.0, "textural": 0.05, "semantic": 0.0},
    "details": {
        "spectral": {...},
        "structural": {...},
        "textural": {...},
        "semantic": {"detected": False, "reason": "no_task_model"},
    },
}
```

### `HallucinationThresholds`

| Field | Default | Meaning |
|---|---|---|
| `sam_threshold_rad` | `0.175` | ≈10° SAM limit |
| `ndvi_min` / `ndvi_max` | `-1.0` / `1.0` | Valid NDVI range |
| `structural_energy_ratio` | `3.0` | Wavelet HH energy ratio |
| `structural_relative_energy` | `0.15` | Relative energy limit |
| `textural_entropy_diff` | `0.20` | 20% GLCM entropy mismatch |
| `semantic_task_drop` | `0.15` | 15% task-metric drop |

### `HallucinationType`

Enum: `SPECTRAL`, `STRUCTURAL`, `TEXTURAL`, `SEMANTIC`, `NONE`, `MULTIPLE`.

### `AnnotationSample`

```python
AnnotationSample(
    sample_id: str,
    sr_image: np.ndarray,
    hr_image: np.ndarray,
    model_name: str,
    patch_coords: tuple[int, int, int, int],  # (y, x, h, w)
    annotations: list[HallucinationType] | None = None,
    annotator_id: str = "",
    notes: str = "",
)
```

### `AnnotationProtocol`

```python
AnnotationProtocol(n_annotators: int = 3, target_kappa: float = 0.7)
```

| Method | Purpose |
|---|---|
| `add_sample(sample)` | Register an `AnnotationSample` |
| `record_annotation(sample_id, annotator_id, labels)` | Record one annotator’s labels |
| `compute_fleiss_kappa()` | Inter-annotator agreement |
| `generate_report()` | Summary dict |

### `validate_hallucination_metrics`

Helper for comparing detector output against human labels (precision, recall, F1).
No human annotation study was conducted for the reported results.

---

## Utilities

```python
from edns.utils import setup_logging, count_parameters, format_time
```

`edns.utils.helpers` also provides `format_bytes`, `tensor_to_numpy`, `numpy_to_tensor`,
`ensure_dir` and `Timer`.
