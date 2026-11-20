# Evaluation

EDN-S evaluates a super-resolved image on two levels: reconstruction quality and
hallucination risk. Both are reference-based.

## Quality metrics

All metrics are implemented in [`src/edns/evaluation/metrics.py`](../src/edns/evaluation/metrics.py)
and re-exported from `edns.evaluation`.

| Function | Metric | Range | Better |
|---|---|---|---|
| `compute_psnr(hr, sr)` | Peak signal-to-noise ratio | dB | Higher |
| `compute_mse(hr, sr)` | Mean squared error | ≥ 0 | Lower |
| `compute_mae(hr, sr)` | Mean absolute error | ≥ 0 | Lower |
| `compute_ssim(hr, sr)` | Structural similarity | [0, 1] | Higher |
| `compute_sam(hr, sr)` | Spectral angle mapper | rad | Lower |
| `compute_ergas(hr, sr, scale)` | Relative global spectral error | ≥ 0 | Lower |
| `compute_spectral_consistency_index(hr, sr)` | Band-wise correlation | [−1, 1] | Higher (≈1) |
| `compute_wavelet_hh_energy(img)` | High-frequency detail energy | ≥ 0 | Diagnostic |
| `compute_glcm_entropy(img)` | Texture entropy | ≥ 0 | Diagnostic |
| `compute_task_utility_score(...)` | Composite utility | — | Higher |

`compute_all_metrics(sr, hr, baseline=None, scale=4)` returns a dictionary. Note the
argument order: **SR first, HR second**.

```python
from edns.evaluation import compute_all_metrics

metrics = compute_all_metrics(sr, hr, scale=4)
print(metrics["psnr"], metrics["ssim"], metrics["sam"])
```

### Interpreting the metrics

- **PSNR** rewards pixel fidelity. It is useful for comparing models on the same data but
  says little about spectral realism.
- **SSIM** is more sensitive to structure than PSNR.
- **SAM** is the angle between spectral vectors. A low SAM means the spectral shape is
  preserved even if brightness differs.
- **ERGAS** combines relative error across bands; it is scale-aware and sensitive to
  spectral distortion.
- **SCI** measures band-wise correlation with the reference. A value near 1 means the
  inter-band relationships are preserved.
- **Wavelet HH energy** and **GLCM entropy** describe high-frequency and texture content.
  They are diagnostic: the “correct” value depends on the scene.
- **TUS** is a composite heuristic, not a validated perceptual score.

## Hallucination detection

Implemented in [`src/edns/hallucination/detection.py`](../src/edns/hallucination/detection.py).

### Thresholds

| Category | Signal | Default threshold |
|---|---|---|
| Spectral | SAM and band-correlation deviation | SAM > 0.175 rad (≈10°); NDVI outside [−1, 1] |
| Structural | Wavelet HH energy in homogeneous regions | Energy ratio > 3σ; relative energy > 0.15 |
| Textural | GLCM entropy mismatch | > 20% difference |
| Semantic | Downstream task drop | > 15% versus HR |

Thresholds are configurable through `HallucinationThresholds`.

### Usage

```python
from edns.hallucination import HallucinationDetector

detector = HallucinationDetector()
result = detector.detect_all(sr, hr)

print(result["classification"])   # none | spectral | structural | textural | multiple
print(result["detected_types"])
print(result["confidences"])
```

With a task model:

```python
result = detector.detect_all(
    sr, hr,
    task_model=task_model,
    task_metric=task_metric,
)
```

If no task model is supplied, the semantic entry is marked as skipped with
`reason="no_task_model"`.

### Output structure

```python
{
    "classification": "textural",
    "detected_types": ["textural"],
    "confidences": {"spectral": 0.01, "structural": 0.0, "textural": 0.05, "semantic": 0.0},
    "details": {
        "spectral": {...},
        "structural": {...},
        "textural": {...},
        "semantic": {...},
    },
}
```

### Annotation protocol

`AnnotationProtocol` supports a multi-annotator study:

```python
from edns.hallucination import AnnotationProtocol, AnnotationSample, HallucinationType

protocol = AnnotationProtocol(n_annotators=3, target_kappa=0.7)
protocol.add_sample(AnnotationSample(
    sample_id="tile_0001",
    sr_image=sr,
    hr_image=hr,
    model_name="edsr_small",
    patch_coords=(0, 0, 256, 256),
))
protocol.record_annotation("tile_0001", "annotator_1", [HallucinationType.TEXTURAL])
report = protocol.generate_report()
print(report["fleiss_kappa"])
```

No human annotation study was conducted for the reported results.

## Reproducing the reported evaluation

```bash
# Quality on 30 validation pairs
python scripts/eval_bicubic_baseline.py

# Hallucination on clean test tiles
python scripts/phase6_hallucination.py

# Statistical tests
python scripts/phase8_statistical_analysis.py
```

Outputs are written to `results/metrics/`.

## Limitations

- Hallucination detection requires an HR reference.
- Semantic detection requires a task model and was not evaluated.
- Structural detection can favour bicubic slightly (2 tiles vs 0 in the reported run).
- The metric suite is not a substitute for a task-driven evaluation.
