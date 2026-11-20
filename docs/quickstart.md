# Quick start

This walkthrough takes a four-band GeoTIFF and produces a ×4 super-resolved GeoTIFF.
It assumes the package is installed and a checkpoint is available.

## 1. Check the input

```python
import rasterio

with rasterio.open("input.tif") as src:
    print(src.count, src.width, src.height, src.dtypes, src.crs)
```

Expected: 4 bands, ordered B2, B3, B4, B8. A 256×256 input produces a 1024×1024 output.

## 2. Run the CLI

```bash
edns-sr input.tif -o output_sr.tif --ckpt experiments/checkpoints/edsr_small_phase5_best.pt
```

Expected output:

```
Processing input.tif -> output_sr.tif
Done in ...s. Saved to output_sr.tif
```

## 3. Run the Python API

```python
from edns.inference import run_sr

result = run_sr(
    input_path="input.tif",
    output_path="output_sr.tif",
    ckpt_path="experiments/checkpoints/edsr_small_phase5_best.pt",
)
print(result)
```

## 4. Add hallucination detection

Hallucination detection needs an HR reference aligned with the SR output:

```bash
edns-sr input.tif -o output_sr.tif \
  --ckpt experiments/checkpoints/edsr_small_phase5_best.pt \
  --hallucination --hr reference_hr.tif --report report.json
```

The report contains spectral, structural and textural results. The semantic category is
skipped unless a task model is supplied.

## 5. Verify without a checkpoint

You can still check that the model builds and runs:

```python
import torch
from edns.models import EDSR

model = EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4, out_channels=4)
x = torch.randn(1, 4, 64, 64)
with torch.no_grad():
    y = model(x)
print(x.shape, "->", y.shape)
# torch.Size([1, 4, 64, 64]) -> torch.Size([1, 4, 256, 256])
```

## 6. Inspect the metrics API

```python
import numpy as np
from edns.evaluation import compute_all_metrics

hr = np.random.rand(4, 256, 256).astype(np.float32)
sr = np.clip(hr + 0.02 * np.random.randn(4, 256, 256), 0, 1).astype(np.float32)

metrics = compute_all_metrics(sr, hr, scale=4)
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")
```

## Next steps

- [Dataset preparation](dataset.md)
- [Training](training.md)
- [Evaluation](evaluation.md)
- [API reference](api.md)
