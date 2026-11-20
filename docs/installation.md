# Installation

## Requirements

- Python 3.10 or newer
- PyTorch 2.x (CPU or CUDA build)
- A C-free scientific stack is not required; `rasterio` ships wheels for common platforms

## From PyPI

```bash
pip install edns
```

This installs the `edns` package and the `edns-sr` command.

## From source

```bash
git clone https://github.com/aimenachraff-jpg/EDN-s.git
cd EDN-s
pip install -e .
```

Editable installs are recommended when you intend to run the training and evaluation
scripts, because those scripts use the research tree (`src/models`, `src/training`, …).

## Development install

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `black`, `flake8`, `mypy` and `pre-commit`.

## Verify the installation

```bash
python -c "import edns; print(edns.__version__)"
# 1.0.2

python -c "from edns.models import EDSR; print(sum(p.numel() for p in EDSR().parameters()))"
# 1518764

edns-sr --help
```

Run the test suite:

```bash
python -m pytest tests/ -q
# 246 passed
```

## Model weights

The PyPI package contains code, not trained weights. Inference needs a checkpoint:

```bash
edns-sr input.tif --ckpt path/to/edsr_small_phase5_best.pt
```

Resolution order for the checkpoint:

1. `--ckpt` argument
2. `EDNS_CKPT` environment variable
3. `experiments/checkpoints/edsr_small_phase5_best.pt` in a source checkout
4. `checkpoints/edsr_small_phase5_best.pt`
5. `<sys.prefix>/share/edns/edsr_small_phase5_best.pt`

The checkpoint used for the reported results is 17.5 MB and is not committed to Git.
Place it at the expected path or pass `--ckpt` explicitly.

## Configuration

The default model config is bundled at `edns/configs/models/edsr_small.yaml`. Override it
with `--config` or `EDNS_CONFIG` if you use a different architecture.

## Troubleshooting

### `ModuleNotFoundError: No module named 'rasterio'`

Install the geospatial dependency separately:

```bash
pip install rasterio
```

On Windows, use a recent Python version (3.10–3.12) so a prebuilt wheel is available.

### `FileNotFoundError: Checkpoint not found`

No weights were found. Pass `--ckpt` or set `EDNS_CKPT`. See
[Model weights](#model-weights).

### `ValueError: Band B2 not found in stats`

The normalization statistics file is missing or malformed. Reinstall the package or pass
a valid stats file through the data API. The packaged copy is
`edns/data/normalization_stats.json`.

### CUDA out of memory

The model is small (1.52 M parameters), but tiled inference still allocates four-channel
float32 buffers. Reduce the tile size in `SRInference` or run on CPU:

```python
import torch
device = torch.device("cpu")
```

### `tkinter` is not available

`tkinter` is part of the Python standard library and is not installed through `pip`. The
optional `gui` extra was removed for this reason. Install it through your operating
system if you need a Tk application.
