# Training

Training code lives in the research tree (`src/training/trainer.py`) because it depends on
the model and data modules under `src/models/` and `src/preprocessing/`. The installable
`edns` package is for inference and evaluation.

## Configuration used for the reported model

| Setting | Value |
|---|---|
| Model | EDSR-small |
| Residual blocks | 16 |
| Feature channels | 64 |
| Scale | ×4 |
| Input / output channels | 4 / 4 |
| Batch size | 16 |
| Patch size | 48×48 |
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Loss | L1 |
| Epochs | 300 |
| Mixed precision | fp16 AMP |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | Kaggle 2× T4 (GPU 0) |
| Time | 6.82 h / 78,020 steps |
| Best validation PSNR | 25.169 dB (patch-level, epoch 239) |

The YAML equivalent is [`configs/models/edsr_small.yaml`](../configs/models/edsr_small.yaml).

## Data prerequisites

`SRDataset` expects paired `.npy` arrays:

```
pairs/
├── train/
│   ├── lr/   # [4, 256, 256] float32
│   └── hr/   # [4, 1024, 1024] float32
└── val/
    ├── lr/
    └── hr/
```

The arrays are produced by [`scripts/prepare_pairs.py`](../scripts/prepare_pairs.py).
See [Dataset](dataset.md) for the full preparation pipeline.

> The Phase 5 run used a variant that precomputed LR arrays and streamed HR tiles with
> windowed rasterio reads. The generic `SRDataset` below loads both sides from `.npy`.
> Use whichever variant matches your storage budget.

## Minimal training loop

```python
import torch
from torch.utils.data import DataLoader

from src.models.edsr import EDSR
from src.training.trainer import SRTrainer, SRDataset, TrainingConfig, get_model_config

train_ds = SRDataset("pairs/train/lr", "pairs/train/hr", patch_size=48, scale=4)
val_ds = SRDataset("pairs/val/lr", "pairs/val/hr", patch_size=48, scale=4, augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

model = EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4, out_channels=4)
config = get_model_config("edsr", TrainingConfig())
config["seed"] = 42
config["deterministic"] = True

trainer = SRTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_mlflow=False,
    experiment_name="edsr_small",
)
trainer.train()
```

`SRTrainer.train()` accepts `max_epochs` and `max_iters` for short runs:

```python
trainer.train(max_epochs=5)
```

## Kaggle execution

The run reported in this repository used
[`experiments/protocols/kaggle_train_template.py`](../experiments/protocols/kaggle_train_template.py)
as a script kernel. The template:

1. Sets up the Kaggle input layout.
2. Builds DataLoaders over the uploaded tiles.
3. Creates the EDSR-small model.
4. Trains with AMP and validation each epoch.
5. Saves the best checkpoint and a training summary.

The dataset and code were uploaded as private Kaggle datasets. The training notebook was
pushed headlessly because interactive sessions time out.

## Checkpoints

`SRTrainer.save_checkpoint(name)` writes:

```python
{
    "epoch": ...,
    "global_step": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_val_psnr": ...,
    "config": ...,
}
```

The inference loader accepts either the full checkpoint or a bare state dict:

```python
state = torch.load(path, map_location="cpu", weights_only=False)
if "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
```

PyTorch 2.6 changed the default of `weights_only` to `True`. The EDN-S checkpoints are
trusted local files and must be loaded with `weights_only=False`.

## Resuming

```python
trainer.load_checkpoint("experiments/checkpoints/edsr_small_phase5_best.pt")
trainer.train()
```

## Monitoring

- MLflow logging is enabled by default (`use_mlflow=True`); pass `False` for a local run.
- Validation PSNR is computed per epoch when a validation loader is supplied.
- The best checkpoint is selected on validation PSNR.

## Hardware notes

- The model has 1.52 M parameters and trains comfortably on a single 16 GB GPU.
- CPU training is possible for smoke tests but not for the full 300-epoch run.
- The reported run used GPU 0 only; GPU 1 was idle.

## Troubleshooting

### Loss becomes NaN

- Check that LR/HR arrays are finite.
- Lower the learning rate.
- Disable AMP for a diagnostic run.

### Validation PSNR does not improve

- Confirm the LR/HR pairs are aligned.
- Confirm the normalisation statistics match the data.
- Check that the model input is 4-channel and in the expected band order.

### `weights_only` error when loading

Pass `weights_only=False` to `torch.load` for trusted local checkpoints.

### Out of memory

Reduce `batch_size` or `patch_size`. The model itself is small; the HR patches dominate
memory.
