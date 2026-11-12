# Reproducibility Documentation

**Project**: MASTER2-A1 — Hallucination-Aware SR Evaluation for ALSAT-2B  
**Standard**: Reproducibility for all claims in thesis and publications  
**Status**: Phase 2 Complete — Protocol and Implementation Verified

---

## 1. Reproducibility Principles

| Principle | Implementation |
|-----------|----------------|
| **Deterministic Execution** | Fixed seeds (42), `torch.use_deterministic_algorithms(True)`, CuDNN benchmark=False |
| **Version Control** | All code in Git, all configs in Git, data manifests hashed |
| **Environment Capture** | `requirements.txt`, `requirements-cloud.txt`, `pip freeze` logged per run |
| **Data Provenance** | Source URLs, acquisition dates, processing pipeline version, split hashes |
| **Compute Specification** | Hardware, driver versions, CUDA/cuDNN versions logged per run |
| **Statistical Rigor** | Bootstrap CIs, paired tests, multiple seeds, correction for multiple comparisons |

---

## 2. Environment Specification

### Local Development (CPU-only)
```bash
# Python
Python 3.10.11

# Core packages (from requirements.txt)
torch==2.3.0+cpu
torchvision==0.18.0+cpu
numpy==1.26.4
scipy==1.13.0
opencv-python==4.10.0
rasterio==1.3.10
geopandas==0.14.4
albumentations==1.4.15
omegaconf==2.3.0
hydra-core==1.3.2
mlflow==2.15.0
pywt==1.5.0
scikit-image==0.23.0
scikit-learn==1.5.0
pandas==2.2.2
tqdm==4.66.4
pyyaml==6.0.1

# Testing
pytest==9.1.1
pytest-cov==5.0.0
```

### Cloud Training (GPU)
```bash
# Additional packages (from requirements-cloud.txt)
torch==2.3.0+cu121
torchvision==0.18.0+cu121
timm==1.0.7
segmentation-models-pytorch==0.3.3
ultralytics==8.2.0
wandb==0.17.2
kaggle==1.7.4
einops==0.8.0
xformers==0.0.26
```

### Environment Capture Per Run
```python
# Automatically logged by SRTrainer
import subprocess
import torch
import sys

env_info = {
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "git_status": subprocess.check_output(["git", "status", "--porcelain"]).decode(),
}
```

---

## 3. Random Seeds & Determinism

### Global Seed Setting
```python
# In src/utils/set_seed.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For PyTorch 2.3+
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### Seed Usage by Component
| Component | Seed Source | Value |
|-----------|-------------|-------|
| Data splits | Config | 42 (fixed) |
| DataLoader shuffle | Trainer | epoch + 42 |
| Model init | Trainer | 42 |
| Degradation noise | Config | 42 |
| Augmentation | Albumentations | epoch + 42 |
| Evaluation bootstrap | Metrics | 42 |

---

## 4. Data Provenance

### Sentinel-2 (Primary Substitute)
| Attribute | Value |
|-----------|-------|
| **Source** | Microsoft Planetary Computer / Copernicus Data Space |
| **Product** | Sentinel-2 L2A (Level-2A, atmospherically corrected) |
| **Bands** | B2 (Blue), B3 (Green), B4 (Red), B8 (NIR) at 10m |
| **AOIs** | 8 regions (4 Algeria, 4 global validation) |
| **Temporal Range** | 2018-01-01 to 2024-12-31 |
| **Cloud Filter** | <20% cloud cover (via SCL band) |
| **Split Strategy** | Temporal: Train 2018-2022 / Val 2023 / Test 2024 |
| **Tile Size** | 256×256 at 10m (HR) → 64×64 at 10m (LR after ×4) |
| **Normalization** | Per-band quantile (2nd-98th percentile) computed on train split |

#### Split Manifest (`data/metadata/split_manifest.json`)
```json
{
  "version": "1.0",
  "created": "2026-11-12",
  "source": "planetary-computer",
  "query_hash": "sha256:...",
  "splits": {
    "train": {"tiles": 12450, "hash": "sha256:..."},
    "val": {"tiles": 1380, "hash": "sha256:..."},
    "test": {"tiles": 1420, "hash": "sha256:..."}
  },
  "normalization": {
    "method": "quantile",
    "percentiles": [2, 98],
    "stats_hash": "sha256:..."
  }
}
```

### PROBA-V MISR (MISR Benchmark)
| Attribute | Value |
|-----------|-------|
| **Source** | ESA Kelvin / PROBA-V MISR Challenge |
| **LR Data** | 300m multi-angle stacks (9-19 images per region) |
| **HR Reference** | 100m from overlapping orbits |
| **Regions** | 74 globally distributed |
| **Official Split** | 1160 train / 290 test (fixed by challenge) |
| **Registration** | NOT provided — sub-pixel shifts present |
| **Scoring** | cPSNR on clear pixels only (cloud mask provided) |
| **Config** | `configs/data/probav.yaml` |

### ALSAT-2B Simulated (Controlled)
| Attribute | Value |
|-----------|-------|
| **Source** | Sentinel-2 HR → Wald degradation with estimated NAOMI params |
| **Degradation Config** | `configs/degradation/alsat2b.yaml` |
| **Purpose** | Ablation studies, sensitivity analysis (H5, H6) |

---

## 5. Degradation Engine Reproducibility

### Wald Protocol Implementation (`src/degradation/alsat2b_degradation.py`)
All components deterministic with seed:

| Component | Parameters | Seed Handling |
|-----------|------------|---------------|
| **Trapezoidal SRF** | Band edges from NAOMI specs (fixed) | N/A (deterministic convolution) |
| **Gaussian MTF** | σ = 0.35 px (estimated from MTF@Nyquist>0.15) | N/A (deterministic convolution) |
| **Poisson-Gaussian Noise** | SNR = 20 dB (conservative), gain = 2.5, read_noise = 5e⁻ | `np.random.Generator(seed)` |
| **Quantization** | 12-bit → 10-bit (exact from specs) | N/A (deterministic) |
| **Sub-pixel Shift** | <0.3 px uniform (exact from specs) | `np.random.Generator(seed)` |
| **Downsampling** | Area-average ×4 (Bicubic for ideal) | N/A (deterministic) |

### Parameter Estimates (Documented in Protocol)
| Parameter | Value | Source | Label |
|-----------|-------|--------|-------|
| MTF σ | 0.35 px | Estimated from MTF@Nyquist > 0.15 | **ESTIMATED** |
| SNR | 20 dB | Conservative (spec: >100) | **ESTIMATED** |
| SRF Shape | Trapezoidal | Band edges from eoPortal | **DOCUMENTED** |
| Quantization | 12→10 bit | Downlink spec | **DOCUMENTED** |
| Registration | <0.3 px | Spec | **DOCUMENTED** |

---

## 6. Model Configuration Reproducibility

### EDSR-small (`configs/models/edsr_small.yaml`)
```yaml
name: edsr_small
scale: 4
n_resblocks: 16
n_feats: 64
res_scale: 1.0
in_channels: 4
out_channels: 4
rgb_range: 1.0
```

### SwinIR-base (`configs/models/swinir_base.yaml`)
```yaml
name: swinir_base
upscale: 4
img_size: 64
window_size: 8
depths: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
in_chans: 4
out_chans: 4
```

### TinySwinIR (local development)
```yaml
# Override in code: depths=[2]*6, embed_dim=60, num_heads=[2]*6
name: swinir_tiny
```

### Real-ESRGAN (`configs/models/realesrgan.yaml`)
```yaml
name: realesrgan
scale: 4
net_g:
  num_in_ch: 4
  num_out_ch: 4
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
```

---

## 7. Training Reproducibility

### Loss Functions
| Model | Loss | Weights | Implementation |
|-------|------|---------|----------------|
| EDSR-small | L1 | 1.0 | `SRLoss(loss_type='l1')` |
| SwinIR-base | L1 + LPIPS | 1.0 + 0.1 | `SRLoss(loss_type='l1+lpips', lpips_weight=0.1)` |
| Real-ESRGAN | L1 + Perceptual + GAN | 1.0 + 0.1 + 0.01 | `SRLoss(loss_type='l1+perceptual+gan', ...)` |

### Training Config (`configs/training/trainer_config.yaml`)
```yaml
trainer:
  epochs: 200
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.0
  optimizer: adam
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  
  scheduler:
    type: cosine
    eta_min: 1e-6
    warmup_epochs: 5
  
  loss:
    type: l1+lpips
    lpips_weight: 0.1
    perceptual_weight: 0.1
    gan_weight: 0.01
  
  validation_interval: 1
  checkpoint_interval: 10
  log_interval: 100
  
  # Reproducibility
  seed: 42
  deterministic: true
  num_workers: 4
  pin_memory: true
  
  # Mixed precision
  amp: true
  grad_clip: 1.0
```

---

## 8. Evaluation Reproducibility

### Metrics Implementation (`src/evaluation/metrics.py`)
All metrics deterministic, no randomness:

| Metric | Implementation | Reference |
|--------|----------------|-----------|
| PSNR | `10 * log10(MAX² / MSE)` | Standard |
| SSIM | 11×11 Gaussian window, σ=1.5 | Wang et al. 2004 |
| LPIPS | AlexNet features, v0.1 | Zhang et al. 2018 |
| SAM | Spectral Angle Mapper (radians) | Kruse et al. 1993 |
| ERGAS | Relative global spectral error | Wald et al. 1997 |
| SCI | Spectral Consistency Index (band-wise corr) | Protocol Section 4.2 |
| QNR | Quality with No Reference | Alparone et al. 2004 |
| HF Energy | Wavelet detail coefficients (db2) | Protocol Section 4.2 |
| Hallucination Index | Composite: spectral + structural + task | Protocol Section 4.3 |
| Structural Hallucination | Wavelet energy ratio SR/HR - 1 | Protocol Section 4.3 |

### Bootstrap Confidence Intervals
```python
# In metrics.bootstrap_ci
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() 
                      for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return np.percentile(boots, [alpha*100, (1-alpha)*100])
```

### Statistical Tests
| Comparison | Test | Correction |
|------------|------|------------|
| Pairwise model comparison | Wilcoxon signed-rank | Benjamini-Hochberg FDR (q=0.05) |
| Correlation (SCI vs task utility) | Spearman ρ | — |
| Multiple seeds | Paired t-test / Wilcoxon | — |

---

## 9. Complete Reproduction Checklist

### For Any Claim in Thesis/Paper
- [ ] Git SHA of code that produced result
- [ ] Exact config YAMLs used (in `configs/`)
- [ ] Random seed(s) used
- [ ] Data split manifest hash
- [ ] Hardware + software environment (from MLflow)
- [ ] Number of runs / seeds aggregated
- [ ] Statistical test + correction method
- [ ] Bootstrap CI parameters (n_bootstrap, seed)
- [ ] Raw metric values available (in `results/metrics/`)

### For Model Training
- [ ] Checkpoint saved (`best.pt`, `last.pt`)
- [ ] Training curve logged (MLflow)
- [ ] Validation metrics per epoch
- [ ] Final test evaluation on held-out split

### For Hallucination Detection
- [ ] Detector version (Git SHA of `src/hallucination/detection.py`)
- [ ] Thresholds used (configurable in protocol)
- [ ] Annotation protocol version (if human labels)
- [ ] Fleiss' κ calculation code

### For Downstream Tasks
- [ ] Task model architecture + weights source
- [ ] Fixed-backbone protocol enforced (weights frozen)
- [ ] Evaluation dataset + split
- [ ] Metric implementation version

---

## 10. Verification Commands

### Run All Tests
```bash
cd MASTER2-A1
python -m pytest tests/ -v
# Expected: 38 passed
```

### Verify Model Smoke Tests
```bash
python -c "
import torch
from src.models.edsr import EDSR
from src.models.swinir import TinySwinIR, SwinIR
from src.models.realesrgan import RealESRGAN

models = {
    'EDSR-small': EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4),
    'TinySwinIR': TinySwinIR(),
    'SwinIR-base': SwinIR(),
    'Real-ESRGAN': RealESRGAN(net_g={'num_in_ch':4,'num_out_ch':4,'num_feat':64,'num_block':23}, scale=4)
}
x = torch.randn(1, 4, 64, 64)
for name, m in models.items():
    m.eval()
    with torch.no_grad():
        y = m(x)
    p = sum(p.numel() for p in m.parameters())/1e6
    print(f'{name}: {p:.2f}M params, {x.shape} -> {y.shape}')
"
```

### Verify Degradation Engine
```bash
python -c "
import numpy as np
from src.degradation.alsat2b_degradation import create_degradation
from configs.degradation.alsat2b import ALSAT2B_DEGRADATION_CONFIG

hr = np.random.rand(4, 256, 256).astype(np.float32)
for name, cfg in ALSAT2B_DEGRADATION_CONFIG.items():
    d = create_degradation(cfg)
    lr = d(hr)
    print(f'{name}: {hr.shape} -> {lr.shape}, range: [{lr.min():.4f}, {lr.max():.4f}]')
"
```

### Verify Evaluation Metrics
```bash
python -c "
import numpy as np
from src.evaluation.metrics import evaluate_sr
np.random.seed(42)
hr = np.random.rand(4, 256, 256).astype(np.float32)
sr = hr + 0.05 * np.random.randn(4, 256, 256)
sr = np.clip(sr, 0, 1)
results = evaluate_sr(sr, hr)
for k, v in results.items():
    if isinstance(v, float):
        print(f'{k}: {v:.4f}')
    elif isinstance(v, dict):
        print(f'{k}: composite={v[\"composite\"]:.4f}')
"
```

### Verify Downstream Tasks
```bash
python -c "
import torch
from src.downstream.tasks import UNetResNet18, BuildingDetector, DeepLabV3Plus, SiameseUNet
x = torch.randn(2, 4, 256, 256)
for name, model in [('Cls', UNetResNet18(10,4)), ('Det', BuildingDetector(1,4)), 
                    ('Seg', DeepLabV3Plus(10,4)), ('CD', SiameseUNet(2,4))]:
    model.eval()
    with torch.no_grad():
        if name == 'CD':
            out = model(x, x)
        else:
            out = model(x)
    print(f'{name}: {out.shape if hasattr(out, \"shape\") else list(out.keys())}')"
```

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| CPU-only local dev | Slow training, tiny models only | Cloud GPU for actual training |
| ALSAT-2B real data unavailable | Cannot validate H1 FID/PSNR gap | Documented as limitation; 4-week gate |
| PlanetScope license pending | H5 cross-sensor limited to NAIP | NAIP + SimALSAT sufficient for core claims |
| No human annotators yet | H2 validation not run | Protocol ready; recruitment Week 9-10 |
| LPIPS requires AlexNet | Fixed version (v0.1) | Version pinned in requirements |

---

## 12. References

- `src/utils/set_seed.py` — Seed setting utility
- `src/training/trainer.py` — `SRTrainer` with deterministic training
- `src/evaluation/metrics.py` — All metrics + bootstrap CI
- `src/degradation/alsat2b_degradation.py` — Wald protocol implementation
- `configs/degradation/alsat2b.yaml` — Degradation parameters
- `research/final_experimental_protocol.md` — Complete protocol with corrections
- MLflow reproducibility: https://mlflow.org/docs/latest/tracking.html#reproducibility