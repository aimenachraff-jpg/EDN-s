# Super-Resolution Model Architecture Comparison for Satellite Imagery

## Executive Summary
**Target**: Lightweight, CPU-trainable baselines + SOTA for cloud GPU training
**Constraint**: Local dev on CPU (Intel Iris Xe) — models must run inference <30s on 256×256 patch
**Strategy**: 3 tiers — Tiny (local dev), Standard (cloud training), SOTA (final comparison)

---

## Tier 1: Tiny Models (Local CPU Development)

| Model | Params | MACs (256²×3) | Inference CPU | Arch Type | PyTorch Impl |
|-------|--------|---------------|---------------|-----------|--------------|
| **ESRNN** (Efficient Sub-pixel) | ~50K | ~0.5G | ~0.5s | CNN + PixelShuffle | Custom (simple) |
| **FSRCNN** | ~12K | ~0.1G | ~0.2s | CNN + Deconv | `torchvision` / custom |
| **Small ESRGAN** (RRDB ×3) | ~200K | ~2G | ~2s | GAN (RRDB) | `basicsr` |
| **Tiny SwinIR** (depth=2, heads=2) | ~150K | ~1.5G | ~1.5s | Swin Transformer | `swinir` repo |

**Recommendation**: **FSRCNN** for absolute minimum; **Tiny SwinIR** for transformer baseline.

---

## Tier 2: Standard Models (Cloud GPU Training — Primary Experiments)

### 2.1 CNN-Based (Established, Strong Baselines)

| Model | Paper | Params | Scale | Key Features | Code |
|-------|-------|--------|-------|--------------|------|
| **EDSR** | Lim et al. CVPR 2017 | 43M (baseline) / 1.5M (small) | ×2,×3,×4 | Residual blocks, no BN, L1 loss | `EDSR-PyTorch`, `basicsr` |
| **RCAN** | Zhang et al. ECCV 2018 | 15.6M | ×2,×3,×4 | Channel attention (CA), residual groups | `RCAN`, `basicsr` |
| **RNAN** | Zhang et al. CVPR 2019 | ~12M | ×2,×3,×4 | Residual non-local attention | `RNAN` |
| **SAN** | Dai et al. CVPR 2019 | ~16M | ×2,×3,×4 | Second-order attention | `SAN` |

**EDSR-small** (64 filters, 16 blocks): ~1.5M params — **RECOMMENDED primary CNN baseline**

### 2.2 Transformer-Based (SOTA for PSNR/SSIM)

| Model | Paper | Params | Scale | Key Features | Code |
|-------|-------|--------|-------|--------------|------|
| **SwinIR** | Liang et al. ICCV 2021 | 11.9M (base) | ×2,×3,×4,×8 | Swin Transformer, RSTB, window attention | `swinir` (official) |
| **Swin2SR** | Cao et al. CVPR 2023 | ~11M | Arbitrary | SwinV2, arbitrary scale | `swin2sr` |
| **HAT** | Chen et al. CVPR 2023 | ~23M | ×2,×3,×4 | Hybrid attention (channel+window), overlapping | `HAT` |
| **DAT** | Guo et al. ICCV 2023 | ~14M | ×2,×3,×4 | Dual attention (channel+spatial) | `DAT` |
| **SRT** | Wang et al. 2024 | ~12M | ×2,×3,×4 | Simple recurrent transformer | `SRT` |

**SwinIR (base)** — **RECOMMENDED primary transformer baseline** (widely used, reproducible, good remote sensing results)

### 2.3 GAN-Based (Perceptual Quality, Hallucination Risk)

| Model | Paper | Params (G) | Scale | Key Features | Code |
|-------|-------|------------|-------|--------------|------|
| **ESRGAN** | Wang et al. ECCV 2018 | 16.7M | ×4 | RRDB, Relativistic GAN, VGG/Perceptual loss | `ESRGAN`, `basicsr` |
| **Real-ESRGAN** | Wang et al. ICCV 2021 | 16.7M | ×4 | Synthetic degradation training, real-world | `Real-ESRGAN` (official) |
| **BSRGAN** | Zhang et al. CVPR 2021 | 16.7M | ×4 | Blind SR, degradation prediction | `BSRGAN` |
| **SwinIR-GAN** | Liang et al. 2021 | ~12M | ×4 | SwinIR generator + discriminator | Custom |

**Real-ESRGAN** — **RECOMMENDED for hallucination studies** (designed for real-world, known hallucination behaviors)

### 2.4 Diffusion / Generative (Highest Hallucination Risk)

| Model | Paper | Params | Scale | Key Features | Code |
|-------|-------|--------|-------|--------------|------|
| **StableSR** | Wang et al. CVPR 2023 | ~1B (LDM) | Arbitrary | Latent diffusion, time-aware conditioning | `StableSR` |
| **DiffIR** | Kong et al. ICCV 2023 | ~500M | ×4 | Diffusion for IR, controllable | `DiffIR` |
| **SwinIR-Diff** | Various 2024 | ~ | ×4 | SwinIR + diffusion refinement | Custom |
| **S2DiffSR** | 2024 (RS-specific) | ~ | ×4,×10 | Sentinel-2 specific diffusion | Custom |

**Note**: Diffusion models require significant GPU (16GB+ VRAM), long training. Include **only if cloud GPU budget allows**.

---

## Tier 3: Remote Sensing Specific Models

| Model | Paper | Target | Params | Key Innovation | Code |
|-------|-------|--------|--------|----------------|------|
| **PanNet** | Yang et al. 2017 | Pansharpening | ~1M | Deep PAN-MS fusion | `PanNet` |
| **TFNet** | Yao et al. 2018 | Pansharpening | ~2M | Two-stream fusion | `TFNet` |
| **FusionNet** | Various | Cross-sensor | ~ | Domain adaptation | Custom |
| **GeoSR** | 2026 (GeoSR-Bench) | RS downstream | ~ | Benchmark, not model | `GeoSR-Bench` |
| **SpatialSR** | 2026 (arXiv:2611.13046) | Arbitrary scale RS | ~ | Resolution-centric pipeline | `SpatialSR` |
| **SALD** | 2026 (arXiv:2604.25319) | Hallucination suppression | ~ | Semantic-aware latent diffusion | `SALD` |

**Key Finding**: Few RS-specific SR architectures — most use adapted natural image models. **Opportunity for contribution**.

---

## Model Selection Matrix for This Thesis

| Role | Model | Justification |
|------|-------|---------------|
| **CNN Baseline** | EDSR-small (1.5M) | Fast, reproducible, strong PSNR, standard benchmark |
| **Transformer Baseline** | SwinIR (base, 11.9M) | SOTA PSNR/SSIM, widely reproduced, good RS transfer |
| **GAN / Perceptual** | Real-ESRGAN (16.7M) | Real-world trained, known hallucination patterns, RS-used |
| **Diffusion (if budget)** | StableSR or S2DiffSR | Highest hallucination risk, SOTA perceptual, 2024+ |
| **RS-Specific** | SpatialSR (2026) | Arbitrary scale, RS-native, very recent |
| **Hallucination-Suppressed** | SALD (2026) | Explicitly addresses hallucination for downstream tasks |

---

## Training Configurations (Cloud GPU — Kaggle/Colab/RunPod)

### Hardware Targets
| Platform | GPU | VRAM | Time Limit | Cost |
|----------|-----|------|------------|------|
| **Kaggle** | T4×2 / P100×2 | 16GB / 32GB | 30h/week | **FREE** |
| **Colab Pro** | T4 / V100 / A100 | 16-40GB | 24h | $10/mo |
| **RunPod** | RTX 3090/4090/A100 | 24-80GB | Unlimited | $0.2-3/h |

**Recommendation**: **Kaggle** (free, persistent, T4×2 sufficient for our models)

### Training Recipes (Per Model)

#### EDSR-small
```yaml
model: EDSR
n_resblocks: 16
n_feats: 64
scale: 4
batch_size: 16
patch_size: 48
lr: 1e-4
epochs: 300
loss: L1
optimizer: Adam (beta1=0.9, beta2=0.999)
lr_scheduler: CosineAnnealingLR (T_max=300)
augment: flip, rot90
precision: fp16 (AMP)
```

#### SwinIR (base)
```yaml
model: SwinIR
upscale: 4
img_size: 64
window_size: 8
depths: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
mlp_ratio: 2
batch_size: 16
patch_size: 64
lr: 2e-4
epochs: 200
loss: L1 + 0.1*Perceptual (VGG)
optimizer: AdamW (weight_decay=1e-4)
lr_scheduler: CosineAnnealingRestart (periods=[100,100])
augment: flip, rot90, color_jitter
precision: fp16 (AMP)
```

#### Real-ESRGAN
```yaml
model: RealESRGAN
net_g: RRDBNet (num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
scale: 4
batch_size: 16
patch_size: 128  # larger for GAN
lr_g: 1e-4
lr_d: 1e-4
epochs: 400k iters
loss_g: L1 + 1.0*Perceptual + 0.1*GAN (Relativistic)
loss_d: Relativistic GAN
optimizer: Adam (beta1=0.9, beta2=0.99)
lr_scheduler: CosineAnnealingRestart
degradation: RealESRGAN degradation pipeline (blur, noise, JPEG, resize)
precision: fp16 (AMP)
```

---

## Evaluation Protocol (All Models)

### Datasets
| Dataset | Role | Split |
|---------|------|-------|
| **Sentinel-2 (Algeria AOIs)** | Primary test | 70/15/15 train/val/test |
| **PROBA-V MISR** | MISR benchmark | Official split |
| **S2→NAIP Cross-sensor** | Domain gap | Held-out ROIs |
| **Synthetic ALSAT-2B** | Target simulation | Held-out |

### Metrics (Tiered)

#### Tier 1: Reference-Based (Full-Reference)
| Metric | Target | Implementation |
|--------|--------|----------------|
| **PSNR** | >30 dB (×4) | `skimage.metrics.peak_signal_noise_ratio` |
| **SSIM** | >0.85 (×4) | `skimage.metrics.structural_similarity` |
| **LPIPS** | <0.15 (×4) | `lpips.LPIPS(net='alex')` |
| **SAM** (Spectral Angle Mapper) | <5° | Custom (RS-specific) |
| **ERGAS** | <3 | Custom (RS-specific) |
| **QNR** (No-ref pansharpening) | >0.7 | Custom |

#### Tier 2: No-Reference Quality (NR-IQA)
| Metric | Purpose | Implementation |
|--------|---------|----------------|
| **BRISQUE** | General quality | `pyiqa` |
| **NIQE** | Naturalness | `pyiqa` |
| **MUSIQ** | Transformer-based | `pyiqa` |
| **SenseBench** | RS-specific NR | Custom (from paper) |

#### Tier 3: Hallucination Detection
| Metric | Principle | Implementation |
|--------|-----------|----------------|
| **Spectral Consistency** | Band-to-band correlation preservation | Custom |
| **Texture Hallucination** | High-freq energy in smooth regions | Custom (wavelet) |
| **Semantic Hallucination** | Classifier confidence on fake structures | Downstream detector |
| **FID / KID** | Distribution shift | `clean-fid` |

#### Tier 4: Downstream Task Performance
| Task | Metric | Protocol |
|------|--------|----------|
| **Land Cover Classification** | mIoU, OA, Kappa | Fine-tune UNet/DeepLab on SR output |
| **Object Detection** | mAP@0.5 | Fine-tune YOLO/RT-DETR on SR output |
| **Change Detection** | F1, IoU | Siamese network on SR pairs |
| **Segmentation** | mIoU | Fixed backbone, train head only |

---

## Reproducibility Requirements

### Version Pinning (requirements.txt / environment.yaml)
```yaml
# Core
torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0

# SR Repos (git commit hashes)
basicsr @ git+https://github.com/XPixelGroup/BasicSR.git@<commit>
swinir @ git+https://github.com/JingyunLiang/SwinIR.git@<commit>
realesrgan @ git+https://github.com/xinntao/Real-ESRGAN.git@<commit>

# Evaluation
lpips==0.1.4
pyiqa==0.1.4
clean-fid==0.1.3
scikit-image==0.22.0
```

### Random Seeds
```python
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Experiment Tracking
- **MLflow** or **W&B** (if API key available)
- Log: config, metrics, curves, checkpoints, predictions
- Artifact: model weights, validation predictions, test predictions

---

## Compute Budget Estimate (Cloud GPU)

| Model | GPU-hours (est.) | Kaggle Weeks (2×T4) |
|-------|------------------|---------------------|
| EDSR-small | 4-6 | 1 |
| SwinIR-base | 12-16 | 2-3 |
| Real-ESRGAN | 24-32 | 4-5 |
| StableSR (if) | 48+ | 8+ |
| **Total (core 3)** | **40-54** | **7-9 weeks** |

**Fits within Phase 5 (6 weeks) if parallelized on Kaggle 2×T4**

---

## Next Steps
1. [ ] Implement model wrappers in `src/models/` (config-driven)
2. [ ] Create training scripts in `src/training/` with Hydra/OmegaConf configs
3. [ ] Set up Kaggle notebooks for each model
4. [ ] Define evaluation pipeline in `src/evaluation/`
5. [ ] Run tiny-model smoke test locally on CPU