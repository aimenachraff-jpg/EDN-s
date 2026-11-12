# Experiment Tracking Documentation

**Project**: MASTER2-A1 — Hallucination-Aware SR Evaluation for ALSAT-2B  
**Status**: Phase 2 Complete — Protocol Ready for Execution  
**Tracking Backend**: MLflow (local + cloud), Git commits, config versioning

---

## 1. Experiment Tracking Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT REGISTRY                       │
├─────────────────────────────────────────────────────────────┤
│  MLflow Server (local: localhost:5000)                      │
│  ├── Experiments (by phase/task)                             │
│  │   ├── Runs (individual training/eval)                     │
│  │   │   ├── Parameters (config hash, seed, model, data)    │
│  │   │   ├── Metrics (train/val/test, hallucination, task)  │
│  │   │   ├── Artifacts (checkpoints, predictions, plots)    │
│  │   │   └── Tags (phase, model_family, dataset, status)    │
│  │   └── Run Comparisons (leaderboards)                      │
│  └── Model Registry (versioned models)                       │
│                                                               │
│  Git Integration:                                             │
│  ├── Commit SHA linked to every run                          │
│  ├── Config YAMLs versioned in configs/                      │
│  └── Phase gates = tagged commits                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Experiment Naming Convention

```
{phase}_{task}_{model}_{dataset}_{degradation}_{seed}
```

| Component | Values | Example |
|-----------|--------|---------|
| **phase** | `p3_data`, `p4_baseline`, `p5_sr`, `p6_halluc`, `p7_downstream` | `p5_sr` |
| **task** | `train`, `eval`, `finetune`, `ablation` | `train` |
| **model** | `edsr_small`, `swinir_base`, `swinir_tiny`, `realesrgan`, `bicubic` | `edsr_small` |
| **dataset** | `s2_algeria`, `probav_misr`, `naip`, `planet`, `simalsat` | `s2_algeria` |
| **degradation** | `wald`, `bicubic`, `bicubic_noise`, `ideal` | `wald` |
| **seed** | `s42`, `s123`, `s456` | `s42` |

**Example**: `p5_sr_train_edsr_small_s2_algeria_wald_s42`

---

## 3. MLflow Experiment Structure

### Phase 3: Data Preparation
- **Experiment**: `p3_data_preparation`
- **Runs**: Sentinel-2 download, PROBA-V prep, tile extraction, split verification
- **Key Artifacts**: `data/metadata/split_manifest.json`, normalization stats

### Phase 4: Baseline Implementation
- **Experiment**: `p4_baseline_training`
- **Runs**: One per model family × degradation config
- **Key Parameters**: `model`, `loss`, `lr`, `batch_size`, `epochs`, `degradation_config`
- **Key Metrics**: `train_loss`, `val_psnr`, `val_ssim`, `val_lpips`, `epoch_time`

### Phase 5: SR Experiments
- **Experiment**: `p5_sr_experiments`
- **Runs**: Full factorial (4 models × 3 degradations × 2 seeds = 24 runs)
- **Key Metrics**: PSNR, SSIM, LPIPS, SAM, ERGAS, SCI, QNR, HF_energy, hallucination_index

### Phase 6: Hallucination Evaluation
- **Experiment**: `p6_hallucination_evaluation`
- **Runs**: Detection on all SR outputs + human annotation protocol
- **Key Metrics**: Per-category detection rates, Fleiss' κ, composite hallucination index

### Phase 7: Downstream Evaluation
- **Experiment**: `p7_downstream_evaluation`
- **Runs**: Fixed-backbone evaluation on 4 tasks × all SR methods
- **Key Metrics**: mIoU, OA, Kappa (classification); mAP@0.5, mAP@0.5:0.95 (detection); mIoU (segmentation); F1, IoU (change detection)

---

## 4. Parameter Logging Schema

### Model Parameters (logged at run start)
```yaml
model:
  name: edsr_small
  config: configs/models/edsr_small.yaml
  num_params: 1520000
  scale: 4
  in_channels: 4
  out_channels: 4

training:
  loss: l1+lpips
  lpips_weight: 0.1
  learning_rate: 0.0001
  batch_size: 16
  epochs: 200
  optimizer: adam
  scheduler: cosine
  seed: 42

data:
  train_split: s2_algeria_train
  val_split: s2_algeria_val
  degradation: wald
  degradation_config: configs/degradation/alsat2b.yaml
  tile_size: 64
  stride: 32

compute:
  device: cuda:0
  mixed_precision: true
  num_workers: 4
```

### Metric Logging (logged per epoch / evaluation)
| Metric | Log Key | Frequency | Aggregation |
|--------|---------|-----------|-------------|
| Training loss | `train/loss` | per epoch | mean |
| Validation PSNR | `val/psnr` | per epoch | mean |
| Validation SSIM | `val/ssim` | per epoch | mean |
| Validation LPIPS | `val/lpips` | per epoch | mean |
| Test PSNR | `test/psnr` | final | mean ± bootstrap CI |
| Test SSIM | `test/ssim` | final | mean ± bootstrap CI |
| Test LPIPS | `test/lpips` | final | mean ± bootstrap CI |
| Test SAM | `test/sam` | final | mean ± bootstrap CI |
| Test ERGAS | `test/ergas` | final | mean ± bootstrap CI |
| Test SCI | `test/sci` | final | mean ± bootstrap CI |
| Test QNR | `test/qnr` | final | mean ± bootstrap CI |
| Hallucination index | `test/hallucination_index` | final | mean ± bootstrap CI |
| Structural hallucination | `test/struct_hallucination` | final | mean ± bootstrap CI |
| Task utility (classification) | `task/cls_miou` | final | mean ± bootstrap CI |
| Task utility (detection) | `task/det_map50` | final | mean ± bootstrap CI |
| Task utility (segmentation) | `task/seg_miou` | final | mean ± bootstrap CI |
| Task utility (CD) | `task/cd_f1` | final | mean ± bootstrap CI |

---

## 5. Artifact Management

### Checkpoints
- **Path**: `experiments/checkpoints/{experiment_name}/{run_name}/`
- **Naming**: `epoch_{epoch:04d}_val_psnr_{val_psnr:.2f}.pt`
- **Best**: `best.pt` (highest val PSNR)
- **Last**: `last.pt` (final epoch)

### Predictions
- **Path**: `experiments/logs/{experiment_name}/{run_name}/predictions/`
- **Format**: `.npz` files (HR, SR, LR, metadata)
- **Structure**: `tile_{idx:06d}.npz`

### Visualizations
- **Path**: `experiments/logs/{experiment_name}/{run_name}/figures/`
- **Types**: Sample grids, metric curves, hallucination heatmaps

---

## 6. Git Integration

### Commit Tagging
| Phase | Tag Format | Example |
|-------|------------|---------|
| Phase 0 | `phase-0-complete` | `phase-0-complete` (commit 004e0e4) |
| Phase 1 | `phase-1-complete` | `phase-1-complete` |
| Phase 2 | `phase-2-complete` | `phase-2-protocol-finalized` |
| Phase 3 | `phase-3-data-ready` | |
| Phase 4 | `phase-4-baselines-trained` | |
| Phase 5 | `phase-5-sr-complete` | |
| Phase 6 | `phase-6-halluc-complete` | |
| Phase 7 | `phase-7-downstream-complete` | |
| Phase 8 | `phase-8-analysis-complete` | |
| Phase 9 | `phase-9-thesis-final` | |

### Config Versioning
All YAML configs in `configs/` are tracked in Git. Each experiment run logs the Git SHA.

```bash
# Get config at experiment time
git show <sha>:configs/models/edsr_small.yaml
```

---

## 7. Reproducibility Checklist per Run

- [ ] Git SHA recorded
- [ ] All config YAMLs logged as artifacts
- [ ] Random seeds set (PyTorch, NumPy, Python)
- [ ] CuDNN deterministic mode enabled
- [ ] Python package versions logged (`pip freeze`)
- [ ] CUDA/cuDNN versions logged
- [ ] Hardware info logged (GPU type, CPU, RAM)
- [ ] Data split manifest hash logged
- [ ] Degradation config hash logged

---

## 8. Dashboard Queries (MLflow)

### Compare Models on Validation PSNR
```sql
SELECT run_name, params.model, metrics.val_psnr
FROM runs
WHERE experiment_id = 'p5_sr_experiments'
  AND tags.phase = 'p5_sr'
ORDER BY metrics.val_psnr DESC
```

### Hallucination Index by Model
```sql
SELECT params.model, metrics.test_hallucination_index
FROM runs
WHERE experiment_id = 'p6_hallucination_evaluation'
ORDER BY metrics.test_hallucination_index ASC
```

### Task Utility Ratio
```sql
SELECT params.model,
       metrics.task_cls_miou / metrics.test_hallucination_index as cls_ratio,
       metrics.task_det_map50 / metrics.test_hallucination_index as det_ratio
FROM runs
WHERE experiment_id = 'p7_downstream_evaluation'
```

---

## 9. Phase 2 → Phase 3 Handoff

### What's Ready
- [x] MLflow tracking infrastructure in `src/training/trainer.py`
- [x] Config files versioned in `configs/`
- [x] Naming convention documented
- [x] Artifact paths defined

### What Needs Setup (Phase 3)
- [ ] MLflow server on cloud (Kaggle/DagsHub/self-hosted)
- [ ] S3/artifact store for large checkpoints
- [ ] Automated run comparison dashboard
- [ ] Alerting for failed runs

---

## 10. References

- `src/training/trainer.py` — `SRTrainer` class with MLflow integration
- `configs/training/trainer_config.yaml` — Training config template
- `experiments/protocols/kaggle_train_template.py` — Cloud training script
- MLflow docs: https://mlflow.org/docs/latest/index.html