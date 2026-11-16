# MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery

> **Master 2 Thesis** — Computer Science / Artificial Intelligence, Algeria  
> **Working title:** *Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery*  
> **Status:** Phase 5 starting - Kaggle upload COMPLETE and VERIFIED (10,697 files / 78.16 GB in one version: 10,680 train+val tiles, 16 CSVs, 1 stats JSON; all 16 AOIxsplit groups match local indexes); code dataset master2-a1-code uploaded · M1–M3 complete (pairs, bicubic baseline, smoke train) · 246/246 tests · Phase 5 next (Kaggle)
> **Last updated:** 2026-11-16

---

## Research Objective

Investigate whether deep-learning super-resolution (SR) methods improve **actual information usefulness** for ALSAT-2B satellite imagery, or merely produce visually convincing but **hallucinated details** that degrade downstream task performance (land-cover classification, building detection, semantic segmentation, change detection).

---

## Research Questions

| ID | Question | Status |
|----|----------|--------|
| **RQ1** | Can a Wald-protocol degradation model using estimated NAOMI parameters produce synthetic LR data statistically indistinguishable from real ALSAT-2B? | Protocol designed · Awaiting data |
| **RQ2** | Do hallucinations in RS-SR cluster into four distinct measurable categories (Spectral, Structural, Textural, Semantic)? | Taxonomy defined · Awaiting annotation |
| **RQ3** | Does the Spectral Consistency Index (SCI) correlate with downstream task utility better than PSNR, SSIM, or LPIPS? | Metric designed · Awaiting experiments |
| **RQ4** | Which SR model family (CNN, Transformer, GAN, Diffusion) maximizes task utility per unit of hallucination? | Experimental protocol defined · Awaiting training |
| **RQ5** | When tested on target sensors (NAIP, PlanetScope, Sim. ALSAT-2B), does the domain gap follow: Diffusion > GAN > Transformer > CNN? | Protocol defined · Conditional on cross-sensor data |
| **RQ6** | Can tiny models (<10% FLOPs) achieve >90% task utility of large counterparts on CPU inference? | Designed · Pending validation |

**Secondary RQs** (H7–H10): Task-dependent hallucination profiles (H7), physics-informed loss ablation (H8), MISR vs SISR comparison on PROBA-V (H9), no-reference model-selection proxy (H10).

Full research questions: [`research/final_research_questions.md`](research/final_research_questions.md)

---

## Hypotheses (Testable & Falsifiable)

Ten hypotheses are pre-registered with explicit falsification criteria. Priority tiers:

| Tier | Hypotheses | Description |
|------|-----------|-------------|
| **P0 — Must** | H1–H6 | Core thesis claims; failure invalidates central contribution |
| **P1 — Should** | H5*, H7 | Significant contributions if confirmed |
| **P2 — Could** | H8, H9 | Mechanistic ablations; enriches discussion |
| **P3 — Exploratory** | H10 | Nice-to-have; not required for defense |

**Key P0 summary:**

- **H1** Degradation model fidelity (FID < 10, PSD KS < 0.15, correlation Frobenius < 0.1) — conditional on any ALSAT-2B sample availability
- **H2** Hallucination taxonomy validity (silhouette > 0.5, Fleiss' κ > 0.7 for 4 clusters)
- **H3** SCI correlates with task utility (ρ > 0.7) better than PSNR (ρ < 0.4)
- **H4** SwinIR (Transformer) achieves highest Task-Utility / Hallucination ratio among 4 families
- **H5** Cross-sensor gap ordering: Diffusion > GAN > Transformer > CNN
- **H6** Tiny models reach >90% task utility at <10% FLOPs, CPU inference < 1s

Full hypotheses: [`research/final_hypotheses.md`](research/final_hypotheses.md)

---

## Current Project Status

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Infrastructure & Environment Audit | ✅ Completed (2026-11-12) |
| 1 | Literature Review & Research Feasibility | ✅ Completed (2026-11-13) |
| 2 | Research Questions & Methodology | ✅ Completed (2026-11-12) |
| 3 | Dataset Acquisition & Preparation | ✅ Complete (2026-11-15): both datasets acquired, tiled, normalized |
| 4 | Baseline Implementation | 🔄 M1–M3 complete: 149 sim-×4 pairs, bicubic baseline (PSNR 37.59 dB), CPU smoke train + roundtrip; M4 approvals received; Kaggle single-version upload in progress (20/33 blobs saved, resumable) |
| 4 | Baseline Implementation | ⏳ Pending |
| 5 | Super-Resolution Experiments | ⏳ Pending |
| 6 | Hallucination-Aware Evaluation | ⏳ Pending |
| 7 | Downstream Task Evaluation | ⏳ Pending |
| 8 | Statistical Analysis & Discussion | ⏳ Pending |
| 9 | Thesis Writing & Final Validation | ⏳ Pending |

---

## Datasets

### Primary Dataset: Sentinel-2 (Proxy for ALSAT-2B MS)

Sentinel-2 Level-2A (Surface Reflectance) is used as a spectral proxy for the unavailable ALSAT-2B multi-spectral band. Both share similar blue-green-red-NIR band positions, making it a scientifically justified substitute for methodological development — though results apply to Sentinel-2, not ALSAT-2B directly.

| Property | Value |
|----------|-------|
| **Source** | Microsoft Planetary Computer STAC API (`sentinel-2-l2a`) |
| **License** | Copernicus / Sentinel-2 Data, CC BY 4.0 |
| **Attribution** | "Sentinel-2 Level-2A surface reflectance. European Space Agency." |
| **Processing level** | L2A (surface reflectance, values in [0, 1]) |
| **Bands acquired** | B2 (Blue 490 nm), B3 (Green 560 nm), B4 (Red 665 nm), B8 (NIR 842 nm) |
| **Native resolution** | 10 m |
| **Actual CRS in download** | EPSG:4326 (reprojected from native UTM); ground resolution ≈ 8.8–11.1 m at ~36.5°N latitude due to geographic-coordinate distortion |

**Eight Area-of-Interest (AOI) plan** (stratified by land cover):

| AOI ID | Location | Land Cover | Size |
|--------|----------|------------|------|
| DZ_01 | Algiers, Algeria | Urban / Coastal | 100 × 100 km |
| DZ_02 | Tell Atlas, Algeria | Agricultural / Forest | 100 × 100 km |
| DZ_03 | Sahara (Ghardaïa), Algeria | Desert / Bare soil | 100 × 100 km |
| DZ_04 | Oran Coast, Algeria | Coastal / Wetland | 100 × 100 km |
| FR_01 | Paris, France | Dense Urban | 50 × 50 km |
| US_01 | Iowa, USA | Intensive Agriculture | 50 × 50 km |
| CL_01 | Atacama, Chile | Hyper-arid Desert | 50 × 50 km |
| BR_01 | Amazon Mouth, Brazil | Tropical Coastal | 50 × 50 km |

**Temporal splits** (no leakage): Train = 2018–2022, Val = 2023, Test = 2024 (held-out year, no cloud filter).

#### Pilot Acquisition (Completed)

- **Acquisition date:** 2026-11-13
- **AOI:** DZ_01 (Algiers)
- **Scenes:** 3 (MGRS tiles 31SEB, 31SEA, 31SDB) — same overpass date 2022-12-30
- **Files:** 12 GeoTIFFs (B2/B3/B4/B8 per scene)
- **Storage:** ~2 014 MB (~2.0 GB) raw
- **Manifest:** [`data/metadata/sentinel2/pilot_manifest.json`](data/metadata/sentinel2/pilot_manifest.json) — contains source provenance, per-band MD5 checksums, shapes, bounds, geotransform
- **Validation:** All 12 files pass checksum verification; reflectance range [0, 1]; CRS consistent

#### Full Acquisition (COMPLETE, 2026-11-15 — supervisor approved 2026-11-14)

- **Scenes:** 87/88 (67.86 GB) across all 8 AOIs — DZ_01 train = 5/6 documented (its 2020-12-23 date had only one candidate granule, rejected as degenerate: 99% of pixels raw DN 0/1; recorded in `data/metadata/sentinel2/full/rejected_items.json`)
- **Manifests:** `data/metadata/sentinel2/full/{AOI}_{split}_manifest.json` (24 files, counts verified)
- **Quality audit:** independent scan of all 89 scene dirs — **89/89 OK, 0 flags** (`sanity_scan.json`)
- **Tiling:** **14,748 tiles** (7,968 train / 2,712 val / 4,068 test), 1024×1024, overlap 128, float32, 100.15 GB; 24 tile-index CSVs row-verified
- **Void-fill bug fixed:** raw GeoTIFFs carry no nodata tag; rotated-granule voids are silent exact-0 fill (14/87 train scenes >5% void, up to 44%). Fix: all-4-band-zero → NaN; affected scenes re-tiled; verified (`void_scan.json`, phase report §5)
- **Normalization stats:** train-only, NaN-excluded — `data/metadata/sentinel2/normalization_stats.json` (percentile 2–98 + mean/std)

#### Limitations (Explicit)

1. **Proxy data only** — Sentinel-2 ≠ ALSAT-2B; spectral characteristics differ
2. **DZ_01 train shortfall** — 5/6 scenes (no alternative granule for the rejected date)
3. **No SCL/cloud masking** — SCL bands not downloaded; tiles record NaN fraction for loader-side filtering
4. **PROBA-V region leakage** — no official region→imgset mapping; split by scene ID (seed 42)

See [`docs/phase_3_full_acquisition.md`](docs/phase_3_full_acquisition.md) for the full report with evidence.

---

### Secondary Dataset: PROBA-V Super-Resolution Dataset

**Official name:** PROBA-V Super-Resolution dataset (not "Super-human Verification Challenge").

| Property | Value |
|----------|-------|
| **Source** | ESA Kelvins Platform / Zenodo record 6327426 |
| **URL** | `https://zenodo.org/api/records/6327426/files/probav_data.zip/content` |
| **File size** | 708 416 603 bytes (~675 MB) |
| **MD5 checksum** | `7101101c7c91eeb5979d472861c1b784` |
| **License** | CC BY 4.0 |
| **Regions** | 74 hand-selected globally |
| **Scenes** | 1 450 total (1 160 train / 290 test, official challenge split) |
| **Bands** | RED (620–690 nm), NIR (770–890 nm) — 2 bands at 300 m → 100 m (×3 scale) |
| **Format** | 14-bit PNG; includes quality masks (QM) and status maps (SM) |
| **Registration** | Not pre-registered; sub-pixel shifts present (scoring is shift-invariant via cPSNR) |

**Correct attribution (CC BY 4.0 requirement):**

> Märtens M., Izzo D., Krzic A. and Cox D. "Super-resolution of PROBA-V images using convolutional neural networks." Astrodynamics 3.4 (2019): 387–402. [arXiv:1907.01821](https://arxiv.org/abs/1907.01821)

Zenodo DOI: `https://doi.org/10.5281/zenodo.6327426`

#### Verification Status

- **Name, region count (74), URL, file size, MD5 checksum** — verified against Zenodo API and ESA Kelvins documentation ✅
- **Configuration file** (`configs/data/probav.yaml`) — corrected and accurate ✅
- **Downloaded & checksum-verified** (2026-11-14): 708 416 603 bytes, MD5 match, extracted 58 202 PNGs + `norm.csv` ✅
- **Layout verified:** imgset IDs partition by band — train/RED 0000–0593 (594), train/NIR 0594–1159 (566), test/RED 1160–1305 (146), test/NIR 1306–1449 (144); totals match spec (1 160 train + 290 test) ✅
- **Loader:** `src/data/probav_loader.py` single-band semantics, 12 tests against the real archive ✅

See [`configs/data/probav.yaml`](configs/data/probav.yaml) and [`docs/license.md`](docs/license.md).

---

### Cross-Sensor Datasets (Conditional)

| Dataset | Role | Access | Notes |
|---------|------|--------|-------|
| **NAIP** | HR target for S2 ×10 cross-sensor test | AWS S3 (US AOIs only) | Open; RGBN only |
| **PlanetScope** | HR target for S2 ×3 cross-sensor test | Academic license pending | Apply early |
| **Simulated ALSAT-2B** | Target domain for H5 | Generated via degradation model | No external dependency |

---

## Code License & Data Attribution

| Component | License | Terms |
|-----------|---------|-------|
| **Code** | MIT | Free use, modification, distribution with attribution |
| **Sentinel-2 data** | CC BY 4.0 | Attribution required — see dataset section above |
| **PROBA-V data** | CC BY 4.0 | Attribution required — citation and Zenodo DOI mandatory |

Full licensing terms and attribution requirements: [`docs/license.md`](docs/license.md)

---

## Technical Stack

| Component | Version / Status |
|-----------|-----------------|
| **OS** | Windows 10/11 |
| **Python** | 3.10.11 |
| **PyTorch** | 2.3.0 (**CPU build** — NO CUDA) |
| **torchvision** | 0.18.0 |
| **rasterio** | 1.3.10 (GeoTIFF / geospatial I/O) |
| **geopandas** | 0.14.3 |
| **Git** | 2.55.0 |
| **OpenCode** | ✅ Running (this session, desktop client) |
| **FreeLLMAPI** | ✅ Running on `http://localhost:31415` (OpenAI-compatible) |

### Hardware Constraints

| Resource | Available | Suitable For |
|----------|-----------|--------------|
| **CPU (Python/PyTorch)** | ✅ | Development, debugging, tiny-model smoke tests, evaluation |
| **Integrated GPU** | Intel Iris Xe, 2 GB shared | Display only; CUDA NOT available |
| **Dedicated GPU / CUDA** | ❌ None | Training requires cloud |
| **RAM** | ≥16 GB (system RAM) | Data loading, small-batch training |
| **Disk (C:)** | ~248 GB free | Datasets, checkpoints, results |

**Implication:** Local training runs must use tiny models (< 2 M params), batch size 1–4, patch size 64–128, few epochs. **Cloud GPU (Kaggle free tier recommended) is required for actual experiment training.**

---

## Project Structure

```
MASTER2-A1/
│
├── AGENTS.md                         # Agent instructions & rules
├── README.md                         # This file
├── PHASE_0_REPORT.md                 # Infrastructure audit report
├── .gitignore                        # Excludes data/, checkpoints/, logs/, etc.
├── .env.example                      # Environment template
├── opencode.json                     # OpenCode configuration
│
├── research/                         # All research documents (16 files)
│   ├── final_research_questions.md   # 6 primary + 4 secondary RQs (frozen)
│   ├── final_hypotheses.md           # 10 hypotheses with falsification criteria
│   ├── methodology.md                # Complete experimental methodology
│   ├── literature_review.md          # Structured review with citations
│   ├── paper_database.csv            # 267+ verified papers
│   ├── paper_database.json           # Machine-readable paper database
│   ├── feasibility_report.md         # Compute/data/timeline assessment
│   ├── feasibility_checklist.md      # Itemized feasibility concerns
│   ├── dataset_investigation.md      # ALSAT-2B specs, substitutes, comparisons
│   ├── dataset_comparison.md         # Sentinel-2 vs PROBA-V vs NAIP vs PlanetScope
│   ├── model_comparison.md           # CNN vs Transformer vs GAN vs Diffusion
│   ├── alsat2b_access.md             # Access status for ALSAT-2B
│   ├── hypotheses.md                 # Initial hypotheses (superseded by final_hypotheses.md)
│   ├── research_log.md               # Ongoing research log
│   ├── research_plan.md              # Phase-by-phase plan
│   └── collect_papers.py             # Paper collection script
│
├── src/                              # Source code
│   ├── data/                         # Data loaders, datasets
│   ├── preprocessing/                # Tiling, normalization, splits
│   ├── degradation/                  # Wald-protocol ALSAT-2B simulator
│   ├── models/                       # EDSR, SwinIR, Real-ESRGAN, StableSR
│   ├── training/                     # Training loops, schedulers, losses
│   ├── evaluation/                   # Quality metrics (PSNR, SSIM, LPIPS, etc.)
│   ├── hallucination/                # H2/H3 metrics (SAM, wavelet, GLCM, SCI)
│   ├── downstream/                   # Task models (classification, detection)
│   └── utils/                        # Helpers, logging, config
│
├── configs/                          # Experiment & dataset configurations
│   ├── data/
│   │   ├── sentinel2.yaml            # 8 AOIs, temporal splits, bands
│   │   └── probav.yaml               # PROBA-V settings (corrected)
│   ├── degradation/
│   │   └── alsat2b.yaml              # Wald-protocol parameters
│   └── models/
│       ├── edsr_small.yaml           # EDSR-small config
│       ├── swinir_base.yaml          # SwinIR-base config
│       └── realesrgan.yaml           # Real-ESRGAN config
│
├── scripts/                          # Utility & acquisition scripts
│   ├── verify_geospatial.py          # Resolution & CRS verification
│   ├── validate_pilot.py             # Post-download pilot validation
│   ├── diag_pilot.py                 # Quick diagnostic for downloaded files
│   └── sentinel2_pilot.py            # Sentinel-2 pilot acquisition
│
├── tests/                            # Unit tests (all synthetic)
│   ├── test_data/                    # Data pipeline tests
│   ├── test_degradation/             # Degradation model tests
│   └── test_evaluation/              # Metric tests
│
├── docs/                             # Phase reports & documentation (15 files)
│   ├── phase_1_report.md             # Literature review completion
│   ├── phase_2_report.md             # Research questions finalized
│   ├── phase_2_scientific_audit.md   # Scientific consistency check
│   ├── phase_3_data_acquisition.md   # Pilot acquisition report
│   ├── phase_3_pilot_validation.md   # Pilot validation results
│   ├── phase_3_final_audit.md        # Pre-approval readiness audit
│   ├── phase_3_data_audit.md         # Data audit details
│   ├── phase_3_readiness.md          # Scale-up readiness assessment
│   ├── phase_3_scaleup_readiness.md  # Scale-up readiness details
│   ├── phase_3_supervisor_review.md  # Supervisor review package
│   ├── license.md                    # Dataset & code licensing
│   ├── methodology_decisions.md      # Decision log with rationale
│   ├── experiment_tracking.md        # MLflow / tracking setup
│   ├── reproducibility.md            # Reproducibility documentation
│   └── supervisor_review_checklist.md # Supervisor review checklist
│
├── data/                             # Satellite imagery (NOT in git — see .gitignore)
│   ├── raw/
│   │   └── sentinel2/
│   │       └── DZ_01/                # Pilot: 3 scenes, 12 GeoTIFFs (~2 GB)
│   └── metadata/
│       └── sentinel2/
│           └── pilot_manifest.json   # Provenance & checksums for pilot
│
├── experiments/                      # Experiment logs, protocols, checkpoints
│   ├── protocols/
│   ├── logs/
│   └── checkpoints/
│
├── results/                          # Generated outputs (not tracked in git)
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── statistical_analysis/
│
├── thesis/                           # LaTeX thesis
│   ├── main.tex
│   ├── chapters/
│   ├── references.bib
│   └── appendices/
│
└── notebooks/                        # Jupyter exploratory notebooks
```

### Git Status

```
Branch: master
Remote: origin/master
Current HEAD: Phase 3 completion batch (see git log)

Untracked (correctly excluded from git via .gitignore):
  data/raw/        — Raw satellite imagery (68 GB: PROBA-V archive + 87 Sentinel-2 scenes)
  data/processed/  — 14,748 tiles (100.15 GB)
  data/metadata/   — manifests, tile indexes, sanity/void scans, normalization stats
```

---

## Testing & Validation

### Unit Tests

| Suite | Count | Status | Notes |
|-------|-------|--------|-------|
| Data pipeline | 12 | ✅ Pass | Sentinel-2 config parsing, AOI properties, temporal splits, split verification |
| PROBA-V loader | 12 | ✅ Pass | Real-archive tests: band-ID partitioning, scene ranges, masks {0,1}, test-scene SM semantics, split reproducibility |
| Degradation model | 26 | ✅ Pass | MTF application, noise, quantization, downsampling, degradation pipeline |
| Evaluation metrics | 33 | ✅ Pass | PSNR, SSIM, SAM, ERGAS, SCI, wavelet HH energy, GLCM entropy, TUS |
| SR models | 19 | ✅ Pass | EDSR, SwinIR, Real-ESRGAN forward pass, shape, gradient flow |
| Hallucination detection | 15 | ✅ Pass | Spectral, structural, textural, semantic detection, annotation |
| Downstream tasks | 22 | ✅ Pass | UNet, DeepLab, SiameseUNet, BuildingDetector, metrics |
| Training pipeline | 14 | ✅ Pass | TrainingConfig, CharbonnierLoss, SRLoss, SRDataset, SRTrainer, CPU smoke |
| Preprocessing | 82 | ✅ Pass | Normalize (mean/std + percentile, streaming, NaN-safe), cloud mask, tiling (tile/untile, tile grid, blend kernel), spatial buffers, band alignment |
| Inference | 11 | ✅ Pass | Model creation, tiled inference, batch inference, bicubic upscale, pair evaluation, checkpoint reload (explicit-config regression test) |
| **Total** | **246** | **✅ 246/246 passing** | **Zero regressions** |

Tests run on synthetic data plus the **real PROBA-V archive** (12 loader tests read actual PNGs and verify the band-ID partitioning). Integration and end-to-end tests on Sentinel-2 tiles are pending (Phase 4).

### Pilot Validation Results

| Check | Result |
|-------|--------|
| File existence (12 GeoTIFFs) | ✅ |
| MD5 checksums match manifest | ✅ |
| CRS consistency (EPSG:4326) | ✅ |
| Band completeness (B2/B3/B4/B8) | ✅ |
| Dimension alignment within scene | ✅ |
| Reflectance range [0, 1] | ✅ |
| Geospatial resolution verified | ✅ (8.8–11.1 m at ~36.5°N) |

---

## Phase 3: What Is Complete vs. Pending

### ✅ Completed

- Scientific consistency audit across all prior phases
- PROBA-V naming, URL, region count (74), and checksum corrections
- PROBA-V configuration aligned with official ESA/Zenodo specification
- Sentinel-2 pilot acquisition (DZ_01, 3 scenes, 12 GeoTIFFs, ~2 GB)
- Pilot manifest generation with full provenance and per-band checksums
- Geospatial verification script confirming actual ground resolution
- CC BY 4.0 licensing documentation for both datasets
- All 217 unit tests passing (66 preprocessing incl. tiling & spatial buffers; zero regressions)
- Git state clean — branch synced to `origin/master`

### ⏳ Still Pending

- [ ] **Full Sentinel-2 acquisition** — expand from DZ_01 to all 8 AOIs across train/val/test periods (estimated ~50–60 GB)
- [ ] **PROBA-V small download test** — validate 675 MB archive before full-scale acquisition
- [x] **Tiling implementation** — `tile_image`/`untile_image` (overlap + feathered blending), `calculate_tiles`, `create_weight_kernel` — implemented & unit-tested on synthetic data (real-data validation pending)
- [x] **Spatial buffer utilities** — padding, edge-padding to tile multiples, validity checks, buffer requirements, hierarchical buffers — implemented & unit-tested (real-data validation pending)
- [ ] **Train-split-only normalization statistics** — per-band percentile computation validated on real data
- [ ] **End-to-end validation** — multi-AOI temporal split, tiling, normalization pipeline tested together
- [ ] **Supervisor approval** for full-scale acquisition and PROBA-V download test

Full status: [`docs/phase_3_final_audit.md`](docs/phase_3_final_audit.md)

---

## How to Use This Repository

### Before You Start

1. Read [`AGENTS.md`](AGENTS.md) for project rules and roles
2. Read [`research/final_research_questions.md`](research/final_research_questions.md) for the research scope
3. Read [`research/methodology.md`](research/methodology.md) for the experimental design
4. Read [`docs/license.md`](docs/license.md) for data-attribution requirements

### Key Commands

```bash
# Run unit tests
python -m pytest tests/ -v --tb=short

# Verify geospatial properties of downloaded data
python scripts/verify_geospatial.py --path data/raw/sentinel2/DZ_01

# Validate pilot data against manifest
python scripts/validate_pilot.py

# Download Sentinel-2 pilot (for reference)
python scripts/sentinel2_pilot.py --aoi DZ_01
```

---

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **No dedicated GPU** — CPU-only local training | Slow iteration | Cloud GPU (Kaggle) for training; local CPU for dev & evaluation |
| **ALSAT-2B data blocked** — ASAL access unavailable | Cannot validate on true target sensor | Sentinel-2 proxy validated; H1 conditional on any ALSAT-2B sample |
| **Single-date pilot only** | Temporal splits not yet validated on real data | Full acquisition plan spans 2018–2024; temporal leakage checks coded |
| **Single-AOI pilot** | Multi-region generalization unknown | 8 AOI plan covers diverse land-cover types |
| **Preprocessing pipeline untested on real data** | Tiling/normalization may reveal issues | Integration tests planned for Phase 4 |
| **Diffusion model training conditional** | H4/H5 may exclude StableSR if compute insufficient | Core claims rest on 3 families (EDSR, SwinIR, Real-ESRGAN) |

---

## References

- **Sentinel-2 data:** Microsoft Planetary Computer — `https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a`
- **PROBA-V dataset:** Zenodo record 6327426 — `https://doi.org/10.5281/zenodo.6327426`
- **Degradation model:** Wald, L. (2000). *The intensities relationships...* Photogrammetrie Fernerkundung Geoinformation
- **EDSR:** Lim et al. (2017). *Enhanced Deep Residual Networks for Single Image Super-Resolution*
- **SwinIR:** Liang et al. (2021). *SwinIR: Image Restoration Using Swin Transformer*
- **Real-ESRGAN:** Wang et al. (2021). *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*
- **PROBA-V baseline:** Märtens et al. (2019). *Super-resolution of PROBA-V images using CNNs*

---

*This repository is maintained as part of a Master 2 thesis. All scientific claims require evidence from executed experiments. No fabricated results.*
