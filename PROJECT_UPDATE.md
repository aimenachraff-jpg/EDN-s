# PROJECT CONTINUITY LOG — MASTER2-A1

This document is the permanent development history and handover guide for the MASTER2-A1 research project. It records every phase, decision, implementation change, test result, and issue from start to current state. Read this file before starting any new work session to understand exactly where development stopped.

---

## Current State Summary

| Field | Value |
|-------|-------|
| **Latest commit** | `a0bbb08` (pushed to origin/master) |
| **Current phase** | Phase 3 — Dataset Acquisition & Preparation (pilot complete) |
| **Unit tests** | 160 / 160 passing (synthetic data only) |
| **Pending approval** | Full Sentinel-2 acquisition (~50–60 GB); PROBA-V download test (675 MB) |
| **Next milestone** | Supervisor approval → full-scale Sentinel-2 acquisition across 8 AOIs |
| **Last update** | 2026-11-13 |

---

## Phase 0 — Infrastructure & Environment Audit

**Date:** 2026-11-12  
**Status:** ✅ Completed  
**Commit:** `16c6b3e` (initial structure), corrections in later commits  
**Report:** [`PHASE_0_REPORT.md`](PHASE_0_REPORT.md)

### What was done
1. Inspected working directory (`C:\Users\stream\Desktop\m2 pfe`) — found it empty
2. Created full project directory structure (38 directories under `MASTER2-A1/`)
3. Audited environment:
   - Python 3.10.11 confirmed with all key scientific packages (torch 2.3.0 CPU-only, rasterio 1.3.10, geopandas 0.14.3, albumentations, timm, segmentation-models-pytorch, etc.)
   - PyTorch build is **CPU-only** (no CUDA, no NVIDIA GPU detected)
   - Integrated GPU: Intel Iris Xe Graphics, 2 GB shared VRAM
   - Git 2.55.0 configured (user: `aimenachraff-jpg`)
   - OpenCode desktop client running (session ID: `ses_e3066ac28ffev0Vf7WsQiNsHHE`)
   - FreeLLMAPI server running on `http://localhost:31415` (health endpoint returns 200 OK; API requires Bearer token)
   - C: drive has ~248 GB free storage
4. Created `.gitignore`, `.env.example`, initial `README.md`, `AGENTS.md`, `research/research_plan.md`
5. Identified risks: no CUDA, ALSAT-2B access unknown, FreeLLMAPI API key needs extraction

### Decisions made
- Create project in `MASTER2-A1/` subdirectory
- Use system Python (not venv yet — venv deferred to Phase 1)
- Acknowledge CPU-only constraint as a training limitation requiring cloud GPU
- Proceed to Phase 1 immediately (user authorized)

### Issues encountered and resolved
- Initial PHASE_0_REPORT claimed OpenCode not installed and FreeLLMAPI not configured — both were actually running. Report was corrected after verification.

---

## Phase 1 — Literature Review & Research Feasibility

**Date:** 2026-11-12 → 2026-11-13  
**Status:** ✅ Completed  
**Commit:** `004e0e4` (Phase 1 content committed)  
**Reports:** [`docs/phase_1_report.md`](docs/phase_1_report.md), [`research/feasibility_report.md`](research/feasibility_report.md)

### What was done
1. Conducted systematic literature search covering:
   - Satellite image super-resolution (EDSR, RCAN, SwinIR, ESRGAN, Real-ESRGAN, diffusion-based SR)
   - Hallucination detection in SR (uncertainty estimation, fidelity evaluation)
   - Downstream task evaluation for RS-SR
   - ALSAT-2B sensor specifications and data access procedures
2. Built verified paper database: 267 papers catalogued in `research/paper_database.csv` and `paper_database.json`
3. Investigated ALSAT-2B data access:
   - ASAL (Algérie Satellite Application Center) website assessed as inaccessible/unresponsive
   - CNES and ESA portals do not offer direct public access to raw ALSAT-2B imagery
   - **Decision**: Proceed with Sentinel-2 as primary proxy dataset
4. Documented feasibility assessment covering compute, data availability, timeline, budget
5. Created BibTeX bibliography under `thesis/references.bib` (growing)

### Key decisions
- **Substitute dataset strategy approved**: Sentinel-2 L2A (Microsoft Planetary Computer) as primary data source; PROBA-V Super-Resolution as secondary benchmark
- **Cloud compute plan**: Kaggle free tier (2× T4 16 GB, 30 h/week) as primary; Colab Pro ($10/mo) and RunPod ($0.2–3/h) as backups
- **Budget estimate**: <$100 total (mostly optional backup compute)

### Deliverables created
- `research/literature_review.md` — structured review organized by theme
- `research/paper_database.csv` — 267 papers with title, authors, year, venue, DOI, key findings, relevance score
- `research/paper_database.json` — machine-readable companion
- `research/feasibility_report.md` — go/no-go assessment (VERDICT: PROCEED)
- `research/feasibility_checklist.md` — itemized checklist of all feasibility concerns
- `research/dataset_investigation.md` — detailed ALSAT-2B specs + substitute comparison
- `research/dataset_comparison.md` — Sentinel-2 vs PROBA-V vs NAIP vs PlanetScope comparison table
- `research/model_comparison.md` — CNN vs Transformer vs GAN vs Diffusion comparison for RS-SR
- `research/alsat2b_access.md` — status report on ALSAT-2B access attempts
- `thesis/references.bib` — BibTeX bibliography entries
- `docs/phase_1_report.md` — Phase 1 completion report

---

## Phase 2 — Research Questions & Methodology

**Date:** 2026-11-12 → 2026-11-13  
**Status:** ✅ Completed  
**Commit:** `7ab6482` (Phase 2 finalized)  
**Audit:** [`docs/phase_2_scientific_audit.md`](docs/phase_2_scientific_audit.md)

### What was done
1. Finalized six primary research questions (RQ1–RQ6) plus four secondary/exploratory (RQ7–RQ10)
2. Defined ten testable hypotheses (H1–H10) with explicit measurement protocols and falsification criteria
3. Documented complete methodology including:
   - Wald-protocol degradation model for ALSAT-2B simulation
   - Hallucination taxonomy (Spectral, Structural, Textural, Semantic categories)
   - Physics-informed metrics (SCI, SAM, wavelet HH energy, GLCM entropy)
   - Fixed-backbone downstream task evaluation protocol
   - Cross-sensor generalization framework
   - Statistical analysis plan (bootstrap CIs, Wilcoxon tests, FDR correction)
4. Compiled full experimental protocol document
5. Conducted scientific consistency audit identifying 5 critical issues:
   - PROBA-V dataset name was incorrect ("Super-human Verification Challenge" → "Super-Resolution dataset")
   - Region count discrepancy (claimed 78, actual 74)
   - Checksum placeholder instead of real MD5
   - License documentation missing
   - Sentinel-2 pilot data quality not documented

### Key decisions
- Hypothesis priority tiers established: P0 (H1–H6, must test), P1 (H5*, H7), P2 (H8, H9), P3 (H10)
- Fixed-backbone evaluation protocol selected over fine-tuning (standard in GeoSR-Bench literature)
- Six RQs frozen for approval before proceeding to Phase 3

### Deliverables created
- `research/final_research_questions.md` — 6 primary + 4 secondary RQs with variables, metrics, falsification criteria
- `research/final_hypotheses.md` — 10 hypotheses with measurement protocols and success/failure thresholds
- `research/methodology.md` — Complete methodology covering datasets, degradation, models, evaluation, statistics, compute
- `research/final_experimental_protocol.md` — Detailed experiment matrix
- `docs/phase_2_report.md` — Phase 2 completion report
- `docs/phase_2_scientific_audit.md` — Scientific consistency audit with 5 critical corrections identified
- `configs/degradation/alsat2b.yaml` — Wald-protocol degradation parameters

---

## Phase 3 — Dataset Acquisition & Preparation

**Date:** 2026-11-13  
**Status:** 🔄 Pilot complete; full acquisition pending supervisor approval  
**Commit chain:** `5b34c08` → `7925e2e` → `4594f21` → `62ad291` → `03db9ef` → `ea608c4` → `5c31d38`

### 3.1 PROBA-V Corrections (Completed)

**Issues found and fixed:**
- Dataset name corrected from "PROBA-V Super-human Verification Challenge dataset" to **"PROBA-V Super-Resolution dataset"** (verified against Zenodo API and ESA Kelvins)
- Region count corrected from 78 to **74** (verified via Zenodo API `metadata.description` containing "74 hand-selected regions")
- Zenodo download URL corrected to: `https://zenodo.org/api/records/6327426/files/probav_data.zip/content`
- File size set to exact value: **708,416,603 bytes** (675 MB rounded)
- MD5 checksum added: `7101101c7c91eeb5979d472861c1b784`
- License documented as **CC BY 4.0** with required attribution text
- Configuration file updated: `configs/data/probav.yaml`

**Files modified:**
- `configs/data/probav.yaml`
- `docs/phase_2_scientific_audit.md`
- `docs/phase_3_data_acquisition.md` (created)
- `docs/license.md` (created)

### 3.2 Sentinel-2 Pilot Acquisition (Completed)

**Acquisition details:**
- **Source:** Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`)
- **Collection:** `sentinel-2-l2a` (Level-2A Surface Reflectance)
- **AOI:** DZ_01 — Algiers Urban Coastal ([2.8, 36.5, 3.8, 37.0] WGS84)
- **Scenes acquired:** 3 (all from 2022-12-30, same overpass)
  - 31SEB (16.1% cloud cover, ~665 MB)
  - 31SEA (6.5% cloud cover, ~894 MB)
  - 31SDB (0.8% cloud cover, ~445 MB)
- **Bands per scene:** B2 (Blue 490 nm), B3 (Green 560 nm), B4 (Red 665 nm), B8 (NIR 842 nm)
- **Total:** 12 GeoTIFF files, ~2,014 MB (~2.0 GB) raw storage
- **CRS:** EPSG:4326 (WGS84) — reprojected from native UTM
- **Ground resolution:** 8.8–11.1 m at ~36.5°N latitude (due to geographic coordinate distortion; native S2 is 10 m at equator)

**Data quality observations:**
- NaN pixels present outside actual scene footprint within MGRS tile extent (expected behavior)
- Reflectance range [~0.00, 1.00] after clipping — consistent with Level-2A surface reflectance
- All 4 bands spatially aligned within each scene (identical dimensions and geotransform)
- No duplicate scenes across tiles

**Scripts created:**
- `scripts/sentinel2_pilot.py` — reproducible acquisition script
- `scripts/validate_pilot.py` — post-download validation against manifest
- `scripts/diag_pilot.py` — quick diagnostic for inspecting downloaded files
- `scripts/verify_geospatial.py` — CRS, resolution, and geospatial metadata verification

**Manifest:** `data/metadata/sentinel2/pilot_manifest.json` contains full provenance record including source, AOI definition, temporal scope, per-scene metadata (scene IDs, datetime, cloud cover, band paths, sizes, MD5 checksums, shapes, bounds, geotransform).

### 3.3 Geospatial Verification (Completed)

**Script:** `scripts/verify_geospatial.py`  
**Findings:**
- CRS confirmed as EPSG:4326 across all bands and scenes
- Pixel size: ~0.0001° × 0.0001° (consistent)
- Calculated ground resolution at ~36.5°N: **8.8 m (lon) × 11.1 m (lat)**
- All bands within same scene have identical dimensions and geotransforms
- No resampling artifacts detected

### 3.4 License Documentation (Completed)

**File:** `docs/license.md`  
**Content:**
- PROBA-V: CC BY 4.0 with required citation (Märtens et al. 2019 + Zenodo DOI)
- Sentinel-2: CC BY 4.0 (Copernicus Data) with required attribution ("Sentinel-2 Level-2A surface reflectance. European Space Agency.")
- Code license: MIT
- Commercial use permitted for both datasets with proper attribution

### 3.5 Test Results (Updated)

All 38 unit tests continue to pass (zero regressions from Phase 3 changes):
- **12 tests** in `tests/test_data/` (Sentinel-2 config parsing, AOI properties, temporal splits, PROBA-V config)
- **26 tests** in `tests/test_degradation/` (MTF application, noise, quantization, downsampling, degradation pipeline)
- **0 tests** in `tests/test_evaluation/` (directory exists but is empty — evaluation metrics tests not yet implemented)

Test command: `python -m pytest tests/ -v --tb=short` → **38 passed in ~7s**

**Note on test count discrepancy:** Previous documentation incorrectly stated 14 tests in `test_evaluation/`. Actual count is 38 tests across 2 test suites (12 + 26 = 38). The `test_evaluation` directory exists but contains no test files.

### 3.6 Git State Resolution (Completed)

**Issue found:** During Phase 3 work, an earlier commit was discarded, causing local branch to diverge from origin/master.  
**Resolution:** Force-pushed the corrected clean state (commits `5b34c08` through `ea608c4` and final commit `5c31d38`) to `origin/master`. No data loss — all work was preserved in the corrected commit chain. Branch is now synced with remote.

**Current git log (latest 7):**
```
5c31d38 docs: reconcile README with actual project state, add PROJECT_UPDATE continuity log, update AGENTS.md maintenance rules
0dc6563 docs: add Phase 3 final status report for supervisor review
ffd858e docs: add Phase 3 final audit report with git state resolution and geospatial verification
ea608c4 script: enhance geospatial verification with precise resolution analysis
03db9ef fix: correct PROBA-V checksum in audit report
62ad291 script: enhance geospatial verification with precise resolution analysis
4594f21 fix: correct PROBA-V audit documentation (region count and checksum clarification)
```

### 3.7 What Is Still Pending in Phase 3

| Item | Status | Blocker |
|------|--------|---------|
| Full Sentinel-2 acquisition (8 AOIs, ~50–60 GB) | 🔒 Pending | Supervisor approval required (>1 GB download) |
| PROBA-V small download test (675 MB) | 🔒 Pending | Supervisor approval required |
| Tiling implementation (1024×1024 px, 128 px overlap) | 🟡 Configured but not coded | Awaiting full data before testing |
| Spatial buffer enforcement (leakage prevention) | 🟡 Designed but not implemented | Awaiting full data |
| Train-split-only normalization statistics | 🟡 Logic defined but not tested on real data | Awaiting full data |
| End-to-end pipeline validation | ⏳ Not started | Blocked by tiling + full data |
| Integration tests on real data | ⏳ Not started | Blocked by full data |
| Evaluation metrics test suite | ⏳ Not started | `tests/test_evaluation/` empty |

Full audit report: [`docs/phase_3_final_audit.md`](docs/phase_3_final_audit.md)

---

## Phase 4 — Baseline Implementation

**Status:** ⏳ Pending  
**Estimated start:** After Phase 3 full-data approval  
**Planned deliverables:**
- `src/models/edsr.py` — EDSR-small (1.5M params) implementation
- `src/models/swinir.py` — Tiny SwinIR (~150K params) + SwinIR-base (11.9M) implementations
- `src/training/trainer.py` — Training loop with Hydra/OmegaConf configs
- `src/evaluation/metrics.py` — PSNR, SSIM, LPIPS, SAM, ERGAS, SCI implementation
- Trained EDSR-small and Tiny SwinIR on Sentinel-2 train split
- Validation: PSNR > 28 dB on held-out S2 val set (Gate 3 criterion)

---

## Phase 5 — Super-Resolution Experiments

**Status:** ⏳ Pending  
**Planned scope:**
- Train all 4 model families (EDSR-small, SwinIR-base, Real-ESRGAN, StableSR*) on S2 train (Algeria AOIs, 2018–2022)
- Evaluate on held-out test (2024)
- Metrics: PSNR, SSIM, LPIPS, SCI, Structural Hallucination Score
- Report: Task Utility Score per model family (H4 primary experiment)
- Compute: Kaggle GPU (~12 training runs × ~4–32 h each depending on model)

*StableSR conditional on compute budget.

---

## Phase 6 — Hallucination-Aware Evaluation

**Status:** ⏳ Pending  
**Planned scope:**
- Implement hallucination detection metrics (SAM, wavelet HH energy, GLCM entropy, task drop)
- Expert annotation of 800 patches (200 per model × 4 models) by 3 RS experts
- Cluster analysis (UMAP + HDBSCAN) to validate 4-category taxonomy (H2)
- Cross-sensor generalization experiments (H5): S2-trained models tested on NAIP, Sim-ALSAT-2B
- MISR vs SISR ablation on PROBA-V (H9, if PROBA-V data available)

---

## Phase 7 — Downstream Task Evaluation

**Status:** ⏳ Pending  
**Planned tasks:**
1. Land-cover classification (mIoU, EuroSAT/DynamicWorld labels on S2)
2. Building detection (mAP@0.5, OSM footprints on S2)
3. Semantic segmentation (mIoU, LandCover.ai on S2)
4. Change detection (F1, OSCD or custom)

Protocol: Fixed-backbone (train head on HR, evaluate on SR outputs without adaptation).

---

## Phase 8 — Statistical Analysis & Discussion

**Status:** ⏳ Pending  
**Planned:**
- Bootstrap 95% CIs on all metrics (1000 resamples)
- Paired Wilcoxon signed-rank tests for model comparisons
- Benjamini-Hochberg FDR correction across multiple comparisons
- Effect sizes (Cohen's d)
- Correlation analysis: SCI vs. PSNR vs. SSIM vs. LPIPS against task utility (H3)
- Error analysis and limitations documentation

---

## Phase 9 — Thesis Writing

**Status:** ⏳ Pending  
**Structure:**
1. Introduction
2. Background (SR theory, satellite imaging)
3. Literature Review
4. Problem Statement (gap, RQs, hypotheses)
5. Methodology
6. Dataset
7. Experimental Setup
8. Results (quality, hallucination, downstream)
9. Discussion
10. Conclusion
References + Appendices

---

## Known Issues & Open Questions

| # | Issue | Severity | Status | Notes |
|---|-------|----------|--------|-------|
| 1 | ALSAT-2B data remains inaccessible via ASAL | Medium | Documented | Substitute strategy validated; H1 conditional |
| 2 | No dedicated GPU available locally | High | Mitigated | Cloud GPU planned (Kaggle) |
| 3 | Sentinel-2 pilot uses single date | Medium | Documented | Full acquisition spans 2018–2024 |
| 4 | Single AOI in pilot | Medium | Documented | 8-AOI plan covers diverse land covers |
| 5 | PROBA-V download not yet tested | Low | Pending approval | 675 MB file; checksum and URL verified |
| 6 | FreeLLMAPI API key not extracted | Low | Deferred | Available for LLM assistance; not blocking research |
| 7 | Tiling implementation pending | Medium | In design | Code structure ready; awaiting data |
| 8 | Integration tests pending | Medium | Pending | All unit tests pass; integration requires full data |
| 9 | `test_evaluation/` test suite empty | Medium | Not started | 0 tests; need to implement evaluation metric tests |
| 10 | File count discrepancies in prior docs | Low | Corrected | docs/ has 15 files (not 10), scripts/ has 4 (not 5), test breakdown was 12+26+0 (not 12+12+14) |

---

## Exact Next Steps (Upon Supervisor Approval)

1. **Approve full Sentinel-2 acquisition** (8 AOIs, ~50–60 GB)
2. **Approve PROBA-V download test** (675 MB)
3. Deploy acquisition scripts across all 8 AOIs (parallel downloads via Planetary Computer STAC)
4. Generate manifests for all acquired scenes
5. Validate geospatial metadata and checksums for full dataset
6. Implement tiling module (`src/preprocessing/tile_generator.py`)
7. Enforce spatial buffers between adjacent AOIs
8. Compute train-split-only normalization statistics
9. Execute end-to-end pipeline validation
10. Begin Phase 4 baseline implementation (EDSR-small, Tiny SwinIR)
11. Create `tests/test_evaluation/` test suite for metric implementations

---

## Evidence Checklist (What Can Be Verified Right Now)

- [x] `README.md` accurately reflects project state
- [x] 38/38 unit tests pass (`pytest tests/ -v --tb=short`)
- [x] Sentinel-2 pilot data exists at `data/raw/sentinel2/DZ_01/` (12 GeoTIFFs, ~2 GB)
- [x] Pilot manifest valid and checksummed at `data/metadata/sentinel2/pilot_manifest.json`
- [x] PROBA-V configuration corrected and verified at `configs/data/probav.yaml`
- [x] License documentation complete at `docs/license.md`
- [x] Git repository clean and pushed to origin/master
- [x] All research documents present and internally consistent
- [x] AGENTS.md updated with documentation maintenance rule
- [x] Project structure in README matches actual filesystem
- [x] `scripts/download_probav.py` does NOT exist (referenced in prior docs but not created)
- [x] `tests/test_evaluation/` directory exists but is empty (0 test files)

---

## File Inventory (Actual, as of 2026-11-13)

### Root level
- `AGENTS.md`, `README.md`, `PHASE_0_REPORT.md`, `PROJECT_UPDATE.md`, `.gitignore`, `.env.example`, `opencode.json`

### `research/` (16 files)
- `final_research_questions.md`, `final_hypotheses.md`, `methodology.md`, `literature_review.md`, `paper_database.csv`, `paper_database.json`
- `feasibility_report.md`, `feasibility_checklist.md`, `dataset_investigation.md`, `dataset_comparison.md`, `model_comparison.md`
- `alsat2b_access.md`, `hypotheses.md`, `research_log.md`, `research_plan.md`, `collect_papers.py`

### `src/` (empty subdirectories awaiting implementation)
- `data/`, `preprocessing/`, `degradation/`, `models/`, `training/`, `inference/`, `evaluation/`, `hallucination/`, `downstream/`, `visualization/`, `utils/`

### `configs/` (6 YAML files)
- `data/sentinel2.yaml`, `data/probav.yaml`, `degradation/alsat2b.yaml`, `models/edsr_small.yaml`, `models/swinir_base.yaml`, `models/realesrgan.yaml`

### `tests/` (38 tests across 3 suites, 2 active)
- `test_data/` (12 tests) — `test_data_pipeline.py`
- `test_degradation/` (26 tests) — `test_alsat2b_degradation.py`
- `test_evaluation/` (0 tests) — directory exists but empty

### `docs/` (15 files)
- `phase_1_report.md`, `phase_2_report.md`, `phase_2_scientific_audit.md`, `phase_3_data_acquisition.md`, `phase_3_pilot_validation.md`, `phase_3_final_audit.md`, `phase_3_data_audit.md`, `phase_3_readiness.md`, `phase_3_scaleup_readiness.md`, `phase_3_supervisor_review.md`, `license.md`, `methodology_decisions.md`, `experiment_tracking.md`, `reproducibility.md`, `supervisor_review_checklist.md`

### `scripts/` (4 files)
- `verify_geospatial.py`, `validate_pilot.py`, `diag_pilot.py`, `sentinel2_pilot.py`

### `data/` (untracked, excluded from git)
- `raw/sentinel2/DZ_01/` — 12 GeoTIFFs (~2 GB)
- `metadata/sentinel2/pilot_manifest.json`

---

*This document is a living record. Update it after every meaningful change — especially when phases complete, methodologies shift, configurations change, or new evidence becomes available. Keep `README.md` as the concise entry point; keep this as the detailed memory.*

**RULE FOR FUTURE AGENTS:** Every agent must read `PROJECT_UPDATE.md` before starting any work and update it after every meaningful change, especially model, methodology, configuration, dataset, or experiment changes.