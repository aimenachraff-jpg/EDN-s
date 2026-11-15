# MASTER2-A1 Agent Instructions

## Project Overview
**Thesis:** "Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery"
**Degree:** Master 2 Computer Science / Artificial Intelligence
**Institution:** Algeria

## Agent Roles & Responsibilities

### Primary Agent (This Session)
- **Research Scientist**: Literature review, hypothesis formation, methodology design
- **Data Engineer**: ALSAT-2B data acquisition, preprocessing, dataset preparation
- **ML Engineer**: PyTorch implementation, training pipelines, evaluation scripts
- **Evaluation Scientist**: Experimental design, metrics, statistical analysis
- **Scientific Reviewer**: Critical review of own work, identify flaws and limitations
- **Thesis Writer**: LaTeX/BibTeX document production

## Key Rules (Non-Negotiable)

1. **Never fabricate** - No fake papers, results, datasets, or sensor specs
2. **Never claim unrun experiments** - Only report actual executions
3. **Distinguish established findings vs. hypotheses vs. proposed methods**
4. **Record all limitations honestly**
5. **Preserve negative/inconclusive results**
6. **Maintain reproducibility** - Seeds, versions, configs, dataset provenance
7. **Request approval before major decisions** - Dataset changes, methodology shifts, training starts
8. **No fictional results in thesis** - Every claim backed by actual output

## Technical Stack
- **OS**: Windows 10/11
- **Python**: 3.10.11
- **PyTorch**: 2.3.0 (CPU only - no CUDA)
- **Git**: 2.55.0
- **OpenCode**: ✅ **RUNNING** (this session, desktop client)
- **FreeLLMAPI**: ✅ **RUNNING** on `http://localhost:31415/v1` (OpenAI-compatible)
- **GPU**: Intel Iris Xe Graphics (integrated, 2GB shared VRAM) - **NO DEDICATED GPU**
- **RAM**: ~248 GB free on C: drive
- **Storage**: ~248 GB free

## Hardware Constraints
- **No NVIDIA GPU / No CUDA** - All PyTorch training runs on CPU
- **Integrated graphics only** - Intel Iris Xe (2GB shared)
- **Training will be SLOW on CPU** - Must select tiny models, small batches, few epochs for local dev
- **Cloud GPU required for actual training** (Kaggle recommended: 2× T4 16 GB free tier, ~30 h/week)

## FreeLLMAPI Status (Verified)
- **Server**: Running on `http://localhost:31415`
- **Health**: `GET /health` → 200 OK
- **API**: OpenAI-compatible at `/v1`
- **Auth**: Bearer token required (API key in OpenCode config)
- **Routing**: `auto` model with fallback chain
- **Tool-calling**: To be verified once authenticated
- **Models**: To be listed via `/v1/models` with valid API key

## Project Structure
```
MASTER2-A1/
├── AGENTS.md
├── README.md
├── opencode.json
├── .gitignore
├── .env.example
├── PHASE_0_REPORT.md
├── research/
│   ├── research_question.md
│   ├── research_gap.md
│   ├── literature_review.md
│   ├── paper_database.csv
│   ├── methodology.md
│   ├── hypotheses.md
│   ├── research_log.md
│   ├── research_plan.md
│   ├── feasibility_report.md
│   ├── feasibility_checklist.md
│   ├── dataset_investigation.md
│   ├── dataset_comparison.md
│   ├── model_comparison.md
│   └── alsat2b_access.md
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── metadata/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── degradation/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── hallucination/
│   ├── downstream/
│   ├── visualization/
│   └── utils/
├── configs/
├── experiments/
│   ├── protocols/
│   ├── logs/
│   └── checkpoints/
├── tests/
├── notebooks/
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   ├── tables/
│   └── statistical_analysis/
├── thesis/
│   ├── main.tex
│   ├── chapters/
│   ├── figures/
│   ├── references.bib
│   ├── appendices/
│   └── final.pdf
└── docs/
    ├── installation.md
    ├── methodology_decisions.md
    ├── experiment_tracking.md
    ├── reproducibility.md
    ├── limitations.md
    └── phase_1_report.md
```

## Phase Workflow
- **Phase 0**: Infrastructure & Environment Audit ✅ **COMPLETE** (2026-11-12)
- **Phase 1**: Literature Review & Research Feasibility ✅ **COMPLETE** (2026-11-13)
- **Phase 2**: Research Questions & Methodology ✅ **COMPLETE** (2026-11-12/13)
## Phase Workflow
- **Phase 0**: Infrastructure & Environment Audit ✅ **COMPLETE** (2026-11-12)
- **Phase 1**: Literature Review & Research Feasibility ✅ **COMPLETE** (2026-11-13)
- **Phase 2**: Research Questions & Methodology ✅ **COMPLETE** (2026-11-12/13)
- **Phase 3**: Dataset Acquisition & Preparation ✅ **COMPLETE** (2026-11-15; full acquisition approved 2026-11-14)
- **Phase 4**: Baseline Implementation ✅ **COMPLETE** (2026-11-15; M4 draft + user approvals recorded; execution starts with the Kaggle upload)
- **Phase 5**: Super-Resolution Experiments ⏳ **Pending**
- **Phase 6**: Hallucination-Aware Evaluation ⏳ **Pending**
- **Phase 7**: Downstream Task Evaluation ⏳ **Pending**
- **Phase 8**: Statistical Analysis & Discussion ⏳ **Pending**
- **Phase 9**: Thesis Writing & Final Validation ⏳ **Pending**

## Approval Gates
**Must request human approval before:**
- Downloading large datasets (>1 GB)
- Starting training runs (>30 min)
- Changing research direction/methodology
- Selecting fundamentally different dataset
- Making major methodological decisions
- Publishing/submitting anything
- Claiming thesis complete

## Documentation Maintenance Rule (Non-Negotiable)

Whenever a project milestone is completed, the agent **must** perform the following steps in order:

1. **Update the relevant phase report** with verified results, evidence, and any corrections made during execution.
2. **Update `README.md`** to reflect the new verified state — including phase statuses, dataset availability, test counts, hardware info, and project structure. Never let README drift from reality.
3. **Update the project structure** in README if new files or directories were added, or existing ones removed/renamed.
4. **Update testing and dataset status** sections to match actual results (passing tests, data integrity checks, provenance records).
5. **Keep all claims consistent** across README, phase reports, configs, and implementation code. If code changes contradict documentation, fix the documentation.
6. **Never report a task as complete without evidence.** Every completion claim must reference actual command output, file existence, checksums, or test results.

### Why This Matters
- `README.md` is the primary entry point for anyone (including future AI sessions) understanding this project. An outdated README creates false assumptions that propagate into incorrect decisions.
- `PROJECT_UPDATE.md` serves as the permanent development history and handover guide. It must be updated after every meaningful change so that context is never lost between sessions.

---

## Communication Protocol
- Report findings with evidence (file paths, command outputs)
- Distinguish: **Fact** (verified) vs **Assumption** vs **Hypothesis** vs **TODO**
- Flag risks and blockers early
- Document all decisions in `docs/methodology_decisions.md`

## Current Status (Phase 3 — COMPLETE, 2026-11-15)

**Full acquisition completed and verified (supervisor approved 2026-11-14):**

- ✅ **PROBA-V:** 1,450 imgsets (1,160 train / 290 test), 58,202 PNGs, ~698 MB; MD5 `7101101c7c91eeb5979d472861c1b784` matches Zenodo
- ✅ **Sentinel-2:** 87/88 scenes (67.86 GB) across 8 AOIs; DZ_01 train = 5/6 (degenerate granule `31SDB_20201223` rejected + recorded in `rejected_items.json`); sanity scan 89/89 OK, 0 flags
- ✅ **Tiling:** 14,748 tiles (7,968 train / 2,712 val / 4,068 test), 1024×1024, overlap 128, float32, 100.15 GB; 24 tile-index CSVs verified
- ✅ **Void-fill bug fixed:** no nodata tag in raw GeoTIFFs → rotated-granule voids were silent exact-0 fill; 14/87 train scenes affected (>5% void, up to 44%); fixed (all-4-band-zero → NaN), affected scenes re-tiled, indexes + stats regenerated. See `docs/phase_3_full_acquisition.md` §5.
- ✅ **Normalization stats:** train-only, NaN-excluded, percentile 2–98 + mean/std (`data/metadata/sentinel2/normalization_stats.json`)
- ✅ **Unit Tests:** All 246 tests passing (incl. 12 PROBA-V loader tests against the real archive, 16 normalization tests, 82 preprocessing tests)

**Phase 4 — Baseline Implementation:** M1–M3 COMPLETE (2026-11-15):
- ✅ **M1 Pairs:** 149 Sim-ALSAT-2B ×4 pairs (119 train / 30 val) in `data/interim/pairs/` — HR = clean 10m tiles (nan==0.0 pools: 4,212 train / 1,060 val), LR = Wald degradation (`forward_ms_only`); verified shapes, finiteness, degradation sanity (diff 0.0288 vs naive resize). Commit `81f1ac0`
- ✅ **M2 Bicubic baseline:** first real numbers — `results/metrics/baseline_bicubic.json`: PSNR 37.59 dB (std 6.97), SSIM 0.899, SAM 0.0204, ERGAS 0.0311, wavelet HH ≈ 0 (TUS rationale). 30 val pairs, 40.3 s
- ✅ **M3 Smoke train:** EDSR-tiny (0.160M params) CPU train 12 s → checkpoint save/reload roundtrip → full-tile eval 23.32 dB. Found+fixed real checkpoint-load bug in `load_model_from_checkpoint` (explicit config was ignored; commit `b564900`, regression test added, 11 inference tests)
- ⏳ **M4:** ✅ DONE — draft `experiments/phase5/phase5_notebook_draft.py` + upload checklist; **user approvals received 2026-11-15** (upload ~72 GB train+val tiles; EDSR-small first, GPU 1 idle; 20m bands deferred). Commit `25dab66`
- Deferred: S2 ×2 native track (20m bands not acquired — approval-gated Phase 5 download); PROBA-V MISR pairs (needs MISR dataset class)

**Phase 5 — Super-Resolution Experiments:** next (Kaggle 2× T4; user approval required).

**OpenCode:** ✅ **RUNNING** (this session)
**FreeLLMAPI:** ✅ **RUNNING** on `http://localhost:31415/v1`
**Hardware:** Intel Iris Xe Graphics (integrated, 2GB shared VRAM) — **NO DEDICATED GPU**

**Methodology Note:** Evaluation metrics now comprehensive with compute_all_metrics convenience function handling both 2D and 3D inputs, with proper handling of PSNR/TUS edge cases for identical images.

---

## Technical Stack
- **OS:** Windows 10/11
- **Python:** 3.10.11
- **PyTorch:** 2.3.0 (CPU only — no CUDA)
- **Git:** 2.55.0
- **OpenCode:** ✅ **RUNNING** (this session, desktop client)
- **FreeLLMAPI:** ✅ **RUNNING** on `http://localhost:31415/v1` (OpenAI-compatible)
- **GPU:** Intel Iris Xe Graphics (integrated, 2GB shared VRAM) - **NO DEDICATED GPU**
- **RAM:** ~248 GB free on C: drive
- **Storage:** ~248 GB free

## Hardware Constraints
- **No NVIDIA GPU / No CUDA** — All PyTorch training runs on CPU
- **Integrated graphics only** — Intel Iris Xe (2GB shared)
- **Training will be SLOW on CPU** — Must select tiny models, small batches, few epochs for local dev
- **Cloud GPU required for actual training** (Kaggle recommended: 2× T4 16 GB free tier, ~30 h/week)

## FreeLLMAPI Status (Verified)
- **Server:** Running on `http://localhost:31415`
- **Health:** `GET /health` → 200 OK
- **API:** OpenAI-compatible at `/v1`
- **Auth:** Bearer token required (API key in OpenCode config)
- **Routing:** `auto` model with fallback chain
- **Tool-calling:** To be verified once authenticated
- **Models:** To be listed via `/v1/models` with valid API key