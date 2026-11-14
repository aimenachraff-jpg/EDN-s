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
- **Phase 3**: Dataset Acquisition & Preparation 🔄 **Pilot complete; full acquisition pending approval**
- **Phase 4**: Baseline Implementation ⏳ **Pending**
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

## Current Status (Phase 3 — Pilot Complete)

**Key Completed in Phase 3 Pilot:**

- ✅ **Dataset Acquisition:** Sentinel-2 DZ_01 (3 scenes, 12 GeoTIFFs, ~2 GB)
- ✅ **PROBA-V Verification:** Specification confirmed against Zenodo/ESA sources  
- ✅ **Unit Tests:** All 217 tests passing (data:12, degradation:26, evaluation:33, models:19, hallucination:15, downstream:22, training:14, preprocessing:66, inference:10)
- ✅ **Geospatial Verification:** Script created and tested (8.8-11.1m resolution at ~36.5°N)
- ✅ **License Documentation:** Completed (CC BY 4.0 for both datasets)
- ✅ **Evaluation Metrics Module:** Implemented (PSNR, SSIM, SAM, ERGAS, SCI, wavelet HH energy, GLCM entropy, TUS)

**Priority Active Work Items (Moving into Phase 4):**

- [ ] **Full Sentinel-2 acquisition** (8 AOIs, ~50–60 GB) — supervisor approval pending
- [ ] **PROBA-V download test** (675 MB) — supervisor approval pending
- [ ] **Tiling implementation and spatial buffer enforcement**
- [ ] **Normalization statistics computation** (pending real-data validation)
- [ ] **Integration/end-to-end tests** (pending full dataset)

**Next Milestone:** Supervisor approval → Phase 4: Baseline Implementation

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