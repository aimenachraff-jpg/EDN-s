# MASTER2-A1 Preliminary Feasibility Checklist

**Date:** 2026-11-12  
**Phase:** 0 — Infrastructure Audit  
**Status:** Initial Assessment

---

## 1. Data Availability Feasibility

| Item | Status | Evidence Needed | Risk |
|------|--------|-----------------|------|
| ALSAT-2B imagery accessible | ❓ Unknown | ASAL/CNES/ESA portal access, academic request process | 🔴 Critical |
| Sensor specifications verified | ❓ Unknown | Official datasheet (ASAL/CNES) | 🔴 Critical |
| Licensing permits research use | ❓ Unknown | License terms, redistribution rights | 🟠 High |
| Sufficient scenes for train/val/test | ❓ Unknown | Catalog search, coverage over Algeria | 🟠 High |
| Reference HR data available | ❓ Unknown | Panchromatic 2.5m as HR for MS 10m | 🟠 High |
| **Substitute dataset identified** | ❓ Unknown | Sentinel-2, PROBA-V, WorldView, DIV2K-Sat | 🟡 Medium |

**Action:** Phase 1 must investigate ASAL (Algerian Space Agency), CNES Kalideos, ESA Third Party Missions, commercial providers (Airbus, Maxar).

---

## 2. Computational Feasibility

| Resource | Local (CPU) | Cloud (Target) | Feasible? |
|----------|-------------|----------------|-----------|
| PyTorch training | ✅ Yes (slow) | ✅ Yes (GPU) | ✅ Yes |
| Model: EDSR (1.5M params) | ✅ ~1 hr/epoch | ✅ ~2 min/epoch | ✅ Yes |
| Model: RCAN (15M params) | ⚠️ ~5 hr/epoch | ✅ ~5 min/epoch | ☑️ Cloud only |
| Model: SwinIR (12M params) | ⚠️ ~4 hr/epoch | ✅ ~4 min/epoch | ☑️ Cloud only |
| Batch size 16, patch 256 | ❌ OOM / too slow | ✅ T4/P100/A100 | ☑️ Cloud only |
| Experiment tracking (wandb) | ✅ Yes | ✅ Yes | ✅ Yes |
| Hyperparameter search | ❌ Too slow | ✅ Yes | ☑️ Cloud only |

**Verdict:** **Local development only. All serious training on cloud (Colab/Kaggle/RunPod).**

---

## 3. Methodological Feasibility

| Component | Feasible? | Approach | Risk |
|-----------|-----------|----------|------|
| Controlled degradation simulation | ✅ Yes | Bicubic downsample + blur + noise + compression | Low |
| PSNR/SSIM/LPIPS computation | ✅ Yes | torchmetrics, lpips, skimage | Low |
| SAM/ERGAS (spectral metrics) | ✅ Yes | Custom implementation, rasterio | Low |
| Hallucination detection metrics | ⚠️ Research needed | Reference-based + no-reference + task-based | 🟠 High |
| Downstream task labels | ⚠️ Depends on data | WorldCover, OSM, custom annotation | 🟠 High |
| Statistical analysis | ✅ Yes | scipy, statsmodels, bootstrap | Low |
| Reproducible configs (seeds, versions) | ✅ Yes | Hydra, requirements.txt, git | Low |

---

## 4. Technical Skill Feasibility

| Skill | Current Level | Gap | Mitigation |
|-------|---------------|-----|------------|
| PyTorch model implementation | Intermediate | Transformer architectures | Use timm, reference implementations |
| Satellite image processing | Beginner | Radiometric/geometric correction | rasterio docs, tutorials |
| Geospatial data handling | Beginner | CRS, tiling, co-registration | geopandas, rasterio courses |
| Experiment design | Intermediate | Satellite-specific protocols | Literature review Phase 1 |
| Statistical analysis | Intermediate | Spatial stats, bootstrap | scipy.stats, statsmodels |
| LaTeX thesis writing | Beginner | Overleaf, templates | University template, practice |
| Academic writing (English) | Intermediate | Technical precision | Supervisor feedback, tools |

---

## 5. Timeline Feasibility (26 weeks + 2 buffer)

| Phase | Weeks | Feasible? | Critical Path |
|-------|-------|-----------|---------------|
| 0: Infrastructure | 1 | ✅ Done | — |
| 1: Literature & Feasibility | 3 | ✅ Yes | Data access investigation |
| 2: Methodology | 2 | ✅ Yes | Supervisor alignment |
| 3: Dataset Prep | 3 | ⚠️ Depends on data | **Data access = critical path** |
| 4: Baselines | 3 | ✅ Yes | Model selection |
| 5: SR Experiments | 4 | ⚠️ Cloud access needed | **GPU access = critical path** |
| 6: Hallucination Eval | 3 | ✅ Yes | Metric validation |
| 7: Downstream Eval | 3 | ⚠️ Labels needed | **Label access = critical path** |
| 8: Analysis | 2 | ✅ Yes | — |
| 9: Thesis Writing | 3 | ✅ Yes | Incremental drafting |
| **Total** | **27** | **✅ Feasible** | **Data + GPU access** |

---

## 6. Supervisor / Institutional Feasibility

| Requirement | Status | Action |
|-------------|--------|--------|
| Supervisor assigned | ❓ Unknown | Confirm with department |
| Topic approved | ❓ Unknown | Submit proposal |
| Ethics clearance (if needed) | ❓ N/A likely | Verify |
| Data sharing agreements | ❓ Unknown | ASAL/CNES contact |
| Compute resources (university) | ❓ Unknown | Check HPC access |
| Thesis format/template | ❓ Unknown | Get LaTeX template |
| Defense timeline | ❓ Unknown | Confirm dates |

---

## 7. Budget Feasibility

| Item | Cost | Source | Status |
|------|------|--------|--------|
| Cloud GPU (Colab Pro) | ~$10/mo | Personal | Optional |
| Cloud GPU (RunPod) | ~$0.30/hr (A100) | Personal | As needed |
| Cloud GPU (Kaggle) | Free (30h/week) | Free | **Primary** |
| API keys (LLM) | Free tier / paid | FreeLLMAPI / OpenAI | TBD |
| Data purchase | €0 - €1000s | ASAL/commercial | TBD |
| Software licenses | Free (open source) | — | ✅ Covered |
| Publication fees | €0 - €2000 | University / grant | TBD |

**Verdict:** **Feasible with free tiers (Kaggle, Colab free, open data). Budget only for emergencies.**

---

## 8. Overall Feasibility Score

| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Data Access | 2 | **Major unknown** — Phase 1 critical |
| Compute | 4 | Free cloud GPUs available |
| Methodology | 4 | Well-defined, metrics exist |
| Skills | 3 | Gaps addressable in Phase 1-2 |
| Timeline | 4 | Realistic with buffer |
| Supervision | 3 | Needs confirmation |
| Budget | 5 | Near-zero cost possible |
| **Overall** | **3.5/5** | **CONDITIONALLY FEASIBLE** |

---

## 9. Go/No-Go Criteria for Phase 1→2

**Must achieve by end of Phase 1:**
- [ ] ALSAT-2B access confirmed **OR** substitute dataset acquired and justified
- [ ] At least 50 verified papers in bibliography
- [ ] Research questions finalized and approved
- [ ] Baseline methods selected with implementations identified
- [ ] Cloud GPU access working (tested training run)
- [ ] Supervisor alignment on methodology

**If ALSAT-2B unavailable by end of Phase 1:**
- Document substitute sensor characteristics
- Quantify domain gap (resolution, bands, noise, geography)
- Adjust research questions to reflect substitute
- Get supervisor approval for scope change

---

## 10. Immediate Action Items

| # | Action | Owner | Deadline |
|---|--------|-------|----------|
| 1 | Get FreeLLMAPI GitHub URL & API key | User | Before Phase 1 |
| 2 | Install OpenCode & test | Assistant | Before Phase 1 |
| 3 | Initialize git repo | Assistant | Before Phase 1 |
| 4 | Create Python venv & requirements.txt | Assistant | Phase 1 start |
| 5 | Contact ASAL for ALSAT-2B access | User/Assistant | Phase 1 week 1 |
| 6 | Test Kaggle/Colab GPU access | Assistant | Phase 1 week 1 |
| 7 | Search Sentinel-2 data over Algeria | Assistant | Phase 1 week 1 |
| 8 | Get supervisor feedback on RQs | User | Phase 1 week 2 |

---

**Checklist Version:** 1.0  
**Next Update:** End of Phase 1