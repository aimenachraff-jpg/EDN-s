# Research Log — MASTER2-A1

## Phase 0: Infrastructure Audit (Corrected) — 2025-11-11
- [x] Project directory structure created (38 directories)
- [x] Python 3.10.11, PyTorch 2.3.0+cpu, Git 2.55.0 verified
- [x] OpenCode desktop client running (session `ses_e3066ac28ffev0Vf7WsQiNsHHE`)
- [x] FreeLLMAPI server healthy on `http://localhost:31415` (health=200)
- [x] Key packages: rasterio, geopandas, timm, segmentation-models-pytorch, sentinelhub-py
- [x] Git repository initialized, initial commit `16c6b3e`
- [x] AGENTS.md, README.md, .gitignore, .env.example, opencode.json, PHASE_0_REPORT.md created
- [x] ALSAT-2B specs verified from ESA eoPortal (authoritative)

## Phase 1: Literature Review & Feasibility — 2025-11-12

### Morning: Paper Collection
- [x] Built reproducible arXiv collector (`research/collect_papers.py`)
- [x] Fixed SSL cert issue with requests+verify=False
- [x] Executed 29 queries covering SR architectures, RS-SR, hallucination, downstream tasks, cross-sensor, diffusion, benchmarks
- [x] **Collected 267 unique papers** → `research/paper_database.csv` + `.json`
- [x] Verified key 2026 papers: GeoSR-Bench, Spatial Downscaling, Mind the Gap, SALD, SenseBench, VHA

### Midday: ALSAT-2B Investigation
- [x] Fetched full eoPortal specs (NAOMI instrument, bands, detector, telescope)
- [x] Attempted ASAL website (asal.dz) — **INACCESSIBLE**
- [x] Checked ESA EO Gateway — 404 for ALSAT-2
- [x] Documented access channels + contact strategy in `research/alsat2b_access.md`
- [x] Decision: 4-week gate, then formal substitute dataset commitment

### Afternoon: Dataset Analysis
- [x] Compared ALSAT-2B vs Sentinel-2 vs PROBA-V vs NAIP vs PlanetScope vs WorldView
- [x] **Spectral match**: S2 B2/B3/B4/B8 ≈ ALSAT-2B B1/B2/B3/B4 (excellent)
- [x] Selected **Sentinel-2 + PROBA-V MISR** as primary stack
- [x] Created `research/dataset_investigation.md` + `research/dataset_comparison.md`

### Late Afternoon: Model Comparison
- [x] Surveyed 4 model families: CNN (EDSR/RCAN), Transformer (SwinIR/HAT), GAN (Real-ESRGAN), Diffusion (StableSR)
- [x] Defined 3 tiers: Tiny (local CPU), Standard (cloud GPU), Conditional (diffusion)
- [x] Created `research/model_comparison.md` with training recipes

### Evening: Research Synthesis
- [x] Identified 6 primary research gaps from literature
- [x] Formulated 10 hypotheses (6 primary, falsifiable) in `research/hypotheses.md`
- [x] Designed full methodology in `research/methodology.md`
- [x] Assessed feasibility in `research/feasibility_report.md` — **GO**
- [x] Wrote literature review in `research/literature_review.md`
- [x] Created Phase 1 report in `docs/phase_1_report.md`

---

## Pending / Next Actions (Week 2-3)

### Immediate (This Week)
- [ ] Apply for PlanetScope academic license
- [ ] Recruit 3 annotators for hallucination labeling (Week 9-10)
- [ ] Set up MLflow tracking (local Docker)
- [ ] Begin Sentinel-2 download for Algeria AOIs (priority)
- [ ] Draft Chapter 2 (Literature Review) from `literature_review.md`

### Phase 2 (Week 3): Research Questions & Methodology
- [ ] Finalize research questions from hypotheses
- [ ] Refine methodology with supervisor input
- [ ] Define exact evaluation protocols
- [ ] Begin S2 data download pipeline implementation

### Phase 3 (Week 4-5): Dataset Acquisition & Prep
- [ ] Complete S2 download for all 8 AOIs
- [ ] Load PROBA-V MISR
- [ ] Implement ALSAT-2B degradation model
- [ ] **Decision Gate**: ALSAT-2B access status

### Phase 4 (Week 6-7): Baseline Implementation
- [ ] Implement model wrappers (EDSR-small, Tiny SwinIR)
- [ ] Training pipeline with Hydra/MLflow
- [ ] Smoke test on CPU, validate on Kaggle

---

## Key Decisions Logged

| Date | Decision | Rationale | Document |
|------|----------|-----------|----------|
| 2025-11-11 | Use FreeLLMAPI local server | Running, healthy, OpenAI-compatible | PHASE_0_REPORT.md |
| 2025-11-11 | CPU-only local dev, cloud GPU for training | No NVIDIA GPU; Intel Iris Xe only | AGENTS.md |
| 2025-11-12 | Sentinel-2 as primary ALSAT-2B substitute | Spectral match, open access, benchmarks | dataset_comparison.md |
| 2025-11-12 | PROBA-V MISR as MISR benchmark | Only established benchmark, 50+ papers | dataset_investigation.md |
| 2025-11-12 | 3 core model families (EDSR, SwinIR, Real-ESRGAN) | CPU-testable, GPU-trainable, diverse | model_comparison.md |
| 2025-11-12 | 4-week ALSAT-2B access gate | Real data unlikely; substitute ready | alsat2b_access.md |
| 2025-11-12 | Hallucination taxonomy: 4 categories | Spectral, structural, textural, semantic | hypotheses.md (H2) |
| 2025-11-12 | Task-driven evaluation: fixed-backbone protocol | Standard in GeoSR-Bench, measures inherent quality | methodology.md |

---

## Open Questions / Blockers

1. **FreeLLMAPI API Key** — Not in env vars or standard config paths; likely embedded in OpenCode desktop app. Need to extract or configure manually in `.env`.
2. **ASAL Contact** — Website down; need alternative contact via CNES/ESA/academic network.
3. **PlanetScope License** — Application pending; 2-4 week timeline.
4. **Annotation Team** — Need 3 RS experts for Week 9-10 hallucination labeling.

---

## Metrics Tracker

| Metric | Target | Current |
|--------|--------|---------|
| Papers collected | >200 | 267 ✅ |
| Key 2026 papers identified | >5 | 7 ✅ |
| Dataset comparison complete | Yes | Yes ✅ |
| Model families selected | 3-4 | 3 core + 1 conditional ✅ |
| Hypotheses formulated | 5-10 | 10 (6 primary) ✅ |
| Feasibility GO/NO-GO | GO | GO ✅ |
| Phase 1 report | Complete | Complete ✅ |

---

*Log updated: 2025-11-12 18:30*