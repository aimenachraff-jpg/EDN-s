# EDN-S Repository Audit

**Date:** 2026-11-20  
**Repository:** https://github.com/aimenachraff-jpg/EDN-s  
**Audited revision:** `9957d24` (local `main` == `origin/main` == `origin/master`)  
**Working tree:** clean before audit; no uncommitted changes.

This audit distinguishes verified facts (checked against code, tests, or result files) from
documentation claims that could not be confirmed.

---

## 1. Repository state

| Item | Verified value |
|---|---|
| Default branch | `main` (remote also has `master` at the same commit) |
| Tracked files | 264 |
| Tracked binaries (`.tif`, `.png`, `.pt`, `.pth`) | none |
| Unit tests | **246 passed, 1 warning, 36.30 s** (`python -m pytest tests/ -q`) |
| Python package version | `pyproject.toml` = 1.0.2; PyPI = 1.0.2 |
| Package version in code | `edns.__version__` = **1.0.0** (inconsistent) |
| GitHub description | old text: “A lightweight PyTorch framework … hallucination-aware evaluation …” |
| GitHub topics | **none set** |
| GitHub homepage | **none set** |
| GitHub size | ~12.7 MB |

No secrets were found in tracked configuration placeholders (`.env.example` and
`opencode.json` use environment-variable references only).

---

## 2. Feature classification

### Fully implemented and exercised

| Area | Evidence |
|---|---|
| EDSR-small model (16 residual blocks, 64 features, ×4, 4-channel) | `src/edns/models/edsr.py`; 1,518,764 parameters measured |
| Tiled inference with overlap and reflect padding | `src/edns/inference/app.py` |
| GeoTIFF read/write and band handling | `src/edns/data/io.py` |
| Z-score normalization with training statistics | `src/edns/data/normalization.py` |
| CLI `edns-sr` with `--hallucination`, `--hr`, `--report` | verified `--help` output |
| Python API `run_sr()` | `src/edns/inference/__init__.py` |
| Metrics: PSNR, MSE, MAE, SSIM, SAM, ERGAS, SCI, wavelet HH energy, GLCM entropy, TUS | `src/edns/evaluation/metrics.py` |
| Hallucination detectors: spectral, structural, textural | `src/edns/hallucination/detection.py`; Phase 6 run on 1,742 tiles |
| Annotation protocol, Fleiss’ κ, metric validation | `src/edns/hallucination/detection.py` |
| ALSAT-2B Wald degradation simulation | `src/degradation/alsat2b_degradation.py` |
| Preprocessing: tiling, buffers, normalization | `src/preprocessing/` |
| Training pipeline (EDSR, SwinIR, Real-ESRGAN) | `src/training/trainer.py`; EDSR-small run completed (300 epochs, 6.82 h) |
| Downstream task framework | `src/downstream/tasks.py` |

### Implemented but not evaluated

| Area | Status |
|---|---|
| Semantic hallucination detection | Code exists; requires a downstream task model. Phase 6 explicitly skipped it. |
| Downstream task evaluation | Framework exists; task models were never trained. Phase 7 deferred. |
| SwinIR-base and Real-ESRGAN training | Code and configs exist; no completed training results in `results/`. |
| PROBA-V MISR pairing | Loader exists; MISR pair generation not performed. |
| Native Sentinel-2 ×2 track | 20 m bands were not acquired; only the ×4 simulated track was run. |

### Planned / not implemented

| Area | Status |
|---|---|
| Real ALSAT-2B imagery experiments | No ALSAT-2B data was obtained; all experiments use Sentinel-2 with simulated ALSAT-2B degradation. |
| Human annotation study | Protocol implemented; no annotators recruited. |
| Task-driven reliability conclusions | Blocked on the downstream task evaluation above. |

---

## 3. Main issues found

### 3.1 Scientific positioning and accuracy

1. **README presents the project as an officially completed Master’s thesis.**
   The “All Phases Complete” table and “Thesis ✅” row conflict with the clarification that
   the research topic was proposed but never officially approved. The README must present
   EDN-S as an independently developed implementation associated with a proposed research
   direction, not an approved university project.
2. **README results table contains incorrect numbers.**
   Actual values from `results/metrics/phase5_edsr_small_eval.json`:

   | Metric | README | Actual |
   |---|---|---|
   | EDSR SSIM | 0.72 | **0.7709** |
   | Bicubic SSIM | 0.61 | **0.6285** |
   | EDSR SAM | 0.058 | **0.0203** |
   | EDSR ERGAS | 2.14 | **0.0514** |

   PSNR (24.86 vs 21.83) and hallucination rate (6.89% vs 13.03%) are correct.
3. **ASAL wording overclaims a relationship.** “in collaboration with ASAL” implies a formal
   partnership. The package description and keywords (`asal`, `algerian-space-agency`) also
   imply affiliation. These must be neutralised.
4. **Version mismatch.** README says 1.0.1; PyPI and `pyproject.toml` say 1.0.2;
   `edns.__version__` says 1.0.0.
5. **Unverified performance claim.** “~50 s for 1024→4096 tile on Intel Iris Xe” is not
   supported by a saved benchmark. The available example reports 1.18 s for 256→1024.

### 3.2 Code defects

1. **`edns.utils` cannot be imported.**
   `src/edns/utils/helpers.py` uses `Optional` and `Path` without importing them.
   Verified failure:
   ```
   NameError: name 'Optional' is not defined
   ```
   This does not affect the CLI because `edns.utils` is not imported by the top-level package,
   but it is a real defect in the public package.
2. **Default normalization statistics path is not installable.**
   `DEFAULT_STATS_PATH` resolves to `<repo>/data/metadata/sentinel2/normalization_stats.json`.
   That file is not packaged in the wheel, so `pip install edns` + `edns-sr` fails at
   preprocessing unless the user supplies statistics manually.
3. **Invalid optional dependency.** `pyproject.toml` declares `gui = ["tkinter"]`; `tkinter`
   is part of the Python standard library and is not a pip-installable distribution.
4. **Stale `setup.py`.** It declares version 1.0.0, the old repository URL
   (`MASTER2-A1`), and a placeholder email (`aimenachraff@example.com`), duplicating
   `pyproject.toml`.
5. **Author identity inconsistency.** `src/edns/__init__.py` and `LICENSE` use
   “Aimen Achraff”; `pyproject.toml` and the README use “Chekalil Aimene Charef Eddine”.

### 3.3 Documentation drift

1. `docs/reproducibility.md` says “Expected: 38 passed” (actual: 246), references
   `src/utils/set_seed.py` (does not exist), and lists split counts (12,450 / 1,380 / 1,420)
   that do not match the actual tiles (7,968 / 2,712 / 4,068).
2. No `CHANGELOG.md`, `CONTRIBUTING.md`, or `CITATION.cff`.
3. No installation, quick-start, dataset, training, evaluation, architecture, or API guides
   under `docs/` (the AGENTS.md tree lists some that do not exist).
4. README omits training instructions, dataset provenance, limitations, reproducibility,
   future work, and references.

### 3.4 Presentation

1. No banner or architecture diagram.
2. `results/figures/` and all `*.png` are ignored by `.gitignore`, so the GitHub repository
   contains no result images. A dedicated `docs/assets/` directory with explicit
   un-ignore rules is needed.
3. The existing 4-panel comparison has no labels.
4. Root-level clutter (tracked): `PHASE_0_REPORT.md`, `README-Phase3.md`,
   `STATUS-Phase3-Final.md`, `phase_2_complete.md`, `sr_app.py`, `run_sr_app.bat`,
   `test_3panel.py`, `test_bicubic_vs_edsr.py`, `test_city_forest.py`,
   `test_high_var_patch.py`, `test_model_quick.py`, `test_real_s2.py`.
   `AGENTS.md` and `opencode.json` are internal agent configuration.

### 3.5 Structure

- Two parallel code trees exist: the packaged `src/edns/` and the research tree
  `src/models/`, `src/evaluation/`, `src/hallucination/`, etc. They have drifted
  (different file sizes and hashes). This is a packaging split, not a bug, but it must be
  documented so contributors edit the right copy.
- `scripts/make_example_images.py` contains a hard-coded absolute Windows path.

---

## 4. Claims verified as correct

- EDSR-small parameter count: **1,518,764** (~1.52 M).
- Phase 5 PSNR: **24.8645 dB** vs bicubic **21.8329 dB** (+3.03 dB), 30 validation pairs.
- Phase 6 hallucination rate: **6.89%** (EDSR) vs **13.03%** (bicubic), 1,742 test tiles.
- Cohen’s d ≈ **6.19**; McNemar p < 0.001 for textural and spectral categories.
- 246 tests pass.
- PyPI package `edns` 1.0.2 is live.
- No model weights, GeoTIFFs, or large binaries are tracked by Git.

---

## 5. What will be preserved

The following will **not** be modified:

- EDSR-small architecture and parameter count.
- Training methodology, loss functions, optimizer, schedules, and configs.
- Completed experimental results and metric JSON files.
- Existing checkpoint files.
- Git history and branches.

Changes will be limited to documentation, metadata, presentation assets, and the two
non-behavioural code defects listed in §3.2 (missing imports; packaged normalization
statistics fallback), each of which leaves model output unchanged.
