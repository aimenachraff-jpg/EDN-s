# Phase 6 — Hallucination-Aware Evaluation

**Date:** 2026-11-18
**Status:** COMPLETE — hallucination detection on 1,742 clean test tiles
**Commit:** `7d3e118` (docs), `10875f4` (status table cleanup)

## Scope

Phase 6 applies the hallucination detection framework (`src/hallucination/detection.py`)
to real SR outputs on the test set, quantifying per-category hallucination rates for
EDSR-small vs bicubic. Test tiles (4,068) stayed local (never uploaded to Kaggle)
because hallucination detection is CPU-based (scipy/skimage) and the SR inference
runs locally on CPU (~1.05 s/tile for EDSR-small).

## Method

For each clean test tile (nan_fraction == 0.0):
1. Read HR tile (4-band 10m, 1024×1024) from `data/processed/sentinel2/<aoi>/test/`
2. Degrade → LR via `ALSAT2BDegradation.forward_ms_only` (Wald, seed 42) → [4,256,256]
3. SR via EDSR-small (best.pt, CPU) and bicubic ×4
4. `HallucinationDetector.detect_all(sr, hr)` for both → spectral / structural / textural
   (semantic skipped — no task model provided)
5. Append per-tile result to `results/metrics/phase6_per_tile.jsonl` (resume-safe)

## Data

- 1,823 clean test tiles across 8 AOIs (4,068 total; 2,245 excluded for non-finite pixels)
- 1,742 processed (81 skipped: missing files or non-finite on read — same exclusion
  rule as Phase 5, never filled)

## Results (any-hallucination rate)

| Category   | EDSR-small | Bicubic ×4 |
|------------|------------|------------|
| Spectral   | 2.53% (44) | 3.62% (63) |
| Structural | 0.11% (2)  | 0.00% (0)  |
| Textural   | 4.54% (79) | 9.82% (171)|
| **ANY**    | **6.89%**  | **13.03%** |

- EDSR-small produces roughly **half** the hallucination rate of bicubic (6.89% vs 13.03%).
- Textural hallucinations are the dominant category for both models and show the
  largest gap (4.54% vs 9.82%) — consistent with EDSR's learned texture prior vs
  bicubic's smoothing/overshoot.
- Spectral hallucinations are rare for both (2.53% vs 3.62%), indicating both
  respect band correlations reasonably well.
- Structural hallucinations are negligible (0.11% EDSR, 0% bicubic).

## Thesis headline (Phases 5 + 6)

- **Fidelity:** EDSR-small 24.86 dB vs bicubic 21.83 dB (+3.03 dB)
- **Hallucination rate:** EDSR-small 6.89% vs bicubic 13.03% (−6.14 pp)
- EDSR improves both fidelity AND reduces hallucinations vs the non-learned baseline.

## Artifacts

- `results/metrics/phase6_per_tile.jsonl` (1,742 per-tile records: classification,
  detected_types, confidences, per-category metrics for both models)
- `results/metrics/phase6_hallucination.json` (aggregated summary)
- `scripts/phase6_hallucination.py` (resume-safe, `--max_tiles` cap for pilots)

## H2 / downstream tasks (deferred)

- **Semantic detection** skipped (no task model provided). `FixedBackboneEvaluator`
  (`src/downstream/tasks.py`) is available but requires training a task model on HR.
- **Expert-annotation validation** (`validate_hallucination_metrics`) requires manual
  labels from domain experts — out of scope for the automated run.
- Both are documented as future work; the detection framework and evaluator are
  implemented and ready to use if labels/models become available.

## Next

Thesis writing & final validation (Phase 9). Statistical analysis (Phase 8) if needed.
All core results are in `results/metrics/`.
