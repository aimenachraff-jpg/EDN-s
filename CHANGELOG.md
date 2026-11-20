# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Documentation and packaging pass. No model architecture, training methodology or reported
result was changed.

### Added

- Documentation suite: installation, quick start, dataset, training, evaluation,
  architecture, API reference, reproducibility and limitations.
- `README.md` rewritten with verified metrics, the EDN-S / research distinction, and an
  explicit academic-status note.
- Repository visuals under `docs/assets/`: banner, results gallery (LR / bicubic / EDN-S /
  HR, plus detail crops), architecture and evaluation diagram, and metrics charts.
- Generation scripts: `scripts/make_banner.py`, `scripts/make_results_gallery.py`,
  `scripts/make_architecture_diagram.py`, `scripts/make_metrics_chart.py`.
- `CHANGELOG.md`, `CONTRIBUTING.md` and `CITATION.cff`.
- `docs/repository_audit.md` recording the verified repository state.

### Fixed

- Missing `Optional` and `Path` imports in `src/edns/utils/helpers.py`.
- `run_sr(compute_hallucination=True)` read a non-existent `any_hallucination` key and
  looked for categories at the wrong level of the `detect_all()` result. The report now
  uses `detect_all()["details"]` and `detected_types`, and the CLI summary prints the real
  per-category results.
- Normalisation statistics and the default model config are now resolved from the
  installed package before falling back to a source checkout, so a pip-installed CLI can
  start.
- `scripts/make_example_images.py` no longer contains a hard-coded absolute path.
- `setup.py` is a compatibility shim; metadata lives in `pyproject.toml`.

### Changed

- Version aligned to `1.0.2` across `pyproject.toml`, `edns.__version__` and the README.
- `pyproject.toml` description, keywords, authors, URLs, extras and package data updated.
- The invalid `gui = ["tkinter"]` extra was removed; `full` now means `edns[dev,cloud]`.
- LICENSE copyright holder updated to the canonical author name.
- `.gitignore` allows `docs/assets/` images to be tracked.

## [1.0.2] - 2026-11-19

### Added

- Hallucination detection in the CLI (`--hallucination`, `--hr`, `--report`) and in the
  Python API (`compute_hallucination`, `hr_reference_path`,
  `hallucination_report_path`).

### Changed

- Package URLs, description and keywords updated for the public repository.
- Version bumped to 1.0.2.

## [1.0.1] - 2026-11-19

### Fixed

- Author name and GitHub URLs corrected in package metadata.

## [1.0.0] - 2026-11-19

### Added

- Initial PyPI release of `edns`.
- EDSR-small model for four-band ×4 satellite super-resolution.
- Tiled inference with overlap blending and GeoTIFF output.
- Evaluation metrics: PSNR, MSE, MAE, SSIM, SAM, ERGAS, SCI, wavelet HH energy, GLCM
  entropy and TUS.
- Reference-based hallucination detection (spectral, structural, textural, semantic
  interface).
- `edns-sr` command-line interface.
