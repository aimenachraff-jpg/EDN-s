# Contributing

Thanks for your interest in EDN-S. This project is a research-oriented implementation, so
contributions are most useful when they keep results reproducible and claims honest.

## Before you start

- Read [`docs/limitations.md`](docs/limitations.md) and the “EDN-S and the broader
  research project” section of the README. Changes must not imply real ALSAT-2B data,
  official ASAL involvement, or an approved thesis.
- Do not commit datasets, checkpoints, credentials or large binaries.
- Do not change reported metric values without regenerating the corresponding
  `results/metrics/*.json` file and explaining the change.

## Development setup

```bash
git clone https://github.com/aimenachraff-jpg/EDN-s.git
cd EDN-s
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python 3.10+, PyTorch 2.x. `rasterio` needs a wheel for your platform.

## Running the tests

```bash
python -m pytest tests/ -q
```

The suite should pass before and after your change. If you add behaviour, add a test that
fails without your change.

Useful subsets:

```bash
python -m pytest tests/test_models.py -q
python -m pytest tests/test_inference.py -q
python -m pytest tests/test_metrics.py -q
python -m pytest tests/test_hallucination.py -q
```

## Code style

- Follow the existing style; the code is formatted with `black` (line length 100) and
  checked with `flake8`.
- Keep public functions typed and documented.
- Prefer small, focused changes over broad refactors.
- Do not reformat unrelated files in the same commit.

```bash
black src tests scripts
flake8 src tests scripts
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/edns/` | Installable package (CLI, inference, metrics, hallucination detection) |
| `src/models`, `src/training`, `src/preprocessing`, `src/degradation`, `src/downstream`, `src/evaluation` | Research and training tree |
| `configs/` | Model and degradation configurations |
| `scripts/` | Data preparation, evaluation and documentation-figure scripts |
| `results/` | Verified metrics, tables and figures |
| `docs/` | Guides, phase reports and audit |
| `tests/` | Unit tests |

The package and research trees share model logic. If you change shared behaviour, update
both and note it in the pull request.

## Data and checkpoints

- Raw data, tiles and checkpoints are not distributed with the repository.
- Do not add a checkpoint unless the maintainer explicitly asks for a release asset.
- If a change depends on a specific dataset state, document the provenance and the
  filtering rule (for example `nan_fraction == 0.0`).

## Reporting a bug

Include:

1. What you ran (command or minimal snippet).
2. What you expected.
3. What happened, including the full traceback.
4. Environment: OS, Python, PyTorch, `edns` version.
5. Whether a checkpoint was involved, and its path (not the file itself).

## Pull request checklist

- [ ] Tests pass locally.
- [ ] New behaviour has a test.
- [ ] Documentation updated if the public API or results changed.
- [ ] No secrets, datasets, checkpoints or large binaries added.
- [ ] Claims remain limited to what was measured.
- [ ] `CHANGELOG.md` updated for user-visible changes.

## Scope

Out of scope for this repository unless discussed first:

- presenting EDN-S as a novel architecture;
- claims about real ALSAT-2B performance;
- reference-free hallucination claims;
- semantic or downstream-task results without a trained task model;
- changes that silently alter reported metrics.
