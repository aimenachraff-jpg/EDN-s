"""
Normalization statistics for Sentinel-2 SR training.

Policy (documented ruling, Phase 3): per-band mean/std are computed ONLY
over train-split tiles. Val/test statistics are never mixed in, to avoid
leakage into hallucination-aware evaluation.

Implementation notes:
  - NaN pixels (nodata) are excluded from statistics and preserved by
    normalize().
  - Statistics are accumulated in float64 with a streaming (sum/sumsq)
    algorithm so full archives can be processed tile-by-tile without
    loading everything into memory.
  - Stats are persisted as JSON with a metadata block (split, bands,
    provenance) for reproducibility.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_stats(
    tiles: Iterable[np.ndarray],
    band_names: list,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Compute per-band mean/std over an iterable of tiles.

    Tiles are either [H, W] (single band, len(band_names)==1) or
    [H, W, C] with C == len(band_names). NaN pixels are excluded.
    Works with generators (streaming); memory use is O(1) per tile.
    """
    n_bands = len(band_names)
    sums = np.zeros(n_bands, dtype=np.float64)
    sumsq = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.int64)

    for tile in tiles:
        arr = np.asarray(tile, dtype=np.float64)
        if arr.ndim == 2:
            if n_bands != 1:
                raise ValueError(f"Got 2D tile but {n_bands} band names")
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[2] != n_bands:
                raise ValueError(
                    f"Tile has {arr.shape[2]} channels, expected {n_bands}"
                )
        else:
            raise ValueError(f"Tile must be 2D or 3D, got shape {arr.shape}")

        for b in range(n_bands):
            ch = arr[:, :, b]
            valid = ch[np.isfinite(ch)]
            sums[b] += valid.sum()
            sumsq[b] += np.square(valid).sum()
            counts[b] += valid.size

    stats: Dict[str, Dict[str, Union[float, int]]] = {}
    for b, name in enumerate(band_names):
        if counts[b] == 0:
            raise ValueError(f"Band {name}: no valid (finite) pixels found")
        mean = sums[b] / counts[b]
        var = max(sumsq[b] / counts[b] - mean * mean, 0.0)
        stats[name] = {
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "n_pixels": int(counts[b]),
        }
        logger.info(
            f"Band {name}: mean={mean:.6f} std={np.sqrt(var):.6f} "
            f"n={counts[b]:,}"
        )
    return stats


def save_stats(
    stats: Dict[str, Dict[str, Union[float, int]]],
    path: Path,
    extra: Optional[dict] = None,
) -> Path:
    """Persist stats as JSON with a metadata block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "split": "train",
            "nan_excluded": True,
            "dtype_note": "accumulated in float64",
            **(extra or {}),
        },
        "bands": stats,
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Stats saved -> {path}")
    return path


def load_stats(path: Path) -> Dict[str, Dict[str, Union[float, int]]]:
    """Load stats saved by save_stats; returns the bands dict."""
    payload = json.loads(Path(path).read_text())
    return payload["bands"]


def normalize(arr: np.ndarray, band_stats: Dict[str, Union[float, int]]) -> np.ndarray:
    """Apply (x - mean) / std to a single band. NaN stays NaN."""
    std = band_stats["std"]
    if std == 0.0:
        raise ValueError("Cannot normalize a zero-variance band (std=0)")
    with np.errstate(invalid="ignore"):
        return (arr.astype(np.float32) - band_stats["mean"]) / std


def compute_percentile_stats(
    tiles: Iterable[np.ndarray],
    band_names: list,
    lower: float = 2.0,
    upper: float = 98.0,
    samples_per_tile: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Per-band percentile statistics (config method: percentile clip 2-98).

    Tiles larger than `samples_per_tile` valid pixels per band are
    subsampled (deterministic seed) to bound memory; smaller tiles are
    used exactly. Mean/std are also recorded for reference.
    """
    n_bands = len(band_names)
    rng = np.random.default_rng(seed)
    samples: list = [[] for _ in range(n_bands)]
    sums = np.zeros(n_bands, dtype=np.float64)
    sumsq = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.int64)

    for tile in tiles:
        arr = np.asarray(tile, dtype=np.float64)
        if arr.ndim == 2:
            if n_bands != 1:
                raise ValueError(f"Got 2D tile but {n_bands} band names")
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3 or arr.shape[2] != n_bands:
            raise ValueError(f"Tile shape {arr.shape} incompatible with {n_bands} bands")

        for b in range(n_bands):
            ch = arr[:, :, b]
            valid = ch[np.isfinite(ch)]
            if valid.size == 0:
                continue
            if valid.size > samples_per_tile:
                idx = rng.choice(valid.size, size=samples_per_tile, replace=False)
                valid_s = valid[idx]
            else:
                valid_s = valid
            samples[b].append(valid_s.astype(np.float32))
            sums[b] += valid.sum()
            sumsq[b] += np.square(valid).sum()
            counts[b] += valid.size

    stats: Dict[str, Dict[str, Union[float, int]]] = {}
    for b, name in enumerate(band_names):
        if counts[b] == 0 or not samples[b]:
            raise ValueError(f"Band {name}: no valid (finite) pixels found")
        pooled = np.concatenate(samples[b])
        mean = sums[b] / counts[b]
        var = max(sumsq[b] / counts[b] - mean * mean, 0.0)
        stats[name] = {
            "p2": float(np.percentile(pooled, lower)),
            "p98": float(np.percentile(pooled, upper)),
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "n_pixels": int(counts[b]),
            "n_samples_percentile": int(pooled.size),
            "percentile_keys": [lower, upper],
        }
        logger.info(
            f"Band {name}: p{lower:g}={stats[name]['p2']:.6f} "
            f"p{upper:g}={stats[name]['p98']:.6f} mean={mean:.6f} "
            f"std={np.sqrt(var):.6f} (percentiles from {pooled.size:,} samples)"
        )
    return stats


def normalize_percentile(
    arr: np.ndarray,
    band_stats: Dict[str, Union[float, int]],
) -> np.ndarray:
    """Clip to [p2, p98] and rescale to [0, 1] (config normalization method)."""
    lo, hi = band_stats["p2"], band_stats["p98"]
    if not hi > lo:
        raise ValueError(f"Degenerate percentile range: p2={lo} p98={hi}")
    with np.errstate(invalid="ignore"):
        out = (arr.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)
