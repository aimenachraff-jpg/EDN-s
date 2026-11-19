"""
Unit tests for normalization statistics (train-split-only policy).

Policy: per-band mean/std computed ONLY over train-split tiles to avoid
leakage into val/test evaluation. NaN pixels (nodata) must be excluded.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.normalization import (
    compute_stats,
    compute_percentile_stats,
    normalize,
    normalize_percentile,
    save_stats,
    load_stats,
)


class TestComputeStats:
    def test_matches_numpy_single_band(self):
        rng = np.random.default_rng(42)
        tiles = [rng.normal(0.3, 0.1, (64, 64)).astype(np.float32) for _ in range(5)]
        stats = compute_stats(tiles, band_names=["B02"])
        assert set(stats.keys()) == {"B02"}
        stacked = np.concatenate([t.ravel() for t in tiles])
        assert stats["B02"]["mean"] == pytest.approx(float(stacked.mean()), abs=1e-6)
        assert stats["B02"]["std"] == pytest.approx(float(stacked.std()), abs=1e-6)

    def test_matches_numpy_multiband(self):
        rng = np.random.default_rng(0)
        # 4-band tiles [H, W, 4]
        tiles = [rng.normal(0.2, 0.05, (32, 32, 4)).astype(np.float32) for _ in range(3)]
        stats = compute_stats(tiles, band_names=["B02", "B03", "B04", "B08"])
        assert set(stats.keys()) == {"B02", "B03", "B04", "B08"}
        stacked = np.concatenate([t.reshape(-1, 4) for t in tiles], axis=0)
        for i, b in enumerate(["B02", "B03", "B04", "B08"]):
            assert stats[b]["mean"] == pytest.approx(float(stacked[:, i].mean()), abs=1e-6)
            assert stats[b]["std"] == pytest.approx(float(stacked[:, i].std()), abs=1e-6)

    def test_excludes_nan(self):
        tiles = [np.array([[0.1, np.nan], [0.3, np.nan]], dtype=np.float32)]
        stats = compute_stats(tiles, band_names=["B02"])
        assert stats["B02"]["mean"] == pytest.approx(0.2, abs=1e-6)
        assert stats["B02"]["n_pixels"] == 2

    def test_counts_reported(self):
        rng = np.random.default_rng(1)
        tiles = [rng.normal(0.5, 0.1, (16, 16)).astype(np.float32) for _ in range(2)]
        stats = compute_stats(tiles, band_names=["B02"])
        assert stats["B02"]["n_pixels"] == 16 * 16 * 2


class TestStreamingConsistency:
    def test_streaming_equals_batch(self):
        """Incremental updates must equal the one-shot numpy result."""
        rng = np.random.default_rng(7)
        tiles = [rng.normal(0.4, 0.2, (50, 50)).astype(np.float64) for _ in range(20)]
        streamed = compute_stats(iter(tiles), band_names=["B02"])
        batch = np.concatenate([t.ravel() for t in tiles])
        assert streamed["B02"]["mean"] == pytest.approx(float(batch.mean()), abs=1e-9)
        assert streamed["B02"]["std"] == pytest.approx(float(batch.std()), abs=1e-9)


class TestPersistence:
    def test_json_roundtrip(self, tmp_path):
        rng = np.random.default_rng(3)
        tiles = [rng.normal(0.3, 0.1, (16, 16, 2)).astype(np.float32)]
        stats = compute_stats(tiles, band_names=["B02", "B08"])
        out = tmp_path / "norm_stats.json"
        save_stats(stats, out, extra={"split": "train"})
        loaded = load_stats(out)
        assert loaded["B02"]["mean"] == pytest.approx(stats["B02"]["mean"])
        assert loaded["B08"]["std"] == pytest.approx(stats["B08"]["std"])
        meta = json.loads(out.read_text())
        assert meta["metadata"]["split"] == "train"

    def test_roundtrip_file_format(self, tmp_path):
        stats = compute_stats([np.full((4, 4), 0.5, np.float32)], band_names=["B02"])
        out = tmp_path / "stats.json"
        save_stats(stats, out)
        raw = json.loads(out.read_text())
        assert "metadata" in raw and "bands" in raw
        assert raw["bands"]["B02"]["mean"] == pytest.approx(0.5)


class TestNormalize:
    def test_zero_mean_unit_std(self):
        rng = np.random.default_rng(11)
        tiles = [rng.normal(0.3, 0.1, (32, 32)).astype(np.float32) for _ in range(4)]
        stats = compute_stats(tiles, band_names=["B02"])
        all_x = np.concatenate([t.ravel() for t in tiles])
        xn = normalize(all_x, stats["B02"])
        assert xn.mean() == pytest.approx(0.0, abs=1e-5)
        assert xn.std() == pytest.approx(1.0, abs=1e-3)

    def test_nan_stays_nan(self):
        stats = {"mean": 0.5, "std": 0.2, "n_pixels": 10}
        x = np.array([[0.5, np.nan]], dtype=np.float32)
        xn = normalize(x, stats)
        assert np.isnan(xn[0, 1])
        assert xn[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_rejects_zero_std(self):
        stats = {"mean": 0.5, "std": 0.0, "n_pixels": 10}
        with pytest.raises(ValueError):
            normalize(np.zeros((2, 2), np.float32), stats)


class TestPercentileStats:
    def test_exact_on_small_tiles(self):
        """When a tile has <= samples_per_tile valid pixels, percentiles are exact."""
        tiles = [np.arange(100, dtype=np.float32).reshape(10, 10)]
        stats = compute_percentile_stats(tiles, band_names=["B02"], lower=2, upper=98)
        expected_lo = float(np.percentile(np.arange(100, dtype=np.float32), 2))
        expected_hi = float(np.percentile(np.arange(100, dtype=np.float32), 98))
        assert stats["B02"]["p2"] == pytest.approx(expected_lo)
        assert stats["B02"]["p98"] == pytest.approx(expected_hi)
        # mean/std also recorded for reference
        assert stats["B02"]["mean"] == pytest.approx(49.5)

    def test_multiband(self):
        rng = np.random.default_rng(5)
        tiles = [rng.normal(0.3, 0.1, (20, 20, 2)).astype(np.float32) for _ in range(3)]
        stats = compute_percentile_stats(tiles, band_names=["B02", "B08"])
        assert set(stats.keys()) == {"B02", "B08"}
        stacked = np.concatenate([t.reshape(-1, 2) for t in tiles])
        for i, b in enumerate(["B02", "B08"]):
            assert stats[b]["p2"] == pytest.approx(float(np.percentile(stacked[:, i], 2)), abs=1e-5)

    def test_nan_excluded(self):
        tiles = [np.array([[np.nan] * 8 + [0.0, 1.0]], dtype=np.float32)]
        stats = compute_percentile_stats(tiles, band_names=["B02"])
        # numpy percentile of [0,1] uses linear interpolation
        assert stats["B02"]["p2"] == pytest.approx(0.02)
        assert stats["B02"]["p98"] == pytest.approx(0.98)


class TestNormalizePercentile:
    def test_output_range(self):
        stats = {"p2": 0.1, "p98": 0.9}
        x = np.array([0.0, 0.1, 0.5, 0.9, 1.0, 2.0], dtype=np.float32)
        xn = normalize_percentile(x, stats)
        assert xn.min() == 0.0 and xn.max() == 1.0
        assert xn[1] == 0.0   # at p2
        assert xn[2] == pytest.approx(0.5)
        assert xn[3] == pytest.approx(1.0)  # at p98 (float32 rounding tolerated)

    def test_nan_stays_nan(self):
        stats = {"p2": 0.1, "p98": 0.9}
        x = np.array([0.5, np.nan], dtype=np.float32)
        xn = normalize_percentile(x, stats)
        assert np.isnan(xn[1])
        assert xn[0] == pytest.approx(0.5)

    def test_rejects_degenerate_range(self):
        with pytest.raises(ValueError):
            normalize_percentile(np.zeros(3, np.float32), {"p2": 0.5, "p98": 0.5})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
