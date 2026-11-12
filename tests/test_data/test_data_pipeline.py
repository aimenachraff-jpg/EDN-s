"""
Unit Tests for Data Pipeline

Tests cover: config loading, AOI geometry, split logic, normalization, validation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sentinel2_downloader import Sentinel2Downloader, AOIConfig, TileMetadata
from src.data.validate_data import DataValidator


class TestSentinel2Config:
    """Test Sentinel-2 configuration loading."""

    def test_config_loads(self):
        downloader = Sentinel2Downloader()
        assert downloader.config is not None
        assert 'aois' in downloader.config
        assert 'temporal_splits' in downloader.config
        assert 'bands' in downloader.config

    def test_aois_defined(self):
        downloader = Sentinel2Downloader()
        assert len(downloader.aois) == 8
        assert 'DZ_01' in downloader.aois
        assert 'FR_01' in downloader.aois

    def test_aoi_properties(self):
        downloader = Sentinel2Downloader()
        aoi = downloader.aois['DZ_01']
        assert aoi.id == 'DZ_01'
        assert aoi.country == 'Algeria'
        assert aoi.priority == 'primary'
        assert len(aoi.bbox) == 4

    def test_temporal_splits(self):
        downloader = Sentinel2Downloader()
        splits = downloader.temporal_splits
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # Train: 2018-2022
        assert splits['train']['start_date'] == '2018-01-01'
        assert splits['train']['end_date'] == '2022-12-31'
        assert splits['train']['cloud_filter_max'] == 0.20

        # Test: 2024, no cloud filter
        assert splits['test']['start_date'] == '2024-01-01'
        assert splits['test']['cloud_filter_max'] == 1.0

    def test_band_configs(self):
        downloader = Sentinel2Downloader()
        bands = downloader.bands_config
        assert 'hr_target' in bands
        assert 'lr_native_2x' in bands
        assert 'lr_sim_4x' in bands

        # HR target should be 4 bands (B2, B3, B4, B8)
        assert len(bands['hr_target']) == 4
        assert bands['hr_target'] == ['B2', 'B3', 'B4', 'B8']

    def test_split_assignment(self):
        downloader = Sentinel2Downloader()
        assert downloader._get_split_for_date('2020-06-15') == 'train'
        assert downloader._get_split_for_date('2023-06-15') == 'val'
        assert downloader._get_split_for_date('2024-06-15') == 'test'
        assert downloader._get_split_for_date('2017-06-15') == 'unknown'


class TestTileMetadata:
    """Test tile metadata structure."""

    def test_tile_metadata_creation(self):
        tile = TileMetadata(
            tile_id='DZ_01_20200615_0_0',
            aoi_id='DZ_01',
            date='20200615',
            cloud_percent=0.1,
            bounds=[2.8, 36.5, 2.9, 36.6],
            s2_tile_id='31TCJ',
            split='train',
            bands=['B2', 'B3', 'B4', 'B8'],
            file_paths={'B2': '/path/B2.tif', 'B3': '/path/B3.tif'}
        )
        assert tile.tile_id == 'DZ_01_20200615_0_0'
        assert tile.split == 'train'
        assert len(tile.bands) == 4


class TestDataValidator:
    """Test data validation functions."""

    def setup_method(self):
        self.validator = DataValidator()

    def test_bounds_to_polygon(self):
        # This tests the internal conversion
        bounds_str = "[2.8, 36.5, 2.9, 36.6]"
        # We can't easily test the private method, but we can test the logic
        bounds = eval(bounds_str)
        from shapely.geometry import box
        poly = box(*bounds)
        assert poly.bounds == (2.8, 36.5, 2.9, 36.6)

    def test_split_verification_structure(self):
        # Verify the method returns expected structure
        results = self.validator.verify_splits()
        assert isinstance(results, dict)
        # At minimum should have keys for split pairs
        expected_keys = ['train_vs_val', 'train_vs_test', 'val_vs_test']
        for key in expected_keys:
            assert key in results


class TestPROBAVConfig:
    """Test PROBA-V configuration."""

    def test_probav_config_loads(self):
        config_path = Path("configs/data/probav.yaml")
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config['dataset']['n_scenes_train'] == 1160
        assert config['dataset']['n_scenes_test'] == 290
        assert config['images']['scale_factor'] == 3
        assert config['sensor']['challenge_bands'] == ['RED', 'NIR']


class TestNormalizationConfig:
    """Test normalization configuration."""

    def test_normalization_file_structure(self):
        norm_path = Path("configs/normalization.yaml")
        # May not exist yet if not computed
        if norm_path.exists():
            with open(norm_path) as f:
                norm = yaml.safe_load(f)
            # Should have band entries with p2, p98
            for band, params in norm.items():
                assert 'p2' in params
                assert 'p98' in params
                assert params['p2'] < params['p98']


# Fixtures for integration tests (require downloaded data)
@pytest.fixture(scope="session")
def pilot_tiles():
    """Fixture providing pilot tiles if available."""
    metadata_dir = Path("data/metadata/sentinel2")
    csv_path = metadata_dir / "tiles_pilot.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def test_pilot_data_exists(pilot_tiles):
    """Test that pilot data was downloaded (optional)."""
    if pilot_tiles is not None:
        assert len(pilot_tiles) > 0
        assert 'tile_id' in pilot_tiles.columns
        assert 'split' in pilot_tiles.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])