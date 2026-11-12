"""
Data Validation and Split Verification

Verifies data integrity, prevents geographical/temporal leakage,
and validates preprocessing outputs.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import rasterio
import yaml

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data pipeline outputs and splits."""

    def __init__(self, config_path: str = "configs/data/sentinel2.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.processing = self.config['processing']
        self.split_cfg = self.config['split_verification']
        self.metadata_dir = Path("data/metadata/sentinel2")

    def load_tile_index(self, split: str) -> pd.DataFrame:
        """Load tile metadata CSV."""
        csv_path = self.metadata_dir / f"tiles_{split}.csv"
        if not csv_path.exists():
            # Try pilot
            csv_path = self.metadata_dir / "tiles_pilot.csv"
        return pd.read_csv(csv_path)

    def check_spatial_overlap(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               buffer: float = 0.0) -> List[Tuple]:
        """Check for spatial overlap between two tile sets."""
        overlaps = []

        # Convert bounds to geometries
        def bounds_to_poly(bounds_str):
            bounds = eval(bounds_str)  # [min_x, min_y, max_x, max_y]
            return box(*bounds)

        gdf1 = gpd.GeoDataFrame(df1.copy(), geometry=df1['bounds'].apply(bounds_to_poly), crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame(df2.copy(), geometry=df2['bounds'].apply(bounds_to_poly), crs="EPSG:4326")

        if buffer > 0:
            gdf1['geometry'] = gdf1.buffer(buffer)
            gdf2['geometry'] = gdf2.buffer(buffer)

        # Spatial join
        joined = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')

        for _, row in joined.iterrows():
            overlaps.append((row['tile_id_left'], row['tile_id_right']))

        return overlaps

    def check_temporal_overlap(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[Tuple]:
        """Check for same-date tiles across splits."""
        overlaps = []
        dates1 = set(df1['date'].unique())
        dates2 = set(df2['date'].unique())
        common_dates = dates1 & dates2

        for date in common_dates:
            tiles1 = df1[df1['date'] == date]['tile_id'].tolist()
            tiles2 = df2[df2['date'] == date]['tile_id'].tolist()
            for t1 in tiles1:
                for t2 in tiles2:
                    overlaps.append((t1, t2, date))

        return overlaps

    def check_granule_overlap(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[Tuple]:
        """Check for tiles from same S2 granule in different splits."""
        overlaps = []
        granules1 = df1.groupby('s2_tile_id')['tile_id'].apply(list).to_dict()
        granules2 = df2.groupby('s2_tile_id')['tile_id'].apply(list).to_dict()

        common_granules = set(granules1.keys()) & set(granules2.keys())
        for granule in common_granules:
            for t1 in granules1[granule]:
                for t2 in granules2[granule]:
                    overlaps.append((t1, t2, granule))

        return overlaps

    def verify_splits(self) -> Dict[str, any]:
        """Run all split verification checks."""
        results = {}

        # Load all splits
        splits = ['train', 'val', 'test']
        dfs = {}
        for split in splits:
            try:
                dfs[split] = self.load_tile_index(split)
            except FileNotFoundError:
                logger.warning(f"Split {split} metadata not found")
                dfs[split] = pd.DataFrame()

        # Check pairwise
        split_pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]

        for split1, split2 in split_pairs:
            if dfs[split1].empty or dfs[split2].empty:
                results[f'{split1}_vs_{split2}'] = {'skipped': True}
                continue

            # Spatial
            spatial_overlaps = self.check_spatial_overlap(
                dfs[split1], dfs[split2],
                buffer=self.split_cfg['spatial_buffer_pixels'] * 10 / 111000  # approx degrees
            )

            # Temporal
            temporal_overlaps = self.check_temporal_overlap(dfs[split1], dfs[split2])

            # Granule
            granule_overlaps = self.check_granule_overlap(dfs[split1], dfs[split2])

            results[f'{split1}_vs_{split2}'] = {
                'spatial_overlaps': len(spatial_overlaps),
                'spatial_details': spatial_overlaps[:10],  # First 10
                'temporal_overlaps': len(temporal_overlaps),
                'temporal_details': temporal_overlaps[:10],
                'granule_overlaps': len(granule_overlaps),
                'granule_details': granule_overlaps[:10],
                'passed': len(spatial_overlaps) == 0 and len(temporal_overlaps) == 0 and len(granule_overlaps) == 0
            }

        return results

    def validate_tile_integrity(self, split: str) -> Dict[str, any]:
        """Validate individual tile files."""
        df = self.load_tile_index(split)
        results = {
            'total_tiles': len(df),
            'missing_files': [],
            'shape_mismatches': [],
            'value_ranges': {},
            'nodata_counts': {}
        }

        for _, row in df.iterrows():
            for band in ['B2', 'B3', 'B4', 'B8']:
                col = f'path_{band}'
                if col in row and pd.notna(row[col]):
                    fpath = row[col]
                    if not os.path.exists(fpath):
                        results['missing_files'].append(fpath)
                    else:
                        try:
                            with rasterio.open(fpath) as src:
                                data = src.read(1)
                                h, w = data.shape
                                expected = self.processing['tile_size_10m']
                                if h != expected or w != expected:
                                    results['shape_mismatches'].append((fpath, (h, w)))

                                # Value range
                                if band not in results['value_ranges']:
                                    results['value_ranges'][band] = {'min': float('inf'), 'max': -float('inf')}
                                results['value_ranges'][band]['min'] = min(results['value_ranges'][band]['min'], float(data.min()))
                                results['value_ranges'][band]['max'] = max(results['value_ranges'][band]['max'], float(data.max()))

                                # Nodata
                                nodata = np.sum(data == 0)
                                if nodata > 0:
                                    results['nodata_counts'][fpath] = int(nodata)
                        except Exception as e:
                            results['missing_files'].append(f"{fpath}: {str(e)}")

        return results

    def validate_normalization(self) -> Dict[str, any]:
        """Validate normalization parameters."""
        norm_path = Path("configs/normalization.yaml")
        if not norm_path.exists():
            return {'error': 'Normalization file not found'}

        with open(norm_path) as f:
            import yaml
            norm_params = yaml.safe_load(f)

        results = {'bands': list(norm_params.keys()), 'params': norm_params}

        # Check that p2 < p98 for all bands
        for band, params in norm_params.items():
            if params['p2'] >= params['p98']:
                results[f'{band}_error'] = 'p2 >= p98'

        return results


def main():
    logging.basicConfig(level=logging.INFO)
    validator = DataValidator()

    # Verify splits
    logger.info("Verifying splits...")
    split_results = validator.verify_splits()
    for check, result in split_results.items():
        if result.get('skipped'):
            logger.warning(f"{check}: SKIPPED (missing metadata)")
        elif result['passed']:
            logger.info(f"{check}: PASSED")
        else:
            logger.error(f"{check}: FAILED - spatial:{result['spatial_overlaps']}, temporal:{result['temporal_overlaps']}, granule:{result['granule_overlaps']}")

    # Validate tile integrity (for available splits)
    for split in ['train', 'val', 'test', 'pilot']:
        try:
            logger.info(f"Validating {split} tiles...")
            tile_results = validator.validate_tile_integrity(split)
            logger.info(f"  Total: {tile_results['total_tiles']}, Missing: {len(tile_results['missing_files'])}, Shape mismatches: {len(tile_results['shape_mismatches'])}")
        except FileNotFoundError:
            pass

    # Validate normalization
    logger.info("Validating normalization...")
    norm_results = validator.validate_normalization()
    logger.info(f"Normalization bands: {norm_results.get('bands', 'none')}")


if __name__ == "__main__":
    main()