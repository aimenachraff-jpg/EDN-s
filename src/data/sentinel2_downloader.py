"""
Sentinel-2 Data Download and Preprocessing Module

Uses Planetary Computer STAC API for Sentinel-2 L2A data access.
Implements AOI-based search, download, tiling, normalization, and splitting.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import yaml

# Lazy imports for STAC/Planetary Computer (only needed at runtime, not for tests)
_planetary_computer = None
_pystac_client = None
_odc_stac = None
_xarray = None

def _import_stac_deps():
    """Import STAC dependencies lazily."""
    global _planetary_computer, _pystac_client, _odc_stac, _xarray
    if _planetary_computer is None:
        import planetary_computer as pc
        import pystac_client as pystac
        import odc.stac as odc_stac
        import xarray as xr
        _planetary_computer = pc
        _pystac_client = pystac
        _odc_stac = odc_stac
        _xarray = xr

logger = logging.getLogger(__name__)


@dataclass
class AOIConfig:
    """Area of Interest configuration."""
    id: str
    name: str
    country: str
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    land_cover: str
    size_km: List[int]
    priority: str


@dataclass
class TileMetadata:
    """Metadata for a single tile."""
    tile_id: str
    aoi_id: str
    date: str
    cloud_percent: float
    bounds: List[float]  # [min_x, min_y, max_x, max_y] in CRS
    s2_tile_id: str
    split: str
    bands: List[str]
    file_paths: Dict[str, str]  # band -> path


class Sentinel2Downloader:
    """Download Sentinel-2 L2A data from Planetary Computer."""

    def __init__(self, config_path: str = "configs/data/sentinel2.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.aois = {k: AOIConfig(id=k, **v) for k, v in self.config['aois'].items()}
        self.temporal_splits = self.config['temporal_splits']
        self.bands_config = self.config['bands']
        self.processing = self.config['processing']
        self.download_cfg = self.config['download']

        # STAC client (lazy - only created when needed)
        self._catalog = None

        # Output directories
        self.raw_dir = Path("data/raw/sentinel2")
        self.interim_dir = Path("data/interim/sentinel2")
        self.processed_dir = Path("data/processed/sentinel2")
        self.metadata_dir = Path("data/metadata/sentinel2")

        for d in [self.raw_dir, self.interim_dir, self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def catalog(self):
        """Lazy STAC catalog initialization."""
        if self._catalog is None:
            _import_stac_deps()
            self._catalog = _pystac_client.Client.open(
                self.download_cfg['providers'][0]['endpoint'],
                modifier=_planetary_computer.sign_inplace
            )
        return self._catalog

    def _get_split_for_date(self, date_str: str) -> str:
        """Determine split based on acquisition date."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        for split_name, split_cfg in self.temporal_splits.items():
            start = datetime.strptime(split_cfg['start_date'], "%Y-%m-%d")
            end = datetime.strptime(split_cfg['end_date'], "%Y-%m-%d")
            if start <= date <= end:
                return split_name
        return "unknown"

    def search_scenes(self, aoi_id: str, start_date: str, end_date: str,
                      max_cloud: float = 0.2) -> List[Dict]:
        """Search for Sentinel-2 L2A scenes matching criteria."""
        aoi = self.aois[aoi_id]
        bbox = aoi.bbox  # [min_lon, min_lat, max_lon, max_lat]

        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud * 100}}
        )

        items = list(search.items())
        logger.info(f"Found {len(items)} scenes for AOI {aoi_id} ({start_date} to {end_date})")

        scenes = []
        for item in items:
            props = item.properties
            scenes.append({
                'id': item.id,
                'datetime': props['datetime'],
                'cloud_cover': props.get('eo:cloud_cover', 0),
                'bbox': item.bbox,
                'geometry': item.geometry,
                'assets': item.assets,
                's2_tile_id': props.get('s2:mgrs_tile', 'unknown')
            })
        return scenes

    def download_scene_bands(self, scene: Dict, bands: List[str],
                             output_dir: Path) -> Dict[str, Path]:
        """Download specific bands for a scene using odc.stac."""
        _import_stac_deps()
        # Load with odc.stac (handles COG loading efficiently)
        xx = _odc_stac.load(
            [scene],
            bands=bands,
            crs="EPSG:4326",
            resolution=0.0001,  # ~10m at equator
            chunks={"x": 1024, "y": 1024}
        )

        # Save each band as COG
        file_paths = {}
        date_str = scene['datetime'].strftime("%Y%m%d")
        s2_tile = scene['s2_tile_id']

        for band in bands:
            if band in xx.data_vars:
                da = xx[band]
                # Create COG
                fname = f"{s2_tile}_{date_str}_{band}.tif"
                fpath = output_dir / fname
                da.rio.to_raster(fpath, driver="COG", compress="DEFLATE")
                file_paths[band] = str(fpath)

        return file_paths

    def tile_scene(self, scene_files: Dict[str, Path], aoi_id: str,
                   date_str: str, split: str) -> List[TileMetadata]:
        """Tile a scene into overlapping tiles with metadata."""
        aoi = self.aois[aoi_id]
        tile_size = self.processing['tile_size_10m']
        overlap = self.processing['overlap']
        stride = tile_size - overlap

        tiles = []
        # Read first band to get dimensions
        with rasterio.open(scene_files[self.bands_config['hr_target'][0]]) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

        # Generate tile windows
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                bounds = rasterio.windows.bounds(window, transform)

                tile_id = f"{aoi_id}_{date_str}_{x}_{y}"
                file_paths = {}

                # Read and save each band for this tile
                for band_name, fpath in scene_files.items():
                    with rasterio.open(fpath) as src:
                        tile_data = src.read(1, window=window)
                        tile_transform = src.window_transform(window)

                    # Save tile
                    tile_fname = f"{tile_id}_{band_name}.tif"
                    tile_fpath = self.processed_dir / split / tile_fname
                    tile_fpath.parent.mkdir(parents=True, exist_ok=True)

                    with rasterio.open(
                        tile_fpath, 'w',
                        driver='COG',
                        compress='DEFLATE',
                        height=tile_size,
                        width=tile_size,
                        count=1,
                        dtype=tile_data.dtype,
                        crs=crs,
                        transform=tile_transform
                    ) as dst:
                        dst.write(tile_data, 1)

                    file_paths[band_name] = str(tile_fpath)

                # Cloud percent for this tile (approximate from scene)
                cloud_percent = 0.0  # TODO: compute from SCL band

                tiles.append(TileMetadata(
                    tile_id=tile_id,
                    aoi_id=aoi_id,
                    date=date_str,
                    cloud_percent=cloud_percent,
                    bounds=list(bounds),
                    s2_tile_id=scene['s2_tile_id'] if 's2_tile_id' in scene else 'unknown',
                    split=split,
                    bands=list(file_paths.keys()),
                    file_paths=file_paths
                ))

        return tiles

    def compute_normalization(self, tiles: List[TileMetadata],
                              bands: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-band percentile normalization parameters."""
        norm_params = {}

        for band in bands:
            values = []
            for tile in tiles[:100]:  # Sample for efficiency
                if band in tile.file_paths:
                    with rasterio.open(tile.file_paths[band]) as src:
                        data = src.read(1)
                        values.append(data.flatten())

            if values:
                all_values = np.concatenate(values)
                # Remove zeros/nodata
                all_values = all_values[all_values > 0]
                if len(all_values) > 0:
                    p2, p98 = np.percentile(all_values, [2, 98])
                    norm_params[band] = {
                        'p2': float(p2),
                        'p98': float(p98),
                        'min': float(all_values.min()),
                        'max': float(all_values.max())
                    }

        return norm_params

    def apply_normalization(self, tile: TileMetadata,
                            norm_params: Dict[str, Dict[str, float]]) -> TileMetadata:
        """Apply percentile normalization to tile bands."""
        for band, fpath in tile.file_paths.items():
            if band in norm_params:
                p2 = norm_params[band]['p2']
                p98 = norm_params[band]['p98']

                with rasterio.open(fpath, 'r+') as src:
                    data = src.read(1).astype(np.float32)
                    # Clip and normalize
                    data = np.clip(data, p2, p98)
                    data = (data - p2) / (p98 - p2 + 1e-8)
                    src.write(data, 1)

        return tile

    def run_pilot(self, max_scenes_per_aoi: int = 2, max_aois: int = 4):
        """Run pilot download for validation."""
        logger.info("Starting pilot download...")

        pilot_aois = list(self.aois.keys())[:max_aois]
        all_tiles = []
        all_norm_params = {}

        for aoi_id in pilot_aois:
            aoi = self.aois[aoi_id]
            # Use train split dates for pilot
            train_cfg = self.temporal_splits['train']
            scenes = self.search_scenes(aoi_id, train_cfg['start_date'],
                                        train_cfg['end_date'], train_cfg['cloud_filter_max'])

            for scene in scenes[:max_scenes_per_aoi]:
                date_str = scene['datetime'].strftime("%Y%m%d")
                split = self._get_split_for_date(scene['datetime'].strftime("%Y-%m-%d"))

                # Download HR bands
                hr_bands = self.bands_config['hr_target']
                scene_files = self.download_scene_bands(scene, hr_bands, self.raw_dir)

                # Tile
                tiles = self.tile_scene(scene_files, aoi_id, date_str, split)
                all_tiles.extend(tiles)

        # Compute normalization on pilot tiles
        hr_bands = self.bands_config['hr_target']
        norm_params = self.compute_normalization(all_tiles, hr_bands)
        all_norm_params.update(norm_params)

        # Apply normalization
        for tile in all_tiles:
            self.apply_normalization(tile, norm_params)

        # Save metadata
        self._save_metadata(all_tiles, "pilot")
        self._save_normalization(norm_params)

        logger.info(f"Pilot complete: {len(all_tiles)} tiles, {len(norm_params)} bands normalized")
        return all_tiles, norm_params

    def _save_metadata(self, tiles: List[TileMetadata], suffix: str):
        """Save tile metadata to CSV and STAC catalog."""
        # CSV index
        rows = []
        for tile in tiles:
            row = {
                'tile_id': tile.tile_id,
                'aoi_id': tile.aoi_id,
                'date': tile.date,
                'cloud_percent': tile.cloud_percent,
                'bounds': json.dumps(tile.bounds),
                's2_tile_id': tile.s2_tile_id,
                'split': tile.split,
            }
            for band, path in tile.file_paths.items():
                row[f'path_{band}'] = path
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.metadata_dir / f"tiles_{suffix}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to {csv_path}")

    def _save_normalization(self, norm_params: Dict):
        """Save normalization parameters."""
        norm_path = Path("configs/normalization.yaml")
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_path, 'w') as f:
            yaml.dump(norm_params, f)
        logger.info(f"Saved normalization to {norm_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    downloader = Sentinel2Downloader()
    tiles, norms = downloader.run_pilot()


if __name__ == "__main__":
    main()