"""
Sentinel-2 Tile Generation Driver
=================================
Generates 1024x1024 tiles with 128 px overlap from the full Sentinel-2
acquisition (per configs/data/sentinel2.yaml: tile_size_10m=1024,
overlap=128, dtype float32), then computes normalization statistics over
the TRAIN split only (leakage policy).

Inputs : data/metadata/sentinel2/full/{AOI}_{split}_manifest.json
         data/raw/sentinel2/{AOI}/{tile}_{date}/{tile}_{date}_{BAND}.tif
Outputs: data/processed/sentinel2/{AOI}/{split}/{tile}_{date}/tile_r{r}_c{c}.tif
         data/metadata/sentinel2/full/{AOI}_{split}_tile_index.csv
         data/metadata/sentinel2/normalization_stats.json  (train only)

Resumable: existing tile files are skipped. Tiles keep NaN nodata; the
NaN fraction is recorded in the index so downstream loaders can filter.
Cloud/SCL masking is NOT applied here (SCL bands were not downloaded;
deferred — documented limitation).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import rasterio
from src.preprocessing.tiling import tile_image
from src.preprocessing.normalization import (
    compute_percentile_stats,
    save_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_tiles")

PROJECT_ROOT = Path(__file__).parent.parent
CFG_PATH = PROJECT_ROOT / "configs" / "data" / "sentinel2.yaml"
MANIFEST_DIR = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "full"
OUT_TILES = PROJECT_ROOT / "data" / "processed" / "sentinel2"

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

BANDS = CFG["bands"]["hr_target"]          # ["B2", "B3", "B4", "B8"]
TILE_SIZE = CFG["processing"]["tile_size_10m"]
OVERLAP = CFG["processing"]["overlap"]
NORM = CFG["processing"]["normalization"]  # percentile 2-98, clip [0,1]


def load_scene_stack(scene_dir: Path, tile: str, dt: str) -> np.ndarray:
    """Load the 4 band GeoTIFFs of one scene as a [4, H, W] float32 stack.

    The raw GeoTIFFs carry no nodata tag: the EPSG:4326 reprojection fills
    the voids around the rotated granule with exact 0.0 (verified
    22MFD_20190630: 37% of pixels exactly 0 in B2, tile r0c0 AND r3c3 fully
    void). A pixel that is exactly 0.0 in ALL four bands simultaneously is
    fill, not surface (real reflectance is >0 in NIR); such pixels are
    converted to NaN so the tile nan_fraction is truthful and downstream
    loaders can filter void tiles.
    """
    planes = []
    for b in BANDS:
        f = scene_dir / f"{tile}_{dt}_{b}.tif"
        with rasterio.open(f) as src:
            planes.append(src.read(1).astype(np.float32))
    stack = np.stack(planes, axis=0)
    with np.errstate(invalid="ignore"):
        void = (stack == 0).all(axis=0) & np.isfinite(stack).all(axis=0)
    stack[:, void] = np.nan
    return stack


def process_scene(aoi_id: str, split: str, entry: dict) -> list:
    """Tile one scene; returns index rows. Skips existing tiles."""
    scene_id = entry["scene_id"]
    mgrs = entry["mgrs_tile"]
    dt = entry["datetime"][:10].replace("-", "")
    scene_dir = PROJECT_ROOT / "data" / "raw" / "sentinel2" / aoi_id / f"{mgrs}_{dt}"
    out_dir = OUT_TILES / aoi_id / split / f"{mgrs}_{dt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = load_scene_stack(scene_dir, mgrs, dt)
    tiles, positions = tile_image(stack, TILE_SIZE, OVERLAP)

    rows = []
    for t, (r, c) in zip(tiles, positions):
        fname = out_dir / f"tile_r{r:05d}_c{c:05d}.tif"
        if not fname.exists():
            with rasterio.open(
                fname, "w", driver="GTiff", height=t.shape[1], width=t.shape[2],
                count=t.shape[0], dtype="float32", compress="DEFLATE",
            ) as dst:
                dst.write(t.astype(np.float32))
        with np.errstate(invalid="ignore"):
            nan_frac = float(np.isnan(t).mean())
        rows.append({
            "tile_id": f"{mgrs}_{dt}_r{r}_c{c}",
            "scene_id": scene_id,
            "aoi_id": aoi_id,
            "split": split,
            "mgrs_tile": mgrs,
            "date": dt,
            "cloud_cover": entry.get("cloud_cover"),
            "row": r, "col": c,
            "path": str(fname.relative_to(PROJECT_ROOT)),
            "nan_fraction": round(nan_frac, 6),
        })
    logger.info(f"[{aoi_id}/{split}] {mgrs}_{dt}: {len(rows)} tiles "
                f"({sum(r['nan_fraction'] for r in rows)/len(rows):.1%} mean NaN)")
    return rows


def compute_train_stats(train_index_rows: list):
    """Percentile + mean/std stats over TRAIN tiles only (streaming)."""
    n_total = len(train_index_rows)

    def tile_gen():
        for i, row in enumerate(train_index_rows):
            if i % 250 == 0:
                # Progress heartbeat with flush: the stats pass is otherwise
                # silent, and stdout is block-buffered when piped, so
                # monitors (shell widget) would see no activity at all.
                print(f"STATS {i}/{n_total}", flush=True)
            with rasterio.open(PROJECT_ROOT / row["path"]) as src:
                yield np.moveaxis(src.read(), 0, -1)  # [H, W, C]

    stats = compute_percentile_stats(
        tile_gen(), band_names=BANDS,
        lower=NORM["lower"], upper=NORM["upper"],
    )
    out = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "normalization_stats.json"
    save_stats(stats, out, extra={
        "dataset": "sentinel2-full",
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "n_train_tiles": len(train_index_rows),
        "method": NORM["method"],
    })
    return out


def main():
    only = sys.argv[1:] or None
    manifests = sorted(MANIFEST_DIR.glob("*_manifest.json"))
    if not manifests:
        logger.error(f"No manifests found in {MANIFEST_DIR}")
        sys.exit(1)

    train_rows: list = []
    total_tiles = 0
    for mf in manifests:
        # Manifest stem is "{AOI}_{split}_manifest" and AOI IDs contain
        # underscores (BR_01) — split from the right, never from the left.
        aoi_id, split, _ = mf.stem.rsplit("_", 2)
        if only and aoi_id not in only:
            continue
        manifest = json.loads(mf.read_text())
        scenes = manifest.get("scenes", [])
        logger.info(f"=== {aoi_id}/{split}: {len(scenes)} scenes ===")
        rows = []
        for entry in scenes:
            try:
                rows.extend(process_scene(aoi_id, split, entry))
            except Exception as e:
                logger.error(f"[{aoi_id}/{split}] {entry.get('scene_id')}: {e}")
        # per-AOI/split tile index CSV
        import csv
        idx_path = MANIFEST_DIR / f"{aoi_id}_{split}_tile_index.csv"
        with open(idx_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tile_id"])
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[{aoi_id}/{split}] Index -> {idx_path} ({len(rows)} tiles)")
        total_tiles += len(rows)
        if split == "train":
            train_rows.extend(rows)

    if train_rows:
        logger.info(f"=== Train-split normalization stats over {len(train_rows)} tiles ===")
        stats_path = compute_train_stats(train_rows)
        logger.info(f"Stats -> {stats_path}")

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Tiles generated: {total_tiles}")
    print(f"  Train tiles for stats: {len(train_rows)}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
