"""
Sentinel-2 Pilot Data Acquisition Script
=========================================
Downloads a small pilot sample from AOI DZ_01 (Algiers Urban Coastal).
Target bands: B2 (blue), B3 (green), B4 (red), B8 (nir) — 10 m resolution.
Period: 2018–2022 (train split). Max 3 scenes.
Source: Microsoft Planetary Computer STAC API (free, no auth required).
License: Copernicus / Sentinel-2, CC BY 4.0.
"""

import os, sys, json, hashlib, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
from shapely.geometry import shape
import rasterio
from rasterio.transform import from_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("s2_pilot")

PROJECT_ROOT = Path(__file__).parent.parent
CFG_PATH     = PROJECT_ROOT / "configs" / "data" / "sentinel2.yaml"
DATA_ROOT    = PROJECT_ROOT / "data"

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

AOI_ID       = "DZ_01"
AOI          = CFG["aois"][AOI_ID]
BOUNDS       = AOI["bbox"]              # [min_lon, min_lat, max_lon, max_lat]
TARGET_BANDS = CFG["bands"]["hr_target"]   # ["B2","B3","B4","B8"]
TRAIN_START  = "2018-01-01"
TRAIN_END    = "2022-12-31"
MAX_SCENES   = 3
CLOUD_MAX    = 0.30

OUT_RAW      = DATA_ROOT / "raw" / "sentinel2" / AOI_ID
OUT_META     = DATA_ROOT / "metadata" / "sentinel2"
OUT_META.mkdir(parents=True, exist_ok=True)
OUT_RAW.mkdir(parents=True, exist_ok=True)

# ODC band names → S2 L2A short names
ODC_BAND_MAP = {
    "B2": "blue", "B3": "green", "B4": "red", "B8": "nir",
    "B5": "rededge1", "B6": "rededge2", "B7": "rededge3",
    "B8a": "nir08",  "B11": "swir16", "B12": "swir22",
}
ODC_BANDS = [ODC_BAND_MAP[b] for b in TARGET_BANDS]


# ─── Search ──────────────────────────────────────────────────────────────────
def search_scenes():
    import planetary_computer as pc
    import pystac_client as pystac_mod

    logger.info(
        f"STAC search  AOI={AOI_ID}  [{TRAIN_START} – {TRAIN_END}]  cloud<{CLOUD_MAX:.0%}"
    )
    client = pystac_mod.Client.open(
        CFG["download"]["providers"][0]["endpoint"],
        modifier=pc.sign_inplace,
    )
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=BOUNDS,
        datetime=f"{TRAIN_START}/{TRAIN_END}",
        query={"eo:cloud_cover": {"lt": CLOUD_MAX * 100}},
        limit=MAX_SCENES + 5,
    )
    items = list(search.items())
    logger.info(f"  Found {len(items)} candidate scenes")
    return items


# ─── Download single scene ────────────────────────────────────────────────────
def download_scene(item, dest_dir: Path):
    """Download one scene's bands as GeoTIFFs. Returns (scene_id, file_info) or None."""
    import odc.stac as odc_stac

    tile  = item.properties.get("s2:mgrs_tile", "unknown")
    dt_str = item.datetime.strftime("%Y%m%d")
    scene_id = item.id
    scene_dir = dest_dir / f"{tile}_{dt_str}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    try:
        xx = odc_stac.load(
            [item],
            bands=ODC_BANDS,
            crs="EPSG:4326",
            resolution=0.0001,       # ≈ 10 m at ~36°N
            chunks={"x": 2048, "y": 2048},
        )
    except Exception as e:
        logger.error(f"  odc.stac load failed for {scene_id}: {e}")
        return None

    file_info: Dict = {}
    for s2_band, odc_var in zip(TARGET_BANDS, ODC_BANDS):
        if odc_var not in xx.data_vars:
            logger.warning(f"  {s2_band} ({odc_var}) missing — skipping")
            continue
        da = xx[odc_var]
        arr = da.values.astype(np.float32)
        # Squeeze singleton time dim (each scene is one date)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        # Sentinel-2 L2A from Planetary Computer: values already in [0, 1] (surface reflectance)
        arr = np.clip(arr, 0.0, 1.0)

        # Derive geotransform from coordinate arrays (ODC uses lat/lon names)
        lons = da.coords["longitude"].values
        lats = da.coords["latitude"].values
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_max, lat_min = float(lats.max()), float(lats.min())
        h, w = arr.shape
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

        fname = scene_dir / f"{tile}_{dt_str}_{s2_band}.tif"
        with rasterio.open(
            fname, "w",
            driver="GTiff", height=h, width=w, count=1,
            dtype=arr.dtype, crs="EPSG:4326",
            transform=transform, compress="DEFLATE",
        ) as dst:
            dst.write(arr, 1)

        sz   = fname.stat().st_size
        md5  = hashlib.md5(fname.read_bytes()).hexdigest()
        file_info[s2_band] = {
            "path":       str(fname.relative_to(PROJECT_ROOT)),
            "size_bytes": sz,
            "md5":        md5,
            "shape":      list(arr.shape),
            "bounds":     [lon_min, lat_min, lon_max, lat_max],
            "transform":  [float(t) for t in transform],
        }
        logger.info(f"  {s2_band}: {sz/1e6:.1f} MB  md5={md5[:8]}…  shape={list(arr.shape)}")

    if not file_info:
        return None
    return scene_id, file_info


# ─── Manifest ────────────────────────────────────────────────────────────────
def build_manifest(scenes, files_list):
    manifest = {
        "pilot_version":   "1.0",
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "source": {
            "provider":   "Microsoft Planetary Computer STAC API",
            "endpoint":   CFG["download"]["providers"][0]["endpoint"],
            "collection": "sentinel-2-l2a",
            "license":    "Copernicus / Sentinel-2, CC BY 4.0",
            "citation":   "Sentinel-2 Level-2A surface reflectance. European Space Agency.",
        },
        "aoi": {
            "id":         AOI_ID,
            "name":       AOI["name"],
            "country":    AOI["country"],
            "bbox":       BOUNDS,
            "land_cover": AOI["land_cover"],
        },
        "temporal_scope": {
            "start": TRAIN_START, "end": TRAIN_END, "split": "train",
        },
        "bands_requested":    TARGET_BANDS,
        "resolution_m":       10,
        "max_scenes_limit":   MAX_SCENES,
        "scenes":             [],
        "summary":            {},
    }

    total_bytes = 0
    for it, finfo in zip(scenes, files_list):
        props  = it.properties
        entry = {
            "scene_id":      it.id,
            "datetime":      props.get("datetime", ""),
            "cloud_cover":   round(props.get("eo:cloud_cover", 0), 1),
            "mgrs_tile":     props.get("s2:mgrs_tile", "?"),
            "files":         finfo,
            "checksums_valid": all(
                hashlib.md5(Path(v["path"]).resolve().read_bytes()).hexdigest() == v["md5"]
                for v in finfo.values()
            ),
        }
        manifest["scenes"].append(entry)
        total_bytes += sum(v["size_bytes"] for v in finfo.values())

    manifest["summary"] = {
        "scenes_downloaded": len(manifest["scenes"]),
        "total_bytes":       total_bytes,
        "total_mb":          round(total_bytes / 1e6, 1),
        "bands_per_scene":   len(TARGET_BANDS),
        "total_files":       sum(len(s["files"]) for s in manifest["scenes"]),
    }
    return manifest


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    logger.info(f"=== Sentinel-2 Pilot : AOI {AOI_ID} ===")
    logger.info(
        f"Bands: {', '.join(TARGET_BANDS)}  |  "
        f"Period: {TRAIN_START}–{TRAIN_END}  |  Max: {MAX_SCENES} scenes"
    )

    items = search_scenes()
    if not items:
        logger.error("No scenes found — check network connectivity or AOI bounds.")
        sys.exit(1)

    # De-duplicate by (MGRS tile, date)
    seen, unique = set(), []
    for it in items:
        key = (it.properties.get("s2:mgrs_tile"), it.datetime.date())
        if key not in seen:
            seen.add(key)
            unique.append(it)
        if len(unique) >= MAX_SCENES:
            break

    logger.info(f"After dedup: {len(unique)} scenes")

    scenes_out, files_list = [], []
    for it in unique:
        result = download_scene(it, OUT_RAW)
        if result is None:
            logger.warning(f"  Skipped {it.id}")
            continue
        sid, finfo = result
        scenes_out.append(it)
        files_list.append(finfo)

    manifest = build_manifest(scenes_out, files_list)
    manifest_path = OUT_META / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Manifest → {manifest_path}")

    s = manifest["summary"]
    sep = "=" * 48
    print(f"\n{sep}")
    print(f"  Scenes : {s['scenes_downloaded']}")
    print(f"  Total  : {s['total_mb']:.1f} MB  ({s['total_files']} files)")
    print(f"  Bands  : {', '.join(TARGET_BANDS)}")
    print(f"  Source : Planetary Computer STAC (free, no auth)")
    print(f"  License: Copernicus / Sentinel-2, CC BY 4.0")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
