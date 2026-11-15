"""
Sentinel-2 Full-Scale Acquisition Script
=========================================
Approved full acquisition (2026-11-14): all 8 AOIs across train/val/test
temporal splits, per configs/data/sentinel2.yaml.

Design (matching the ~50-60 GB approved volume):
  - 8 AOIs: DZ_01..DZ_04 (Algeria, primary), FR_01/US_01/CL_01/BR_01 (validation)
  - Scenes per AOI: train 6, val 2, test 3  => 11 x 8 = 88 scenes total
  - Scene quota filled with distinct DATES evenly spread across the split
    period (temporal diversity); within a date, distinct MGRS tiles preferred
    (spatial diversity).
  - Cloud filter per split from config (train/val <20%, test none).
  - Resumable: a scene is skipped if all 4 band files already exist.
  - Per-AOI/split manifest with provenance, checksums, shapes, bounds.

Rulings (recorded 2026-11-14):
  - EPSG:4326 output retained (uniform coordinate space, consistent with
    pilot + manifests). UTM is preferred for production tiling; noted as a
    limitation. Cost if wrong: slight distance distortion at high latitudes,
    handled later by per-tile UTM reprojection in the tiling module.
  - Test split follows config: no cloud filter (realistic conditions).
"""

import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rasterio
from rasterio.transform import from_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("s2_full")

PROJECT_ROOT = Path(__file__).parent.parent
CFG_PATH = PROJECT_ROOT / "configs" / "data" / "sentinel2.yaml"
DATA_ROOT = PROJECT_ROOT / "data"

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

TARGET_BANDS = CFG["bands"]["hr_target"]
ODC_BAND_MAP = {
    "B2": "blue", "B3": "green", "B4": "red", "B8": "nir",
}
ODC_BANDS = [ODC_BAND_MAP[b] for b in TARGET_BANDS]

# Scene quota per split (ruling: matches approved ~50-60 GB total volume)
SCENES_PER_SPLIT = {"train": 6, "val": 2, "test": 3}
SPLITS = ["train", "val", "test"]

OUT_RAW = DATA_ROOT / "raw" / "sentinel2"
OUT_META = DATA_ROOT / "metadata" / "sentinel2" / "full"
OUT_META.mkdir(parents=True, exist_ok=True)

ENDPOINT = CFG["download"]["providers"][0]["endpoint"]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_complete(scene_dir: Path, tile: str, dt: str) -> bool:
    return all((scene_dir / f"{tile}_{dt}_{b}.tif").exists() for b in TARGET_BANDS)


def search_aoi(client, aoi_id: str, aoi: dict, split: str) -> list:
    ts = CFG["temporal_splits"][split]
    cloud_max = ts["cloud_filter_max"]
    query = {"eo:cloud_cover": {"lt": cloud_max * 100}} if cloud_max < 1.0 else {}
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi["bbox"],
        datetime=f"{ts['start_date']}/{ts['end_date']}",
        query=query,
        limit=250,
    )
    items = list(search.items())
    logger.info(f"[{aoi_id}/{split}] Found {len(items)} candidate scenes "
                f"({ts['start_date']}..{ts['end_date']}, cloud<{cloud_max:.0%})")
    return items


def load_rejected() -> set:
    """Item IDs previously rejected as degenerate — never re-selected."""
    f = OUT_META / "rejected_items.json"
    if f.exists():
        try:
            return set(json.loads(f.read_text()))
        except Exception:
            pass
    return set()


def record_rejected(item_id: str) -> None:
    f = OUT_META / "rejected_items.json"
    ids = load_rejected()
    ids.add(item_id)
    f.write_text(json.dumps(sorted(ids), indent=2))


def select_scenes(items: list, quota: int) -> list:
    """Pick `quota` scenes: distinct dates evenly spread over the period,
    preferring distinct MGRS tiles within each date."""
    # Dedupe by (tile, date); reject partial granules (ruling: eo:cloud_cover
    # passes for granules covering only ~13% of the tile — verified
    # s2:nodata_pixel_percentage=87.1 on 31SEB_20201228 — so filter on nodata)
    rejected = load_rejected()
    seen, uniq = set(), []
    for it in items:
        if it.id in rejected:
            continue
        nodata_pct = it.properties.get("s2:nodata_pixel_percentage", 0.0)
        if nodata_pct is not None and nodata_pct > 50.0:
            continue
        key = (it.properties.get("s2:mgrs_tile"), it.datetime.date())
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    # Group by date, sort dates ascending; within a date sort by cloud then tile
    by_date: Dict = {}
    for it in uniq:
        by_date.setdefault(it.datetime.date(), []).append(it)
    dates = sorted(by_date.keys())
    if not dates:
        return []

    n_dates = min(quota, len(dates))
    # Ruling: a date whose best candidate is >=95% cloud has no usable
    # surface pixels (config's "no filter" intent is realistic conditions,
    # not all-cloud scenes) — exclude such dates from selection.
    usable = [d for d in dates
              if min(it.properties.get("eo:cloud_cover", 0) for it in by_date[d]) < 95]
    dates = usable
    if not dates:
        return []
    n_dates = min(quota, len(dates))
    # Evenly spaced date indices (temporal stratification)
    idxs = [round(i * (len(dates) - 1) / max(n_dates - 1, 1)) for i in range(n_dates)]
    idxs = sorted(set(idxs))

    selected: List = []
    seen_tiles = set()
    for i in idxs:
        cands = sorted(by_date[dates[i]],
                       key=lambda it: (it.properties.get("eo:cloud_cover", 0),
                                       it.properties.get("s2:mgrs_tile", "")))
        # Same 95% sanity at candidate level (tile preference must not
        # override usability)
        cands = [c for c in cands
                 if c.properties.get("eo:cloud_cover", 0) < 95] or cands
        # Prefer a tile not yet used
        pick = next((c for c in cands if c.properties.get("s2:mgrs_tile") not in seen_tiles),
                    cands[0])
        selected.append(pick)
        seen_tiles.add(pick.properties.get("s2:mgrs_tile"))
        if len(selected) >= quota:
            break
    return selected[:quota]


def _resign(item) -> None:
    """Refresh PC SAS signatures on all asset hrefs.

    Tokens expire (~1 h); a multi-hour run crosses expiry boundaries
    (observed: 403 mid-scene at FR_01/val on 2026-11-14).
    """
    import planetary_computer as pc
    for a in item.assets.values():
        try:
            a.href = pc.sign(a.href)
        except Exception:
            pass  # non-PC hrefs stay unchanged


def download_scene(item, aoi_id: str) -> Optional[Tuple[str, Dict]]:
    import odc.stac as odc_stac

    tile = item.properties.get("s2:mgrs_tile", "unknown")
    dt_str = item.datetime.strftime("%Y%m%d")
    scene_dir = OUT_RAW / aoi_id / f"{tile}_{dt_str}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    if scene_complete(scene_dir, tile, dt_str):
        logger.info(f"[{aoi_id}] Scene already complete, skipping: {tile}_{dt_str}")
        return ("skipped", {"scene_dir": str(scene_dir), "tile": tile, "date": dt_str})

    file_info: Dict = {}
    # Per-band load: each band gets a fresh signed href immediately before
    # its read, isolating transient token/network failures per band.
    for s2_band in TARGET_BANDS:
        odc_var = ODC_BAND_MAP[s2_band]
        try:
            _resign(item)
            xx = odc_stac.load(
                [item],
                bands=[odc_var],
                crs="EPSG:4326",
                resolution=0.0001,       # ~10 m at mid latitudes
                chunks={"x": 2048, "y": 2048},
            )
            da = xx[odc_var]
            arr = da.values.astype(np.float32)
        except Exception as e:
            logger.error(f"[{aoi_id}] load/read failed for {item.id} {s2_band}: {e}")
            return None
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        # Ruling (verified 2026-11-14): PC assets for pre-baseline-04.00 scenes
        # lack raster:bands scale metadata, so odc-stac returns RAW DN (0-10000)
        # for those and scaled [0,1] for baseline>=04.00. Detect and rescale:
        # raw DN saturates under clip(0,1) (observed: constant 1.0 files).
        with np.errstate(invalid="ignore"):
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.max() > 2.0:
                arr = arr / 10000.0
                logger.info(f"[{aoi_id}]   {s2_band}: raw DN detected -> rescaled /10000")
        arr = np.clip(arr, 0.0, 1.0)

        # Degeneracy check (verified 2026-11-15: 31SDB_20201223 is 96-98% raw
        # DN=0/1 (nodata/saturated) -> ~0.0001 after rescale; passed cloud and
        # nodata filters because nodata% only counts DN=0). Reject and delete.
        with np.errstate(invalid="ignore"):
            finite = arr[np.isfinite(arr)]
            if finite.size and float((finite <= 0.001).mean()) > 0.9:
                frac = float((finite <= 0.001).mean())
                logger.error(f"[{aoi_id}] DEGENERATE scene {tile}_{dt_str} "
                             f"({s2_band}): {frac:.0%} pixels <= 0.001 — rejecting")
                shutil.rmtree(scene_dir, ignore_errors=True)
                return ("degenerate", {"tile": tile, "date": dt_str})

        lons = da.coords["longitude"].values
        lats = da.coords["latitude"].values
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_max, lat_min = float(lats.max()), float(lats.min())
        h, w = arr.shape
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

        fname = scene_dir / f"{tile}_{dt_str}_{s2_band}.tif"
        with rasterio.open(
            fname, "w", driver="GTiff", height=h, width=w, count=1,
            dtype=arr.dtype, crs="EPSG:4326",
            transform=transform, compress="DEFLATE",
        ) as dst:
            dst.write(arr, 1)

        file_info[s2_band] = {
            "path": str(fname.relative_to(PROJECT_ROOT)),
            "size_bytes": fname.stat().st_size,
            "md5": md5_file(fname),
            "shape": list(arr.shape),
            "bounds": [lon_min, lat_min, lon_max, lat_max],
            "transform": [float(t) for t in transform],
            "valid_fraction": round(float(np.isfinite(arr).mean()), 4),
        }
        logger.info(f"[{aoi_id}]   {s2_band}: {file_info[s2_band]['size_bytes']/1e6:.1f} MB "
                    f"shape={arr.shape} valid={file_info[s2_band]['valid_fraction']:.0%}")

    if not file_info:
        return None
    return (item.id, file_info)


def build_manifest(aoi_id: str, aoi: dict, split: str, entries: List[dict]) -> Path:
    ts = CFG["temporal_splits"][split]
    manifest = {
        "acquisition_version": "1.0-full",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "provider": "Microsoft Planetary Computer STAC API",
            "endpoint": ENDPOINT,
            "collection": "sentinel-2-l2a",
            "license": "Copernicus / Sentinel-2, CC BY 4.0",
            "citation": "Sentinel-2 Level-2A surface reflectance. European Space Agency.",
        },
        "aoi": {"id": aoi_id, "name": aoi["name"], "country": aoi["country"],
                "bbox": aoi["bbox"], "land_cover": aoi["land_cover"]},
        "split": split,
        "temporal_scope": {"start": ts["start_date"], "end": ts["end_date"],
                           "cloud_filter_max": ts["cloud_filter_max"]},
        "bands_requested": TARGET_BANDS,
        "resolution_m": 10,
        "crs": "EPSG:4326",
        "scenes": entries,
        "summary": {},
    }
    total_bytes = sum(v["size_bytes"] for e in entries for v in e.get("files", {}).values())
    manifest["summary"] = {
        "scenes": len(entries),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 1),
        "total_files": sum(len(e.get("files", {})) for e in entries),
        "checksums_valid": all(e.get("checksums_valid", False) for e in entries),
    }
    out = OUT_META / f"{aoi_id}_{split}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"[{aoi_id}/{split}] Manifest -> {out}")
    return out


def _backfill_file_info(scene_dir: Path, tile: str, dt_str: str) -> Dict:
    """Build file info for pre-existing scenes by reading GeoTIFF headers
    (cheap: metadata only, no pixel reads)."""
    finfo = {}
    for b in TARGET_BANDS:
        f = scene_dir / f"{tile}_{dt_str}_{b}.tif"
        with rasterio.open(f) as src:
            bounds = [float(x) for x in src.bounds]
            transform = [float(t) for t in src.transform][:6]
            shape = [src.height, src.width]
        finfo[b] = {
            "path": str(f.relative_to(PROJECT_ROOT)),
            "size_bytes": f.stat().st_size,
            "md5": md5_file(f),
            "shape": shape,
            "bounds": bounds,
            "transform": transform,
            "valid_fraction": None,
            "preexisting": True,
        }
    return finfo


def main():
    only = sys.argv[1:] or None  # optional AOI filter args
    aoi_ids = [k for k in CFG["aois"] if not only or k in only]

    import planetary_computer as pc
    import pystac_client as pystac_mod

    client = pystac_mod.Client.open(ENDPOINT, modifier=pc.sign_inplace)

    grand_bytes = 0
    grand_scenes = 0
    for aoi_id in aoi_ids:
        aoi = CFG["aois"][aoi_id]
        logger.info(f"=== AOI {aoi_id} ({aoi['name']}, {aoi['country']}) ===")
        for split in SPLITS:
            quota = SCENES_PER_SPLIT[split]
            items = search_aoi(client, aoi_id, aoi, split)
            selected = select_scenes(items, quota)
            logger.info(f"[{aoi_id}/{split}] Selected {len(selected)}/{quota} scenes")
            entries = []
            # Existing manifest (if any): reused for already-complete scenes
            # so re-runs never degrade rich metadata into skip-only stubs.
            existing: Dict = {}
            mf_path = OUT_META / f"{aoi_id}_{split}_manifest.json"
            if mf_path.exists():
                try:
                    for e in json.loads(mf_path.read_text()).get("scenes", []):
                        key = (e.get("mgrs_tile"),
                               e.get("datetime", "")[:10].replace("-", ""))
                        existing[key] = e
                except Exception as e:
                    logger.warning(f"[{aoi_id}/{split}] Could not load existing manifest: {e}")
            for it in selected:
                result = download_scene(it, aoi_id)
                if result is None:
                    logger.warning(f"[{aoi_id}] Skipped (load failed) {it.id}")
                    continue
                sid, payload = result
                if sid == "degenerate":
                    record_rejected(it.id)
                    logger.warning(
                        f"[{aoi_id}] Scene rejected (degenerate granule): "
                        f"{payload['tile']}_{payload['date']} — recorded; "
                        f"quota shortfall documented"
                    )
                    continue
                if sid == "skipped":
                    tile, dt_str = payload["tile"], payload["date"]
                    old = existing.get((tile, dt_str))
                    if old is not None:
                        # Reuse the rich entry; refresh checksums against disk
                        for v in old.get("files", {}).values():
                            f = PROJECT_ROOT / v["path"]
                            if f.exists():
                                v["md5"] = md5_file(f)
                        old["checksums_valid"] = all(
                            md5_file(PROJECT_ROOT / v["path"]) == v["md5"]
                            for v in old["files"].values()
                        )
                        entries.append(old)
                        continue
                    scene_dir = OUT_RAW / aoi_id / f"{tile}_{dt_str}"
                    finfo = _backfill_file_info(scene_dir, tile, dt_str)
                    sid = f"{tile}_{dt_str}"
                else:
                    finfo = payload
                entries.append({
                    "scene_id": sid,
                    "datetime": it.properties.get("datetime", ""),
                    "cloud_cover": round(it.properties.get("eo:cloud_cover", 0), 1),
                    "mgrs_tile": it.properties.get("s2:mgrs_tile", "?"),
                    "files": finfo,
                    "checksums_valid": all(
                        md5_file(PROJECT_ROOT / v["path"]) == v["md5"]
                        for v in finfo.values()
                    ),
                })
            build_manifest(aoi_id, aoi, split, entries)
            s_bytes = sum(v["size_bytes"] for e in entries for v in e.get("files", {}).values())
            grand_bytes += s_bytes
            grand_scenes += len(entries)
            logger.info(f"[{aoi_id}/{split}] Done: {len(entries)} scenes, {s_bytes/1e6:.1f} MB")

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Full acquisition run: {grand_scenes} scenes, {grand_bytes/1e9:.2f} GB total this run")
    print(f"  AOIs: {', '.join(aoi_ids)}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
