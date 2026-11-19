"""Validate downloaded Sentinel-2 pilot data against manifest."""
import json, sys
from pathlib import Path
import rasterio
import numpy as np

MANIFEST = Path("data/metadata/sentinel2/pilot_manifest.json")
assert MANIFEST.exists(), f"Manifest missing: {MANIFEST}"

with open(MANIFEST) as f:
    manifest = json.load(f)

errors = []
for scene in manifest["scenes"]:
    print(f"Scene: {scene['scene_id']}  date={scene['datetime'][:10]}  cloud={scene['cloud_cover']}%  tile={scene['mgrs_tile']}")
    for band, info in scene["files"].items():
        p = Path(info["path"])
        if not p.exists():
            errors.append(f"MISSING: {p}")
            continue
        with rasterio.open(p) as src:
            arr = src.read(1)
            non_nan = arr[~np.isnan(arr)]
            n_total = arr.size
            n_nan = int(np.isnan(arr).sum())
            has_data = len(non_nan) > 0 and non_nan.min() >= 0 and non_nan.max() <= 1.0
            status = "OK" if has_data else "FAIL"
            print(f"  {band}: {status}  shape={arr.shape}  valid_px={len(non_nan)}/{n_total}  range=[{non_nan.min():.4f},{non_nan.max():.4f}]  CRS={src.crs}")
            if not has_data or n_nan == n_total:
                errors.append(f"{p} has no valid data")
    if not scene["checksums_valid"]:
        errors.append(f"Checksum mismatch for {scene['scene_id']}")

total_mb = sum(v["size_bytes"] for s in manifest["scenes"] for v in s["files"].values()) / 1e6
print(f"\nTotal: {manifest['summary']['scenes_downloaded']} scenes, {manifest['summary']['total_files']} files, {total_mb:.1f} MB")
if errors:
    print(f"VALIDATION FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL VALIDATIONS PASSED")
