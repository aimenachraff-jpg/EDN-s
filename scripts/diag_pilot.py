"""Quick diagnostic: inspect raw band values from one downloaded file."""
import rasterio, numpy as np
from pathlib import Path

p = Path("data/raw/sentinel2/DZ_01/31SDB_20221230/31SDB_20221230_B2.tif")
print(f"File: {p}  exists={p.exists()}  size={p.stat().st_size}")
with rasterio.open(p) as src:
    arr = src.read(1)
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    non_nan = arr[~np.isnan(arr)]
    print(f"  nans={int(np.isnan(arr).sum())}  total={arr.size}  valid={len(non_nan)}")
    if len(non_nan) > 0:
        print(f"  valid min={non_nan.min():.6f}  max={non_nan.max():.6f}  median={np.median(non_nan):.4f}  mean={non_nan.mean():.4f}")
