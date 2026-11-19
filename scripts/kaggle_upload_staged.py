"""
Phase 5: staged Kaggle dataset upload for s2-tiles-trainval
===========================================================
Kaggle's web UI rejects >1000 loose files (we have ~14,750 tiles), and a
single 72 GB zip cannot fit on disk (~56 GB free). Solution: one zip per
AOI x split (train/val only), each counting as ONE file; Kaggle unzips
each automatically after upload.

Modes:
  python scripts/kaggle_upload_staged.py --plan      # dry run: zips + sizes
  python scripts/kaggle_upload_staged.py --zip-only 5  # make zip #5 only (manual browser upload)
  python scripts/kaggle_upload_staged.py             # FULL AUTO: needs CLI + token

Full auto (requires kaggle CLI + %USERPROFILE%\\.kaggle\\kaggle.json):
  1. create dataset (version 1 = 16 tile-index CSVs + normalization_stats.json)
  2. for each (aoi, split): zip tiles -> `kaggle datasets version` upload
     -> delete zip only after successful upload -> next

Zips are ZIP_STORED (no compression: float32 GeoTIFFs barely compress,
storing keeps zip time ~disk speed).
"""

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
S2 = PROJECT_ROOT / "data" / "processed" / "sentinel2"
META_FULL = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "full"
STATS = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "normalization_stats.json"
STAGE = PROJECT_ROOT / "data" / "kaggle_stage"
DATASET_TITLE = "s2-tiles-trainval"


def combos():
    """(aoi, split) pairs that exist on disk, train+val only."""
    out = []
    for aoi in sorted(p.name for p in S2.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            if (S2 / aoi / split).is_dir():
                out.append((aoi, split))
    return out


def zip_tiles(aoi: str, split: str, out_zip: Path) -> int:
    files = sorted((S2 / aoi / split).rglob("*.tif"))
    t0 = time.time()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as z:
        for i, f in enumerate(files):
            arc = Path(aoi) / split / f.relative_to(S2 / aoi / split)
            z.write(f, arcname=str(arc))
            if (i + 1) % 500 == 0:
                print(f"ZIP {aoi}_{split} {i + 1}/{len(files)} "
                      f"({(time.time() - t0):.0f}s)", flush=True)
    print(f"ZIPDONE {aoi}_{split} {len(files)} tiles, "
          f"{out_zip.stat().st_size / 1e9:.2f} GB, "
          f"{time.time() - t0:.0f}s -> {out_zip.name}", flush=True)
    return len(files)


def kaggle_username() -> str:
    tok = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(tok.read_text())["username"]


def cli_ok() -> bool:
    try:
        r = subprocess.run(["kaggle", "--version"], capture_output=True,
                           text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def push_version(msg: str) -> None:
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(STAGE),
         "-m", msg, "-r", "skip", "--dir-mode", "skip"],
        capture_output=True, text=True, timeout=3600 * 6)
    tail = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(ln for ln in tail.splitlines()
                     if "Warning" not in ln and "warnings.warn" not in ln)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle version failed: {tail[-800:]}")
    print(f"UPLOADED version '{msg}' ({tail.strip().splitlines()[-1][:120]})",
          flush=True)


def create_dataset_first_version(username: str) -> None:
    """Version 1: only the metadata files (17 loose files, under the limit).

    NOTE: files sit FLAT in the stage root — `dir-mode skip` skips
    subfolders entirely, so a metadata/ subfolder would never upload.
    """
    STAGE.mkdir(parents=True, exist_ok=True)
    for csv in list(META_FULL.glob("*_train_tile_index.csv")) + \
            list(META_FULL.glob("*_val_tile_index.csv")):
        (STAGE / csv.name).write_bytes(csv.read_bytes())
    (STAGE / "normalization_stats.json").write_bytes(STATS.read_bytes())
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_TITLE,
        "id": f"{username}/{DATASET_TITLE}",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }, indent=2))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(STAGE)],
                       capture_output=True, text=True, timeout=1800)
    tail = "\n".join(ln for ln in ((r.stdout or "") + (r.stderr or "")).splitlines()
                     if "Warning" not in ln)
    if r.returncode != 0 and "already exists" not in tail:
        raise RuntimeError(f"dataset create failed: {tail[-800:]}")
    print(f"DATASET created/exists: {username}/{DATASET_TITLE} "
          f"(version 1 = metadata only)", flush=True)


def main():
    if "--plan" in sys.argv:
        total = 0
        print(f"zips planned (one per AOI x split):")
        for i, (aoi, split) in enumerate(combos(), 1):
            n = len(list((S2 / aoi / split).rglob("*.tif")))
            total += n
            print(f"  {i:2d}. {aoi}_{split}: {n} tiles")
        print(f"  total: {total} tiles -> {len(combos())} zips")
        return

    zs = [f"{aoi}_{split}" for aoi, split in combos()]
    if "--zip-only" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--zip-only") + 1]) - 1
        aoi, split = combos()[idx]
        STAGE.mkdir(parents=True, exist_ok=True)
        zip_tiles(aoi, split, STAGE / f"{aoi}_{split}.zip")
        print(f"MANUAL: on Kaggle -> New Version -> upload "
              f"{STAGE / f'{aoi}_{split}.zip'} -> after it finishes, delete "
              f"the zip locally to free disk", flush=True)
        return

    if not cli_ok():
        print("kaggle CLI not usable - missing token? Save kaggle.json to "
              f"{Path.home() / '.kaggle'}", flush=True)
        sys.exit(2)
    user = kaggle_username()
    create_dataset_first_version(user)
    for i, (aoi, split) in enumerate(combos(), 1):
        zp = STAGE / f"{aoi}_{split}.zip"
        zip_tiles(aoi, split, zp)
        push_version(f"{aoi}_{split} tiles ({i}/{len(zs)})")
        zp.unlink()
        print(f"DELETED {zp.name} (disk freed)", flush=True)
    print("ALL DONE: 16 tile zips uploaded as versions", flush=True)


if __name__ == "__main__":
    main()
