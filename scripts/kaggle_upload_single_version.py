"""
Phase 5: single-version Kaggle upload (accumulates ALL tiles into one dataset version).

Unlike kaggle_upload_staged.py (which created one version per zip, leaving only
the last batch in the "current" version), this script uploads every zip (and the
metadata files) as blobs first, then creates ONE version referencing all of them.
Kaggle auto-unzips each zip, so the final version contains all 10,680 tiles.

Disk-safe: only one ~8 GB zip exists at a time (deleted right after upload).
Resumable: progress is saved to data/kaggle_stage/upload_tokens.json; re-running
skips files already uploaded.

Usage:
  python scripts/kaggle_upload_single_version.py            # full run
  python scripts/kaggle_upload_single_version.py --status   # show progress
"""
import json
import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
S2 = PROJECT_ROOT / "data" / "processed" / "sentinel2"
META_FULL = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "full"
STATS = PROJECT_ROOT / "data" / "metadata" / "sentinel2" / "normalization_stats.json"
STAGE = PROJECT_ROOT / "data" / "kaggle_stage"
TOKENS_FILE = STAGE / "upload_tokens.json"
DATASET_TITLE = "s2-tiles-trainval"
OWNER = "aymencharafedn"
SLUG = "s2-tiles-trainval"

# ---- auth (Bearer via env var, like the staged script) ----
tok_path = Path.home() / ".kaggle" / "kaggle.json"
if not os.environ.get("KAGGLE_API_TOKEN") and tok_path.exists():
    os.environ["KAGGLE_API_TOKEN"] = json.loads(tok_path.read_text())["key"]

sys.path.insert(0, str(PROJECT_ROOT))
from kaggle import KaggleApi
from kaggle.api.kaggle_api_extended import ApiBlobType, ResumableUploadContext

api = KaggleApi()
api.authenticate()


def combos():
    out = []
    for aoi in sorted(p.name for p in S2.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            if (S2 / aoi / split).is_dir():
                out.append((aoi, split))
    return out


def load_tokens():
    if TOKENS_FILE.exists():
        return json.loads(TOKENS_FILE.read_text())
    return {}


def save_tokens(tok):
    STAGE.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps(tok, indent=2))


def zip_tiles(aoi, split, out_zip):
    files = sorted((S2 / aoi / split).rglob("*.tif"))
    t0 = time.time()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as z:
        for i, f in enumerate(files):
            arc = Path(aoi) / split / f.relative_to(S2 / aoi / split)
            z.write(f, arcname=str(arc))
            if (i + 1) % 500 == 0:
                print(f"  ZIP {aoi}_{split} {i+1}/{len(files)} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"  ZIPDONE {aoi}_{split} {len(files)} tiles, "
          f"{out_zip.stat().st_size/1e9:.2f} GB, {time.time()-t0:.0f}s", flush=True)


def upload_one(path, ctx):
    """Upload a single file as a blob; return its token."""
    path = Path(path)
    token = api._upload_blob(str(path), True, ApiBlobType.DATASET, ctx)
    if token is None:
        raise RuntimeError(f"upload failed: {path.name}")
    print(f"  UPLOADED {path.name} -> token {str(token)[:12]}...", flush=True)
    return token


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens()

    if "--status" in sys.argv:
        print(f"uploaded blobs: {len(tokens)}")
        for k in sorted(tokens):
            print("  ", k)
        return

    # --- 1. metadata files (CSVs + stats) ---
    meta_files = []
    for csv in sorted(META_FULL.glob("*_train_tile_index.csv")) + \
                sorted(META_FULL.glob("*_val_tile_index.csv")):
        meta_files.append(csv)
    meta_files.append(STATS)

    print(f"=== Uploading {len(meta_files)} metadata files ===", flush=True)
    with ResumableUploadContext() as ctx:
        for f in meta_files:
            if f.name in tokens:
                print(f"  SKIP (already uploaded) {f.name}", flush=True)
                continue
            tokens[f.name] = upload_one(f, ctx)
            save_tokens(tokens)

    # --- 2. tile zips (one at a time) ---
    zs = [(aoi, split) for aoi, split in combos()]
    print(f"=== Uploading {len(zs)} tile zips ===", flush=True)
    with ResumableUploadContext() as ctx:
        for i, (aoi, split) in enumerate(zs, 1):
            zip_name = f"{aoi}_{split}.zip"
            if zip_name in tokens:
                print(f"  SKIP (already uploaded) {zip_name}", flush=True)
                continue
            zp = STAGE / zip_name
            # validate any leftover zip (partial from a killed run); re-create if bad
            if zp.exists():
                try:
                    with zipfile.ZipFile(zp) as _z:
                        _z.testzip()
                except (zipfile.BadZipFile, OSError):
                    print(f"  discarding corrupt zip {zip_name}", flush=True)
                    zp.unlink()
            if not zp.exists():
                zip_tiles(aoi, split, zp)
            tok = upload_one(zp, ctx)
            tokens[zip_name] = tok
            save_tokens(tokens)
            zp.unlink()
            print(f"  DELETED {zip_name} (disk freed)  [{i}/{len(zs)}]", flush=True)

    # --- 3. create ONE version with all tokens ---
    print("=== Creating single dataset version with all files ===", flush=True)
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiCreateDatasetVersionRequest,
        ApiCreateDatasetVersionRequestBody,
        ApiDatasetNewFile,
    )

    body = ApiCreateDatasetVersionRequestBody()
    body.version_notes = ("Single version containing all 16 train/val tile zips "
                          "(accumulated via blob upload).")
    body.files = [ApiDatasetNewFile(token=t) for t in tokens.values()]

    req = ApiCreateDatasetVersionRequest()
    req.owner_slug = OWNER
    req.dataset_slug = SLUG
    req.body = body

    with api.build_kaggle_client() as kaggle:
        msg = kaggle.datasets.dataset_api_client.create_dataset_version
        resp = api.with_retry(msg)(req)
    print("VERSION RESPONSE:", resp, flush=True)
    print("ALL DONE: single version created with", len(tokens), "files", flush=True)


if __name__ == "__main__":
    main()
