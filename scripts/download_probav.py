"""
PROBA-V Super-Resolution Dataset Download Script
================================================
Downloads probav_data.zip (~675 MB) from Zenodo record 6327426,
verifies MD5 checksum against the API-verified value, extracts the
archive, and records full provenance in a manifest.

Source (verified 2026-11-13):
  - ESA Kelvins: https://kelvins.esa.int/proba-v-super-resolution/data/
  - Zenodo API:  https://zenodo.org/api/records/6327426
    files[0].key = "probav_data.zip"
    files[0].size = 708416603 bytes (~675 MB)
    files[0].checksum = md5:7101101c7c91eeb5979d472861c1b784
  - License: CC BY 4.0
"""

import hashlib
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probav_download")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ZIP = DATA_ROOT / "raw" / "proba_v" / "probav_data.zip"
OUT_DIR = DATA_ROOT / "raw" / "proba_v"
OUT_META = DATA_ROOT / "metadata" / "probav"
OUT_META.mkdir(parents=True, exist_ok=True)
OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

DOWNLOAD_URL = "https://zenodo.org/api/records/6327426/files/probav_data.zip/content"
EXPECTED_SIZE = 708_416_603
EXPECTED_MD5 = "7101101c7c91eeb5979d472861c1b784"
CHUNK = 1024 * 1024  # 1 MB


def md5_stream(path: Path, chunk_size: int = CHUNK) -> str:
    """Compute MD5 of a file without loading it fully into memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    """Stream the archive to disk with progress logging. Resumes are not
    supported by the Zenodo content endpoint, so a fresh start is used."""
    if OUT_ZIP.exists():
        logger.info(f"Existing file found: {OUT_ZIP} ({OUT_ZIP.stat().st_size} bytes)")
        actual = md5_stream(OUT_ZIP)
        if actual == EXPECTED_MD5:
            logger.info("Existing file checksum matches — skipping download")
            return OUT_ZIP
        logger.warning("Existing file checksum mismatch — re-downloading")
        OUT_ZIP.unlink()

    logger.info(f"Downloading: {DOWNLOAD_URL}")
    logger.info(f"Expected size: {EXPECTED_SIZE:,} bytes ({EXPECTED_SIZE/1e6:.1f} MB)")
    with requests.get(DOWNLOAD_URL, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        logger.info(f"Content-Length: {total:,} bytes")
        done = 0
        last_pct = -1
        with open(OUT_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    pct = int(done * 100 / total) if total else 0
                    if pct >= last_pct + 5:
                        last_pct = pct
                        logger.info(f"  {done:,}/{total:,} bytes ({pct}%)")
    logger.info(f"Download complete: {OUT_ZIP.stat().st_size:,} bytes")
    return OUT_ZIP


def verify(path: Path) -> bool:
    size = path.stat().st_size
    logger.info(f"Verifying: size={size:,} bytes (expected {EXPECTED_SIZE:,})")
    if size != EXPECTED_SIZE:
        logger.warning(f"Size mismatch (may still be OK if Zenodo re-encoded); verifying MD5 anyway")
    actual = md5_stream(path)
    ok = actual == EXPECTED_MD5
    logger.info(f"MD5: {actual} (expected {EXPECTED_MD5}) -> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def extract(path: Path) -> Path:
    extract_dir = OUT_DIR / "probav_data"
    if extract_dir.exists():
        logger.info(f"Extract dir exists — removing: {extract_dir}")
        shutil.rmtree(extract_dir)
    logger.info(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        logger.info(f"Archive contains {len(names)} entries")
        z.extractall(extract_dir)
    logger.info("Extraction complete")
    return extract_dir


def summarize(extract_dir: Path) -> dict:
    """Count imagesets per split/band to verify against official spec."""
    summary = {"splits": {}, "total_files": 0, "total_bytes": 0}
    for root_dir in sorted(extract_dir.rglob("*")):
        if root_dir.is_dir() and root_dir.name in ("train", "test"):
            split = root_dir.name
            bands = {}
            for band_dir in sorted(root_dir.iterdir()):
                if not band_dir.is_dir():
                    continue
                imgsets = [d for d in sorted(band_dir.iterdir()) if d.is_dir()]
                n_files = sum(1 for p in band_dir.rglob("*.png"))
                bands[band_dir.name] = {"imgsets": len(imgsets), "png_files": n_files}
                summary["total_files"] += n_files
            summary["splits"][split] = bands
    for p in extract_dir.rglob("*"):
        if p.is_file():
            summary["total_bytes"] += p.stat().st_size
    return summary


def build_manifest(zip_path: Path, extract_dir: Path, checksum_ok: bool, summary: dict) -> Path:
    manifest = {
        "dataset": "PROBA-V Super-Resolution dataset",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "provider": "Zenodo (ESA Kelvins challenge mirror)",
            "url": DOWNLOAD_URL,
            "record": "https://zenodo.org/record/6327426",
            "license": "CC BY 4.0",
            "citation": "Märtens et al. 'Super-resolution of PROBA-V images using CNNs', "
                        "Astrodynamics 3.4 (2019); PROBA-V Super-Resolution dataset, Zenodo 6327426.",
        },
        "download": {
            "zip_path": str(zip_path.relative_to(PROJECT_ROOT)),
            "size_bytes": zip_path.stat().st_size,
            "expected_size_bytes": EXPECTED_SIZE,
            "md5": md5_stream(zip_path),
            "expected_md5": EXPECTED_MD5,
            "checksum_valid": checksum_ok,
        },
        "structure_summary": summary,
        "notes": [
            "74 hand-selected regions; official split 1160 train / 290 test scenes",
            "Per-scene: 9-19 LR images (300 m, 128x128), 1 HR image (100 m, 384x384)",
            "Bands RED and NIR only; 14-bit radiometry stored in 16-bit PNG",
            "Images NOT pre-registered; cPSNR metric is shift-invariant",
        ],
    }
    out = OUT_META / "probav_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Manifest -> {out}")
    return out


def main():
    logger.info("=== PROBA-V Super-Resolution dataset download ===")
    zip_path = download()
    checksum_ok = verify(zip_path)
    if not checksum_ok:
        logger.error("CHECKSUM MISMATCH — aborting extraction. Data integrity not proven.")
        sys.exit(1)
    extract_dir = extract(zip_path)
    summary = summarize(extract_dir)
    build_manifest(zip_path, extract_dir, checksum_ok, summary)

    sep = "=" * 48
    print(f"\n{sep}")
    print(f"  Checksum : {'VALID' if checksum_ok else 'INVALID'}")
    print(f"  Files    : {summary['total_files']} PNGs, {summary['total_bytes']/1e6:.1f} MB extracted")
    for split, bands in summary["splits"].items():
        for band, info in bands.items():
            print(f"  {split}/{band}: {info['imgsets']} imgsets, {info['png_files']} files")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
