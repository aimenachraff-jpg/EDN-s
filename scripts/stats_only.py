"""Stats-only recovery: recompute train-split normalization stats from the
existing per-AOI train tile index CSVs (tiles already generated; the killed
rerun completed tiling+indexes but died before the stats pass).

Run:  python scripts/stats_only.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from generate_tiles import MANIFEST_DIR, compute_train_stats  # noqa: E402


def main():
    rows = []
    for f in sorted(MANIFEST_DIR.glob("*_train_tile_index.csv")):
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    print(f"train rows loaded: {len(rows)}")
    if not rows:
        sys.exit("no train index rows found")
    out = compute_train_stats(rows)
    print(f"stats -> {out}")


if __name__ == "__main__":
    main()
