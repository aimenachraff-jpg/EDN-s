"""Verify that the Kaggle s2-tiles-trainval dataset (current version) contains
exactly the expected train+val tiles, compared against the local tile-index CSVs."""
import collections
import json
import sys

import requests

D = json.load(open(r"C:\Users\stream\.kaggle\kaggle.json"))
H = {"Authorization": "Bearer " + D["key"]}
BASE = "https://www.kaggle.com/api/v1/datasets/list/aymencharafedn/s2-tiles-trainval"

files = []
page = ""
import time
while True:
    url = BASE + ("?pagesize=100" if not page else "?pagesize=100&pageToken=" + page)
    for attempt in range(5):
        r = requests.get(url, headers=H, timeout=30)
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"429 rate-limited; waiting {wait}s...", flush=True)
            time.sleep(wait)
        else:
            break
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}")
    data = r.json()
    files.extend(f.get("name") for f in data.get("datasetFiles", []))
    page = data.get("nextPageTokenNullable")
    if not page:
        break
    time.sleep(1)  # be gentle with pagination

print("kaggle files:", len(files))
ext = collections.Counter(f.rsplit(".", 1)[-1] if "." in f else "?" for f in files)
print("by ext:", dict(ext))

# per aoi/split counts for tifs: first path component is the zip name prefix
# (e.g. BR_01_train/BR_01/train/... )
counts = collections.Counter()
for f in files:
    if not f.endswith(".tif"):
        continue
    parts = f.split("/")
    counts[parts[0]] += 1
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")

# expected from local index CSVs
import csv
import glob
from pathlib import Path

expected = {}
for p in glob.glob(r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1\data\metadata\sentinel2\full\*_tile_index.csv"):
    if "_test_" in p:
        continue
    name = Path(p).name.replace("_tile_index.csv", "")
    expected[name] = sum(1 for _ in csv.DictReader(open(p)))

ok = True
for k in sorted(expected):
    got = counts.get(k, 0)
    mark = "OK " if got == expected[k] else "MISMATCH"
    if got != expected[k]:
        ok = False
    print(f"  {mark} {k}: kaggle {got} / expected {expected[k]}")
extra = set(counts) - set(expected)
missing = set(expected) - set(counts)
if extra:
    ok = False
    print("  EXTRA groups:", extra)
if missing:
    ok = False
    print("  MISSING groups:", missing)
total_exp = sum(expected.values())
total_tif = sum(counts.values())
print(f"total tifs: kaggle {total_tif} / expected {total_exp}")
print("csvs:", ext.get("csv"), "/ expected 16   json:", ext.get("json"), "/ expected 1")
print("RESULT:", "ALL DATA PRESENT" if ok and total_tif == total_exp and ext.get("csv") == 16 and ext.get("json") == 1 else "INCOMPLETE")
