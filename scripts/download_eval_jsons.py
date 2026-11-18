"""Download phase5_eval.json and train_summary.json from the eval kernel output."""
import json
import os
import requests

d = json.load(open(r"C:\Users\stream\.kaggle\kaggle.json"))
h = {"Authorization": "Bearer " + d["key"]}
SLUG = "phase-5-edsr-small-eval-vs-bicubic"
targets = {"phase5_eval.json", "train_summary.json"}

# collect file objects across pages
pages = [None]
tok = None
all_files = []
while True:
    url = (f"https://www.kaggle.com/api/v1/kernels/output?userName=aymencharafedn"
           f"&kernelSlug={SLUG}" + (f"&pageToken={tok}" if tok else ""))
    r = requests.get(url, headers=h, timeout=60)
    out = r.json()
    all_files.extend(out.get("files", []))
    tok = out.get("nextPageToken")
    if not out.get("hasNextPageToken") or not tok:
        break

dest = r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1\tmp\kaggle_eval_out"
os.makedirs(dest, exist_ok=True)
for f in all_files:
    name = f.get("fileName") or f.get("fileNameNullable")
    if not name or os.path.basename(name) not in targets:
        continue
    furl = f.get("url") or f.get("urlNullable")
    print("downloading", name)
    rr = requests.get(furl, headers=h, timeout=120)
    outpath = os.path.join(dest, os.path.basename(name))
    with open(outpath, "wb") as fh:
        fh.write(rr.content)
    print("  saved", outpath, len(rr.content), "bytes")

# print the jsons
for name in ("phase5_eval.json", "train_summary.json"):
    p = os.path.join(dest, name)
    if os.path.exists(p):
        print("=====", name, "=====")
        print(open(p).read()[:1500])
