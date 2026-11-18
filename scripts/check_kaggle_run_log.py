"""Inspect the Kaggle kernel run log fully (times, pages) and file pagination."""
import json
import requests

d = json.load(open(r"C:\Users\stream\.kaggle\kaggle.json"))
h = {"Authorization": "Bearer " + d["key"]}
url = ("https://www.kaggle.com/api/v1/kernels/output?userName=aymencharafedn"
       "&kernelSlug=notebookcd261f5e8e")
r = requests.get(url, headers=h, timeout=60)
out = r.json()
print("hasNextPageToken:", out.get("hasNextPageToken"),
      "| nextPageToken:", out.get("nextPageToken"))

log = json.loads(out["log"])
print(f"--- all {len(log)} log entries ---")
for e in log:
    s = e["data"].strip().replace("\n", " | ")
    print(f"t={e.get('time'):.1f}s [{e['stream_name']}] {s[:180]}")

# files with pagination
page = ""
n = 0
while True:
    u = url + ("&pageToken=" + page if page else "")
    rr = requests.get(u, headers=h, timeout=60)
    o = rr.json()
    fs = o.get("files", [])
    n += len(fs)
    for f in fs:
        print("file:", f.get("name"))
    if not o.get("hasNextPageToken") or not o.get("nextPageToken"):
        break
    page = o["nextPageToken"]
print("total output files:", n)
