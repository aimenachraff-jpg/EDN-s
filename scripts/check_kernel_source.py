"""Verify the pushed kernel's effective settings and run status."""
import json
import requests

d = json.load(open(r"C:\Users\stream\.kaggle\kaggle.json"))
h = {"Authorization": "Bearer " + d["key"]}
base = "https://www.kaggle.com/api/v1"
SLUG = "phase-5-edsr-small-x4-training-s2-sim-alsat-2b"

r = requests.get(f"{base}/kernels/status?userName=aymencharafedn&kernelSlug={SLUG}",
                 headers=h, timeout=30)
print("status:", r.json())

r = requests.get(f"{base}/kernels/pull?userName=aymencharafedn&kernelSlug={SLUG}",
                 headers=h, timeout=60)
o = r.json()
meta = o.get("metadata", {})
if isinstance(meta, str):
    try:
        meta = json.loads(meta)
    except Exception:
        pass
k = meta.get("kaggle", meta) if isinstance(meta, dict) else {}
print("accelerator:", k.get("accelerator"))
print("gpu enabled:", k.get("isGpuEnabled"))
print("internet:", k.get("isInternetEnabled"))
print("dataSources:", k.get("dataSources"))
src = o.get("blob", "")
if isinstance(src, dict):
    src = json.dumps(src)
for marker in ("timing probe", "precompute done", "TRAINING DONE", "VERDICT"):
    print(f"  contains {marker!r}:", marker in src)
