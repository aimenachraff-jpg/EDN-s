"""Pull the eval kernel log and print the verdict + key metrics."""
import json
import re
import requests

d = json.load(open(r"C:\Users\stream\.kaggle\kaggle.json"))
h = {"Authorization": "Bearer " + d["key"]}
SLUG = "phase-5-edsr-small-eval-vs-bicubic"
r = requests.get(
    f"https://www.kaggle.com/api/v1/kernels/output?userName=aymencharafedn&kernelSlug={SLUG}",
    headers=h, timeout=120)
out = r.json()
log = json.loads(out["log"])
print("total log entries:", len(log))
pat = re.compile(
    r"VERDICT|BEATS|DOES NOT|EDSR-small .* dB|bicubic .* dB|delta_psnr|"
    r"best_val_psnr|TRAINING DONE|regenerated .* val LR|loaded best.pt|"
    r"evaluating on", re.I)
hits = [e for e in log if pat.search(e.get("data", ""))]
print("interesting:", len(hits))
for e in hits:
    print(e.get("data", "").strip()[:300])
print("--- output files ---")
for f in out.get("files", []):
    print(" ", f.get("name"))
