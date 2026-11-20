"""Create the metrics comparison chart from verified result files."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "results", "metrics", "phase5_edsr_small_eval.json")) as f:
    p5 = json.load(f)
with open(os.path.join(ROOT, "results", "metrics", "phase6_hallucination.json")) as f:
    p6 = json.load(f)

e = p5["edsr"]["metrics_mean"]
b = p5["bicubic"]["metrics_mean"]

quality = [
    ("PSNR (dB)", e["psnr"], b["psnr"], "{:.2f}", True),
    ("SSIM", e["ssim"], b["ssim"], "{:.3f}", True),
    ("SAM (rad)", e["sam"], b["sam"], "{:.4f}", False),
    ("ERGAS", e["ergas"], b["ergas"], "{:.4f}", False),
]

hall = [
    ("Spectral", 100 * p6["models"]["edsr"]["spectral"]["detection_rate"],
     100 * p6["models"]["bicubic"]["spectral"]["detection_rate"]),
    ("Structural", 100 * p6["models"]["edsr"]["structural"]["detection_rate"],
     100 * p6["models"]["bicubic"]["structural"]["detection_rate"]),
    ("Textural", 100 * p6["models"]["edsr"]["textural"]["detection_rate"],
     100 * p6["models"]["bicubic"]["textural"]["detection_rate"]),
    ("Any", 100 * p6["models"]["edsr"]["any_hallucination_rate"],
     100 * p6["models"]["bicubic"]["any_hallucination_rate"]),
]

BG = "#0A1428"
PANEL = "#0F1F3A"
EDSR = "#5EB0FF"
BIC = "#8A9BB5"
TEXT = "#F0F6FF"
MUTED = "#9DB4D4"

fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), dpi=200)
fig.patch.set_facecolor(BG)

for ax, (name, ev, bv, fmt, higher_better) in zip(axes[0], quality):
    ax.set_facecolor(PANEL)
    bars = ax.bar(["EDN-S", "Bicubic"], [ev, bv], color=[EDSR, BIC], width=0.58)
    ax.set_title(name, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#24456E")
    ax.grid(axis="y", color="#1E3A5F", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ymax = max(ev, bv) * 1.28 if max(ev, bv) > 0 else 1
    ax.set_ylim(0, ymax)
    for rect, val in zip(bars, [ev, bv]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + ymax * 0.02,
                fmt.format(val), ha="center", va="bottom", color=TEXT, fontsize=9)

for ax, (name, ev, bv) in zip(axes[1], hall):
    ax.set_facecolor(PANEL)
    bars = ax.bar(["EDN-S", "Bicubic"], [ev, bv], color=[EDSR, BIC], width=0.58)
    ax.set_title(name + " hallucination", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#24456E")
    ax.grid(axis="y", color="#1E3A5F", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ymax = max(ev, bv, 0.5) * 1.35
    ax.set_ylim(0, ymax)
    for rect, val in zip(bars, [ev, bv]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + ymax * 0.02,
                f"{val:.2f}%", ha="center", va="bottom", color=TEXT, fontsize=9)

fig.suptitle("EDN-S vs bicubic x4 — verified experimental results", color=TEXT,
             fontsize=15, fontweight="bold", y=0.98)
fig.text(0.5, 0.925,
         "Top: 30 validation pairs (Phase 5).  Bottom: 1,742 clean test tiles (Phase 6).  "
         "All values from results/metrics/*.json.",
         ha="center", color=MUTED, fontsize=9)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
out = os.path.join(OUT, "metrics.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("saved", out)
