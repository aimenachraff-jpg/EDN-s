"""Draw the EDN-S architecture and evaluation diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

BG = "#0A1428"
BOX = "#122544"
BOX_EDGE = "#3D6FA8"
ACCENT = "#5EB0FF"
TEXT = "#F0F6FF"
MUTED = "#9DB4D4"
EVAL = "#0E2A33"
EVAL_EDGE = "#2E8B8B"

fig, ax = plt.subplots(figsize=(16, 8.6), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 8.6)
ax.axis("off")


def box(x, y, w, h, title, sub="", edge=BOX_EDGE, face=BOX, title_size=11, sub_size=8.5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.2, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                color=MUTED, fontsize=sub_size, zorder=3, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold", zorder=3)


def arrow(x1, y1, x2, y2, color=ACCENT, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
    ))


# ---------------------------------------------------------------------------
# Super-resolution pipeline (top row)
# ---------------------------------------------------------------------------
top_y, top_h = 6.05, 1.35
w, gap = 1.86, 0.40
xs = [0.45 + i * (w + gap) for i in range(7)]

labels = [
    ("LR input", "4 bands (B2, B3, B4, B8)\n256 x 256"),
    ("Normalization", "z-score using\ntraining statistics"),
    ("Head", "3 x 3 conv\n64 feature maps"),
    ("Body", "16 residual blocks\nconv - ReLU - conv"),
    ("Upsampler", "2 x [3 x 3 conv\n+ PixelShuffle(2)]"),
    ("Tail", "3 x 3 conv\n4 output channels"),
    ("SR output", "4 bands\n1024 x 1024"),
]
for x, (t, s) in zip(xs, labels):
    box(x, top_y, w, top_h, t, s)

for i in range(6):
    arrow(xs[i] + w + 0.03, top_y + top_h / 2, xs[i + 1] - 0.03, top_y + top_h / 2)

# global residual connection: head output -> body output
res_y = top_y - 0.40
ax.plot([xs[2] + w / 2, xs[2] + w / 2], [top_y, res_y], color=ACCENT, lw=1.3)
ax.plot([xs[2] + w / 2, xs[3] + w / 2], [res_y, res_y], color=ACCENT, lw=1.3)
arrow(xs[3] + w / 2, res_y, xs[3] + w / 2, top_y - 0.01, color=ACCENT, lw=1.3)
ax.text((xs[2] + xs[3]) / 2 + w / 2, res_y - 0.12, "global residual",
        ha="center", va="top", color=MUTED, fontsize=8.5, style="italic")

ax.text(0.45, 8.10, "EDN-S super-resolution pipeline", color=TEXT,
        fontsize=15, fontweight="bold")
ax.text(0.45, 7.72, "EDSR-small  |  x4 upscaling  |  1,518,764 parameters",
        color=ACCENT, fontsize=10.5)

# ---------------------------------------------------------------------------
# Evaluation pipeline (bottom)
# ---------------------------------------------------------------------------
ax.plot([0.45, 15.55], [5.05, 5.05], color="#1E3A5F", lw=1)

ax.text(0.45, 4.62, "Reference-based evaluation", color=TEXT,
        fontsize=13, fontweight="bold")
ax.text(0.45, 4.28, "Requires a high-resolution reference image",
        color=MUTED, fontsize=9.5)

ev_y, ev_h = 1.55, 1.55

# left: quality metrics
box(0.45, ev_y, 5.15, ev_h, "Quality metrics",
    "PSNR  -  SSIM  -  SAM  -  ERGAS\nSCI  -  wavelet HH energy  -  GLCM entropy  -  TUS",
    edge=EVAL_EDGE, face=EVAL)

# centre: HR reference
hr_x, hr_w = 6.35, 2.45
box(hr_x, ev_y, hr_w, ev_h, "HR reference", "4 bands\n1024 x 1024",
    edge=EVAL_EDGE, face=EVAL)

# right: hallucination detection
box(9.60, ev_y, 5.95, ev_h, "Hallucination detection",
    "spectral  -  structural  -  textural\nsemantic (implemented; requires a task model)",
    edge=EVAL_EDGE, face=EVAL)

# HR reference feeds both evaluation modules
arrow(hr_x - 0.03, ev_y + ev_h / 2, 5.62, ev_y + ev_h / 2)
arrow(hr_x + hr_w + 0.03, ev_y + ev_h / 2, 9.58, ev_y + ev_h / 2)

# SR output bus above the evaluation boxes
bus_y = 3.55
sr_cx = xs[6] + w / 2
ax.plot([sr_cx, sr_cx], [top_y - 0.01, bus_y], color=ACCENT, lw=1.5)
ax.plot([3.05, sr_cx], [bus_y, bus_y], color=ACCENT, lw=1.5)
arrow(3.05, bus_y, 3.05, ev_y + ev_h + 0.02)
arrow(12.55, bus_y, 12.55, ev_y + ev_h + 0.02)
ax.text(7.8, bus_y + 0.10, "SR output", color=MUTED, fontsize=8.5, ha="center")

# footnote
ax.text(0.45, 0.55,
        "Phase 6 evaluated spectral, structural and textural detection on 1,742 clean test tiles. "
        "Semantic detection was not evaluated.",
        color=MUTED, fontsize=8.5, style="italic")

out = os.path.join(OUT, "architecture.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("saved", out)
