"""Compose the labelled EDN-S results gallery from real experiment outputs."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

BG = (10, 20, 40)
TEXT = (240, 246, 255)
MUTED = (150, 176, 208)

PANEL = 400
GAP = 26
HEADER = 52
ROW_GAP = 42
FOOTER = 70
W = 4 * PANEL + 5 * GAP
H = HEADER + PANEL + ROW_GAP + HEADER + PANEL + FOOTER

bold = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 22)
small = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 16)

items = [
    ("01_LR_input.png", "LR input", "256 x 256", Image.NEAREST),
    ("02_bicubic_SR.png", "Bicubic x4", "1024 x 1024", Image.LANCZOS),
    ("03_EDSR_SR.png", "EDN-S (EDSR-small) x4", "1024 x 1024", Image.LANCZOS),
    ("04_HR_reference.png", "HR reference", "1024 x 1024", Image.LANCZOS),
]

canvas = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(canvas)


def row(y, title_prefix=""):
    for i, (fname, title, size, resample) in enumerate(items):
        x = GAP + i * (PANEL + GAP)
        img = Image.open(os.path.join(FIG, fname)).convert("RGB")
        img = img.resize((PANEL, PANEL), resample)
        label = title_prefix + title
        draw.text((x, y + 14), label, font=bold, fill=TEXT)
        tw = draw.textlength(size, font=small)
        draw.text((x + PANEL - tw, y + 20), size, font=small, fill=MUTED)
        canvas.paste(img, (x, y + HEADER))
        draw.rectangle(
            [(x - 1, y + HEADER - 1), (x + PANEL, y + HEADER + PANEL)],
            outline=(60, 96, 146),
            width=1,
        )


row(0)

# --- detail row: pick the most textured 200x200 window in the HR image -------
hr = np.asarray(Image.open(os.path.join(FIG, "04_HR_reference.png")).convert("L"), dtype=np.float32)
crop = 200
gy, gx = np.gradient(hr)
energy = np.abs(gy) + np.abs(gx)
# integral images for fast window sums (energy and brightness)
ii = energy.cumsum(0).cumsum(1)
bi = hr.cumsum(0).cumsum(1)
best = (-1, 0, 0)
for y in range(32, hr.shape[0] - crop - 32, 32):
    for x in range(32, hr.shape[1] - crop - 32, 32):
        s = ii[y + crop, x + crop] - ii[y, x + crop] - ii[y + crop, x] + ii[y, x]
        b = bi[y + crop, x + crop] - bi[y, x + crop] - bi[y + crop, x] + bi[y, x]
        mean_b = b / (crop * crop)
        if 35 < mean_b < 165 and s > best[0]:  # skip clouds and empty border
            best = (s, y, x)
_, cy, cx = best

detail_y = HEADER + PANEL + ROW_GAP
for i, (fname, title, size, resample) in enumerate(items):
    x = GAP + i * (PANEL + GAP)
    img = Image.open(os.path.join(FIG, fname)).convert("RGB")
    if img.width == 256:  # LR: map the HR crop back by the x4 factor
        box = (cx // 4, cy // 4, (cx + crop) // 4, (cy + crop) // 4)
    else:
        box = (cx, cy, cx + crop, cy + crop)
    crop_img = img.crop(box).resize((PANEL, PANEL), Image.LANCZOS)
    short = title.replace("EDN-S (EDSR-small) x4", "EDN-S x4")
    draw.text((x, detail_y + 14), "Detail: " + short, font=bold, fill=TEXT)
    tw = draw.textlength("200 x 200 crop", font=small)
    draw.text((x + PANEL - tw, detail_y + 20), "200 x 200 crop", font=small, fill=MUTED)
    canvas.paste(crop_img, (x, detail_y + HEADER))
    draw.rectangle(
        [(x - 1, detail_y + HEADER - 1), (x + PANEL, detail_y + HEADER + PANEL)],
        outline=(60, 96, 146),
        width=1,
    )

caption = (
    "Sentinel-2 test tile (4 bands: B2, B3, B4, B8)  |  simulated ALSAT-2B x4 degradation "
    "(Wald protocol, seed 42)  |  false-colour composite (NIR, Red, Green)"
)
draw.text((GAP, H - FOOTER + 26), caption, font=small, fill=MUTED)

out = os.path.join(OUT, "results_gallery.png")
canvas.save(out, optimize=True)
print("saved", out, canvas.size, "crop", (cx, cy))
