"""Build the EDN-S documentation banner from a real project image."""
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

W, H = 1600, 440
NAVY = (8, 18, 38)
NAVY_2 = (13, 30, 58)
ACCENT = (94, 176, 255)
TEXT = (244, 248, 255)
MUTED = (158, 183, 214)

# --- background: real HR reference crop -------------------------------------
bg_path = os.path.join(ROOT, "results", "figures", "04_HR_reference.png")
bg = Image.open(bg_path).convert("RGB")

# cover-crop to banner aspect ratio
target_ratio = W / H
w, h = bg.size
if w / h > target_ratio:
    new_w = int(h * target_ratio)
    bg = bg.crop(((w - new_w) // 2, 0, (w + new_w) // 2, h))
else:
    new_h = int(w / target_ratio)
    bg = bg.crop((0, (h - new_h) // 2, w, (h + new_h) // 2))
bg = bg.resize((W, H), Image.LANCZOS)

# desaturate and darken so the overlay stays dominant
bg = ImageEnhance.Color(bg).enhance(0.35)
bg = ImageEnhance.Brightness(bg).enhance(0.55)

canvas = Image.new("RGB", (W, H), NAVY)
canvas.paste(bg, (0, 0))

# --- navy gradient overlay ---------------------------------------------------
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
for x in range(W):
    # darker on the left where the title sits
    t = x / (W - 1)
    alpha = int(232 - 78 * t)
    od.line([(x, 0), (x, H)], fill=NAVY + (alpha,))
canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)

draw = ImageDraw.Draw(canvas)

# --- subtle technical grid ---------------------------------------------------
grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(grid)
for x in range(0, W, 80):
    gd.line([(x, 0), (x, H)], fill=(120, 170, 230, 16), width=1)
for y in range(0, H, 80):
    gd.line([(0, y), (W, y)], fill=(120, 170, 230, 16), width=1)
canvas = Image.alpha_composite(canvas, grid)
draw = ImageDraw.Draw(canvas)

# --- typography --------------------------------------------------------------
bold = r"C:\Windows\Fonts\segoeuib.ttf"
regular = r"C:\Windows\Fonts\segoeui.ttf"
title_font = ImageFont.truetype(bold, 138)
sub_font = ImageFont.truetype(regular, 42)
tag_font = ImageFont.truetype(regular, 24)
mono = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 21)

# accent bar
draw.rectangle([(72, 118), (78, 300)], fill=ACCENT)

draw.text((104, 92), "EDN-S", font=title_font, fill=TEXT)
draw.text((110, 250), "Satellite Image Super-Resolution", font=sub_font, fill=ACCENT)
draw.text(
    (112, 312),
    "Sentinel-2  ·  ALSAT-2B simulation  ·  PyTorch",
    font=tag_font,
    fill=MUTED,
)

# right-side technical caption
caption = ["EDSR-small", "16 residual blocks", "x4 upscaling", "1.52 M parameters"]
y0 = 132
for i, line in enumerate(caption):
    draw.text((W - 330, y0 + i * 34), line, font=mono, fill=(170, 200, 235, 220))

# thin frame line
draw.rectangle([(0, 0), (W - 1, H - 1)], outline=(70, 120, 190, 90), width=1)

out_path = os.path.join(OUT, "banner.png")
canvas.convert("RGB").save(out_path, optimize=True)
print("saved", out_path, canvas.size)
