"""
ALSAT-2B Super-Resolution Inference App
=======================================
Standalone desktop application (tkinter) that takes a 4-band satellite GeoTIFF
tile (B2/B3/B4/B8) and produces a 4x super-resolved output using EDSR-small.

Usage:
    python sr_app.py                 # launch GUI
    python sr_app.py input.tif       # CLI mode (saves SR to input_sr.tif)

Build .exe:
    pyinstaller --onefile --windowed --name ALSAT2B_SR --add-data "experiments/checkpoints/edsr_small_phase5_best.pt;checkpoints" sr_app.py
"""
import os, sys, threading, time
import numpy as np
import torch, yaml
from omegaconf import OmegaConf
import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Resolve checkpoint path (works both as script and PyInstaller bundle)
def ckpt_path():
    base = getattr(sys, "_MEIPASS", ROOT)
    cand = os.path.join(base, "checkpoints", "edsr_small_phase5_best.pt")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(ROOT, "experiments", "checkpoints", "edsr_small_phase5_best.pt")
    if os.path.exists(cand2):
        return cand2
    return None

MODEL = None
def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from src.models.edsr import EDSR
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models", "edsr_small.yaml")))
    mcfg = OmegaConf.create(cfg["model"])
    m = EDSR(n_resblocks=mcfg.n_resblocks, n_feats=mcfg.n_feats, scale=mcfg.scale,
             in_channels=mcfg.in_channels, out_channels=mcfg.out_channels)
    cp = ckpt_path()
    if cp is None:
        raise FileNotFoundError("Checkpoint not found. Place edsr_small_phase5_best.pt in checkpoints/")
    state = torch.load(cp, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    MODEL = m
    return m

def to_display(arr, bands=(3, 2, 1), size=384):
    """False-color composite (NIR,Red,Green) -> PIL Image, percentile stretch."""
    rgb = np.stack([arr[b] for b in bands], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    img = Image.fromarray((out * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    return img

def run_sr(input_path, out_path):
    """Run SR on a GeoTIFF. Tiles large inputs into 256x256 patches, stitches 4x output."""
    with rasterio.open(input_path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform

    C, H, W = arr.shape
    if arr.max() > 1.5:
        arr = arr / 10000.0
    arr = np.clip(arr, 0, 1)

    # tile into 256x256 patches (pad if needed)
    tile = 256
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = arr.shape[1], arr.shape[2]
    nH, nW = pH // tile, pW // tile

    model = load_model()
    sr = np.zeros((C, pH * 4, pW * 4), dtype=np.float32)
    for i in range(nH):
        for j in range(nW):
            patch = arr[:, i*tile:(i+1)*tile, j*tile:(j+1)*tile]
            t = torch.from_numpy(patch).unsqueeze(0)
            with torch.no_grad():
                out = model(t).squeeze(0).cpu().numpy().astype(np.float32)
            sr[:, i*tile*4:(i+1)*tile*4, j*tile*4:(j+1)*tile*4] = out

    # undo padding
    sr = sr[:, :H*4, :W*4]

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=C, height=sr.shape[1], width=sr.shape[2],
                       transform=Affine(transform.a/4, transform.b/4, transform.c,
                                       transform.d/4, transform.e/4, transform.f))
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(sr)

    return arr[:C, :H, :W], sr

class SRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ALSAT-2B Super-Resolution (EDSR-small)")
        self.geometry("900x520")
        self.resizable(False, False)

        self.input_path = None
        self.sr_path = None
        self.lr_arr = None
        self.sr_arr = None

        # --- layout ---
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Open Input Image (GeoTIFF)", command=self.open_file).pack(side="left")
        self.process_btn = ttk.Button(top, text="Process (SR)", command=self.process, state="disabled")
        self.process_btn.pack(side="left", padx=8)
        self.save_btn = ttk.Button(top, text="Save Output", command=self.save_output, state="disabled")
        self.save_btn.pack(side="left")
        self.status = ttk.Label(top, text="Load a 4-band GeoTIFF tile to begin.", foreground="gray")
        self.status.pack(side="right")

        mid = ttk.Frame(self, padding=8)
        mid.pack(fill="both", expand=True)
        self.left_lbl = ttk.Label(mid, text="Input (LR)", anchor="center")
        self.left_lbl.pack(side="left", expand=True)
        self.right_lbl = ttk.Label(mid, text="Super-Resolved (SR)", anchor="center")
        self.right_lbl.pack(side="right", expand=True)

        self.left_img = ttk.Label(mid)
        self.left_img.pack(side="left", expand=True, padx=4)
        self.right_img = ttk.Label(mid)
        self.right_img.pack(side="right", expand=True, padx=4)

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.info = ttk.Label(bottom, text="", foreground="gray")
        self.info.pack()

    def open_file(self):
        p = filedialog.askopenfilename(
            title="Select input GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if not p:
            return
        self.input_path = p
        try:
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 10000.0
            self.lr_arr = np.clip(arr, 0, 1)
            self.left_img.configure(image=ImageTk.PhotoImage(to_display(self.lr_arr)))
            self.left_img.image = self.left_img.cget("image")
            self.status.configure(text=f"Loaded: {os.path.basename(p)} ({arr.shape[1]}x{arr.shape[2]}, {arr.shape[0]} bands)", foreground="black")
            self.process_btn.configure(state="normal")
            self.save_btn.configure(state="disabled")
            self.sr_arr = None
            self.right_img.configure(image="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def process(self):
        if self.lr_arr is None:
            return
        self.process_btn.configure(state="disabled")
        self.status.configure(text="Processing... (CPU, ~1-2s per 256x256 tile)", foreground="blue")
        self.update()

        def worker():
            try:
                t0 = time.time()
                arr = self.lr_arr.copy()
                C, H, W = arr.shape
                tile = 256
                pad_h = (tile - H % tile) % tile
                pad_w = (tile - W % tile) % tile
                if pad_h or pad_w:
                    arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                pH, pW = arr.shape[1], arr.shape[2]
                nH, nW = pH // tile, pW // tile
                sr = np.zeros((C, pH * 4, pW * 4), dtype=np.float32)
                model = load_model()
                total_tiles = nH * nW
                done = 0
                for i in range(nH):
                    for j in range(nW):
                        patch = arr[:, i*tile:(i+1)*tile, j*tile:(j+1)*tile]
                        t = torch.from_numpy(patch).unsqueeze(0)
                        with torch.no_grad():
                            out = model(t).squeeze(0).cpu().numpy().astype(np.float32)
                        sr[:, i*tile*4:(i+1)*tile*4, j*tile*4:(j+1)*tile*4] = out
                        done += 1
                        if done % 4 == 0 or done == total_tiles:
                            self.status.configure(text=f"Processing tile {done}/{total_tiles}...", foreground="blue")
                            self.update()
                sr = sr[:, :H*4, :W*4]
                self.sr_arr = sr
                elapsed = time.time() - t0
                self.right_img.configure(image=ImageTk.PhotoImage(to_display(sr)))
                self.right_img.image = self.right_img.cget("image")
                self.status.configure(text=f"Done in {elapsed:.1f}s - SR: {sr.shape[1]}x{sr.shape[2]} ({total_tiles} tiles)", foreground="green")
                self.save_btn.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Processing failed:\n{e}")
                self.status.configure(text="Error", foreground="red")
            finally:
                self.process_btn.configure(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    def save_output(self):
        if self.sr_arr is None:
            return
        p = filedialog.asksaveasfilename(
            title="Save SR output",
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif"), ("PNG preview", "*.png")])
        if not p:
            return
        try:
            if p.lower().endswith(".png"):
                Image.fromarray(to_display(self.sr_arr)).save(p)
            else:
                # save as float32 GeoTIFF with original transform/CRS if available
                with rasterio.open(self.input_path) as src:
                    profile = src.profile.copy()
                    transform = src.transform
                C, H, W = self.sr_arr.shape
                profile.update(dtype="float32", count=C, height=H, width=W,
                              transform=Affine(transform.a/4, transform.b/4, transform.c,
                                             transform.d/4, transform.e/4, transform.f))
                with rasterio.open(p, "w", **profile) as dst:
                    dst.write(self.sr_arr)
            messagebox.showinfo("Saved", f"Output saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".tif", "_sr.tif")
        print(f"Processing {inp} -> {outp}")
        arr, sr = run_sr(inp, outp)
        print(f"Input: {arr.shape} -> SR: {sr.shape}. Saved to {outp}")
    else:
        app = SRApp()
        app.mainloop()
