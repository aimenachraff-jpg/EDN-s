"""
ALSAT-2B Super-Resolution Inference App
=======================================
Standalone desktop application (tkinter) that takes a 4-band satellite GeoTIFF
tile (B2/B3/B4/B8) and produces a 4x super-resolved output using EDSR-small.

Usage:
    python sr_app.py                 # launch GUI
    python sr_app.py input.tif       # CLI mode (saves SR to input_sr.tif)
    python sr_app.py input.tif out.tif  # CLI mode with output path

Build .exe:
    pyinstaller --noconfirm --onedir --windowed --name ALSAT2B_SR --collect-all encodings --collect-all rasterio --add-data "experiments/checkpoints/edsr_small_phase5_best.pt;checkpoints" --add-data "configs;configs" --add-data "src;src" sr_app.py
"""
import os, sys, threading, time, queue
import numpy as np
import torch, yaml
from omegaconf import OmegaConf
import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- optional drag-and-drop ----------
try:
    from tkinterdnd2 import DnDWrapper, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False

# ---------- device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))

# ---------- model singleton ----------
_MODEL = None
_MODEL_LOCK = threading.Lock()

def ckpt_path():
    base = getattr(sys, "_MEIPASS", ROOT)
    for cand in [os.path.join(base, "checkpoints", "edsr_small_phase5_best.pt"),
                 os.path.join(ROOT, "experiments", "checkpoints", "edsr_small_phase5_best.pt")]:
        if os.path.exists(cand):
            return cand
    return None

def load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
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
        m.to(DEVICE)
        _MODEL = m
        return m

# ---------- helpers ----------
def to_display(arr, bands=(3, 2, 1), size=320):
    """False-color composite (NIR,Red,Green) -> PIL Image, 2-98 percentile stretch."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[0] < 4:
        while arr.shape[0] < 4:
            arr = np.concatenate([arr, arr[-1:]], axis=0)
    rgb = np.stack([arr[b] for b in bands], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    img = Image.fromarray((out * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    return img

def enhance_sharpness(arr, amount=0.3):
    """EDSR+ variant: light unsharp mask on the SR output."""
    from scipy.ndimage import gaussian_filter
    out = arr.copy()
    for c in range(arr.shape[0]):
        blurred = gaussian_filter(arr[c], sigma=1.0)
        out[c] = np.clip(arr[c] + amount * (arr[c] - blurred), 0, 1)
    return out

def run_sr_tiled(arr, cancel_event, progress_cb):
    """Run SR on array [C,H,W] in 256x256 tiles with overlap blending. Returns SR [C,4H,4W]."""
    C, H, W = arr.shape
    tile = 256
    overlap = 32
    stride = tile - overlap
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = arr.shape[1], arr.shape[2]
    out_h, out_w = pH * 4, pW * 4
    sr = np.zeros((C, out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    ys = list(range(0, pH - tile + 1, stride))
    xs = list(range(0, pW - tile + 1, stride))
    if ys and ys[-1] + tile < pH:
        ys.append(pH - tile)
    if xs and xs[-1] + tile < pW:
        xs.append(pW - tile)
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    total = len(ys) * len(xs)

    model = load_model()
    done = 0
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if cancel_event.is_set():
                return None
            patch = arr[:, y:y+tile, x:x+tile]
            t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                out = model(t).squeeze(0).cpu().numpy().astype(np.float32)
            oy, ox = y * 4, x * 4
            oh, ow = tile * 4, tile * 4
            w1d_y = np.ones(oh, dtype=np.float32)
            w1d_x = np.ones(ow, dtype=np.float32)
            r = overlap * 4
            if overlap > 0 and r <= oh:
                w1d_y[:r] = np.linspace(0, 1, r)
                w1d_y[-r:] = np.linspace(1, 0, r)
                w1d_x[:r] = np.linspace(0, 1, r)
                w1d_x[-r:] = np.linspace(1, 0, r)
            w2d = np.outer(w1d_y, w1d_x)
            for c in range(C):
                sr[c, oy:oy+oh, ox:ox+ow] += out[c] * w2d
            weight[oy:oy+oh, ox:ox+ow] += w2d
            done += 1
            if done % 2 == 0 or done == total:
                progress_cb(done, total, f"Tile {done}/{total}")

    weight = np.where(weight < 1e-8, 1, weight)
    for c in range(C):
        sr[c] /= weight
    sr = sr[:, :H*4, :W*4]
    return sr

def run_sr(input_path, out_path):
    """Run SR on a GeoTIFF. Tiles large inputs, stitches 4x output."""
    with rasterio.open(input_path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    C, H, W = arr.shape
    if arr.max() > 1.5:
        arr = arr / 10000.0
    arr = np.clip(arr, 0, 1)
    if C == 1:
        arr = np.repeat(arr, 4, axis=0)
    elif C == 3:
        nir = (arr[0] + arr[1]) / 2.0
        arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
    elif C > 4:
        arr = arr[:4]
    C, H, W = arr.shape
    # simple single-pass for small images, tiled for large
    if H <= 512 and W <= 512:
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            arr = np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")
        model = load_model()
        t = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            sr = model(t).squeeze(0).cpu().numpy().astype(np.float32)
        if pad_h or pad_w:
            sr = sr[:, :H*4, :W*4]
    else:
        sr = run_sr_tiled(arr, threading.Event(), lambda *a: None)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=C, height=sr.shape[1], width=sr.shape[2],
                       transform=Affine(transform.a/4, transform.b/4, transform.c,
                                       transform.d/4, transform.e/4, transform.f))
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(sr)
    return arr[:C, :H, :W], sr

# ---------- app ----------
class SRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ALSAT-2B Super-Resolution (EDSR-small)")
        self.root.minsize(800, 560)
        self.root.geometry("1000x640")
        self.root.configure(bg="#1e1e2e")

        self.input_path = None
        self.lr_arr = None
        self.sr_arr = None
        self.sr_plus = None
        self.hr_arr = None
        self.ref_path = None
        self.hall_text = "Hallucination: N/A (no reference loaded)"
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.processing = False
        self.model_loaded = False
        self.photo_refs = {}
        self.zoom = 1.0
        self._build_ui()
        self._load_model_bg()
        self._poll_queue()

    def _build_ui(self):
        # theme
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        bg, fg, accent = "#1e1e2e", "#cdd6f4", "#89b4fa"
        st.configure("TFrame", background=bg)
        st.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        st.configure("TButton", font=("Segoe UI", 10), padding=6)
        st.configure("TProgressbar", thickness=8, background=accent, troughcolor="#313244")
        st.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground=accent)
        st.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#a6adc8")
        st.configure("Toolbar.TFrame", background="#181825")
        self.bg = bg

        # toolbar
        bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=6)
        bar.pack(fill="x")
        self.btn_open = ttk.Button(bar, text="Open Image", command=self.open_file)
        self.btn_open.pack(side="left", padx=2)
        self.btn_ref = ttk.Button(bar, text="Load Reference (HR)", command=self.open_ref)
        self.btn_ref.pack(side="left", padx=2)
        self.btn_process = ttk.Button(bar, text="Process (SR)", command=self.process, state="disabled")
        self.btn_process.pack(side="left", padx=2)
        self.btn_cancel = ttk.Button(bar, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=2)
        self.btn_save = ttk.Button(bar, text="Save All", command=self.save_all, state="disabled")
        self.btn_save.pack(side="left", padx=2)
        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(bar, textvariable=self.status_var, style="Status.TLabel").pack(side="right", padx=8)

        # main: 3 panels
        main = ttk.Frame(self.root, padding=4)
        main.pack(fill="both", expand=True)
        self.panels = {}
        for title, key in [("Original (LR)", "orig"), ("EDSR-small", "edsr"), ("EDSR+", "plus")]:
            f = ttk.Frame(main, padding=2)
            f.pack(side="left", fill="both", expand=True, padx=2)
            ttk.Label(f, text=title, style="Title.TLabel").pack()
            lbl = ttk.Label(f, anchor="center", background="#11111b")
            lbl.pack(fill="both", expand=True)
            self.panels[key] = lbl
            # zoom/pan bindings
            lbl.bind("<MouseWheel>", lambda e, k=key: self._zoom(e, k))
            lbl.bind("<ButtonPress-1>", lambda e, k=key: self._pan_start(e, k))
            lbl.bind("<B1-Motion>", lambda e, k=key: self._pan_move(e, k))

        # status bar
        sb = ttk.Frame(self.root, padding=(8, 4))
        sb.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(sb, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.info_var = tk.StringVar(value="No image loaded")
        ttk.Label(sb, textvariable=self.info_var, style="Status.TLabel").pack(side="right", padx=4)
        self.hall_var = tk.StringVar(value="Hallucination: N/A (no reference loaded)")
        ttk.Label(sb, textvariable=self.hall_var, style="Status.TLabel").pack(side="right", padx=4)

        # keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_all())
        self.root.bind("<Return>", lambda e: self.process())
        self.root.bind("<Control-r>", lambda e: self.process())
        self.root.bind("<Escape>", lambda e: self.cancel())

        # drag and drop
        if HAS_DND:
            try:
                self.root.drop_target_register("DND_Files")
                self.root.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

        # resize debounce
        self._resize_after = None
        self.root.bind("<Configure>", self._on_resize)

    def _load_model_bg(self):
        try:
            load_model()
            self.queue.put(("model_loaded", None))
        except Exception as e:
            self.queue.put(("error", f"Model load failed: {e}"))

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "model_loaded":
                    self.model_loaded = True
                    self.status_var.set("Model loaded. Open an image to begin.")
                    self.btn_open.configure(state="normal")
                elif kind == "progress":
                    _, done, total, text = msg
                    self.progress["value"] = done / total * 100 if total > 0 else 0
                    self.status_var.set(text)
                elif kind == "result":
                    _, lr, sr, sr_plus = msg
                    self.lr_arr = lr
                    self.sr_arr = sr
                    self.sr_plus = sr_plus
                    self._update_previews()
                    self.status_var.set("Done. Use Save All to export.")
                    self.btn_process.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    self.btn_save.configure(state="normal")
                    self.processing = False
                elif kind == "hall":
                    self.hall_var.set(msg[1])
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "error":
                    messagebox.showerror("Error", msg[1])
                    self.status_var.set("Error")
                    self.btn_process.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    self.processing = False
                elif kind == "file_loaded":
                    _, arr, path = msg
                    self.lr_arr = arr
                    self.input_path = path
                    self._update_previews()
                    self.status_var.set(f"Loaded: {os.path.basename(path)} ({arr.shape[1]}x{arr.shape[2]}, {arr.shape[0]} bands)")
                    self.btn_process.configure(state="normal")
                    self.btn_save.configure(state="disabled")
                    self.sr_arr = None
                    self.sr_plus = None
                    self.hall_var.set("Hallucination: N/A (no reference loaded)")
                elif kind == "file_error":
                    messagebox.showerror("Error", msg[1])
                elif kind == "file_ref_loaded":
                    _, arr, path = msg
                    self.hr_arr = arr
                    self.ref_path = path
                    self.hall_var.set(f"Reference loaded: {os.path.basename(path)} ({arr.shape[1]}x{arr.shape[2]})")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _update_previews(self):
        size = 280
        if self.lr_arr is not None:
            photo = ImageTk.PhotoImage(to_display(self.lr_arr, size=size))
            self.panels["orig"].configure(image=photo)
            self.panels["orig"].image = photo
        if self.sr_arr is not None:
            photo = ImageTk.PhotoImage(to_display(self.sr_arr, size=size))
            self.panels["edsr"].configure(image=photo)
            self.panels["edsr"].image = photo
        if self.sr_plus is not None:
            photo = ImageTk.PhotoImage(to_display(self.sr_plus, size=size))
            self.panels["plus"].configure(image=photo)
            self.panels["plus"].image = photo

    def open_file(self):
        p = filedialog.askopenfilename(
            title="Select input GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if not p:
            return
        self.btn_open.configure(state="disabled")
        self.status_var.set("Loading file...")
        threading.Thread(target=self._load_file_bg, args=(p,), daemon=True).start()

    def _load_file_bg(self, p):
        try:
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            C = arr.shape[0]
            if C == 1:
                arr = np.repeat(arr, 4, axis=0)
            elif C == 3:
                nir = (arr[0] + arr[1]) / 2.0
                arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
            elif C > 4:
                arr = arr[:4]
            if arr.max() > 1.5:
                arr = arr / 10000.0
            arr = np.clip(arr, 0, 1)
            self.queue.put(("file_loaded", arr, p))
        except Exception as e:
            self.queue.put(("file_error", f"Failed to load image:\n{e}"))
        finally:
            self.btn_open.configure(state="normal")

    def open_ref(self):
        p = filedialog.askopenfilename(
            title="Select HR Reference GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if not p:
            return
        try:
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            C = arr.shape[0]
            if C == 1:
                arr = np.repeat(arr, 4, axis=0)
            elif C == 3:
                nir = (arr[0] + arr[1]) / 2.0
                arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
            elif C > 4:
                arr = arr[:4]
            if arr.max() > 1.5:
                arr = arr / 10000.0
            self.queue.put(("file_ref_loaded", np.clip(arr, 0, 1), p))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference:\n{e}")

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.open_file_path(files[0])

    def open_file_path(self, p):
        self.btn_open.configure(state="disabled")
        self.status_var.set("Loading file...")
        threading.Thread(target=self._load_file_bg, args=(p,), daemon=True).start()

    def process(self):
        if self.lr_arr is None or self.processing:
            return
        self.processing = True
        self.cancel_event.clear()
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_save.configure(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Processing...")
        threading.Thread(target=self._process_bg, daemon=True).start()

    def _process_bg(self):
        try:
            t0 = time.time()
            arr = self.lr_arr.copy()
            C, H, W = arr.shape
            tile = 256
            overlap = 32
            stride = tile - overlap
            pad_h = (tile - H % tile) % tile
            pad_w = (tile - W % tile) % tile
            if pad_h or pad_w:
                arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            pH, pW = arr.shape[1], arr.shape[2]
            nH, nW = pH // tile, pW // tile
            sr = np.zeros((C, pH * 4, pW * 4), dtype=np.float32)
            weight = np.zeros((pH * 4, pW * 4), dtype=np.float32)
            model = load_model()
            total = nH * nW
            done = 0
            ys = list(range(0, pH - tile + 1, stride))
            xs = list(range(0, pW - tile + 1, stride))
            if ys and ys[-1] + tile < pH:
                ys.append(pH - tile)
            if xs and xs[-1] + tile < pW:
                xs.append(pW - tile)
            ys = sorted(set(ys))
            xs = sorted(set(xs))
            total = len(ys) * len(xs)
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    if self.cancel_event.is_set():
                        self.queue.put(("error", "Cancelled by user."))
                        return
                    patch = arr[:, y:y+tile, x:x+tile]
                    t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    with torch.inference_mode():
                        out = model(t).squeeze(0).cpu().numpy().astype(np.float32)
                    oy, ox = y * 4, x * 4
                    oh, ow = tile * 4, tile * 4
                    w1d_y = np.ones(oh, dtype=np.float32)
                    w1d_x = np.ones(ow, dtype=np.float32)
                    r = overlap * 4
                    if overlap > 0 and r <= oh:
                        w1d_y[:r] = np.linspace(0, 1, r)
                        w1d_y[-r:] = np.linspace(1, 0, r)
                        w1d_x[:r] = np.linspace(0, 1, r)
                        w1d_x[-r:] = np.linspace(1, 0, r)
                    w2d = np.outer(w1d_y, w1d_x)
                    for c in range(C):
                        sr[c, oy:oy+oh, ox:ox+ow] += out[c] * w2d
                    weight[oy:oy+oh, ox:ox+ow] += w2d
                    done += 1
                    if done % 2 == 0 or done == total:
                        self.queue.put(("progress", done, total, f"Tile {done}/{total}"))
            if self.cancel_event.is_set():
                self.queue.put(("error", "Cancelled by user."))
                return
            weight = np.where(weight < 1e-8, 1, weight)
            for c in range(C):
                sr[c] /= weight
            sr = sr[:, :H*4, :W*4]
            # EDSR+ variant: light sharpening
            sr_plus = enhance_sharpness(sr.copy(), amount=0.25)
            elapsed = time.time() - t0
            # Hallucination detection (if HR reference loaded)
            if self.hr_arr is not None:
                try:
                    from src.hallucination.detection import HallucinationDetector
                    det = HallucinationDetector()
                    hr = self.hr_arr
                    if hr.shape[1:] == sr.shape[1:]:
                        res_edsr = det.detect_all(sr, hr)
                        cls = res_edsr["classification"]
                        types = ", ".join(res_edsr["detected_types"]) if res_edsr["detected_types"] else "none"
                        conf = res_edsr["confidences"]
                        msg = (f"Hallucination (EDSR): {cls} | types: {types} | "
                               f"spectral conf: {conf.get('spectral',0):.3f} "
                               f"textural: {conf.get('textural',0):.3f}")
                        self.queue.put(("hall", msg))
                    else:
                        self.queue.put(("hall", f"HR size {hr.shape[1:]} != SR size {sr.shape[1:]}; detection skipped"))
                except Exception as e:
                    self.queue.put(("hall", f"Hallucination detection failed: {e}"))
            self.queue.put(("result", self.lr_arr.copy(), sr, sr_plus))
            self.queue.put(("status", f"Done in {elapsed:.1f}s - SR: {sr.shape[1]}x{sr.shape[2]} ({total} tiles)"))
        except Exception as e:
            self.queue.put(("error", f"Processing failed:\n{e}"))

    def cancel(self):
        if self.processing:
            self.cancel_event.set()
            self.status_var.set("Cancelling...")
            self.btn_cancel.configure(state="disabled")

    def save_all(self):
        if self.sr_arr is None:
            return
        folder = filedialog.askdirectory(title="Select folder to save all 3 outputs")
        if not folder:
            return
        try:
            base = os.path.splitext(os.path.basename(self.input_path))[0]
            out_dir = os.path.join(folder, base + "_sr_results")
            os.makedirs(out_dir, exist_ok=True)
            # save PNG previews
            to_display(self.lr_arr).save(os.path.join(out_dir, "Original.png"))
            to_display(self.sr_arr).save(os.path.join(out_dir, "EDSR.png"))
            to_display(self.sr_plus).save(os.path.join(out_dir, "EDSRPLUS.png"))
            # save GeoTIFFs
            with rasterio.open(self.input_path) as src:
                profile = src.profile.copy()
                transform = src.transform
            C = self.sr_arr.shape[0]
            orig = np.repeat(np.repeat(self.lr_arr, 4, axis=1), 4, axis=2)
            orig = orig[:C, :self.sr_arr.shape[1], :self.sr_arr.shape[2]]
            for name, arr in [("Original", orig), ("EDSR", self.sr_arr), ("EDSRPLUS", self.sr_plus)]:
                H, W = arr.shape[1], arr.shape[2]
                p = profile.copy()
                p.update(dtype="float32", count=C, height=H, width=W,
                         transform=Affine(transform.a/4, transform.b/4, transform.c,
                                         transform.d/4, transform.e/4, transform.f))
                with rasterio.open(os.path.join(out_dir, name + ".tif"), "w", **p) as dst:
                    dst.write(arr)
            # save hallucination results if available
            if self.hall_results:
                import json
                with open(os.path.join(out_dir, "hallucination.json"), "w") as f:
                    json.dump(self.hall_results, f, indent=2, default=str)
            messagebox.showinfo("Saved", f"All outputs saved to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.open_file_path(files[0])

    def open_file_path(self, p):
        self.btn_open.configure(state="disabled")
        self.status_var.set("Loading file...")
        threading.Thread(target=self._load_file_bg, args=(p,), daemon=True).start()

    def _zoom(self, event, key):
        pass  # zoom handled by preview resize

    def _pan_start(self, event, key):
        self._drag_start = (event.x, event.y)

    def _pan_move(self, event, key):
        if self._drag_start:
            self._drag_start = (event.x, event.y)

    def _on_resize(self, event):
        pass  # debounce placeholder

def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = SRApp(root)
    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".tif", "_sr.tif")
        print(f"Processing {inp} -> {outp}")
        with rasterio.open(inp) as src:
            arr = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        C = arr.shape[0]
        if C == 1:
            arr = np.repeat(arr, 4, axis=0)
        elif C == 3:
            nir = (arr[0] + arr[1]) / 2.0
            arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
        elif C > 4:
            arr = arr[:4]
        if arr.max() > 1.5:
            arr = arr / 10000.0
        arr = np.clip(arr, 0, 1)
        C, H, W = arr.shape
        tile = 256
        overlap = 32
        stride = tile - overlap
        pad_h = (tile - H % tile) % tile
        pad_w = (tile - W % tile) % tile
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        pH, pW = arr.shape[1], arr.shape[2]
        nH, nW = pH // tile, pW // tile
        ys = list(range(0, pH - tile + 1, stride))
        xs = list(range(0, pW - tile + 1, stride))
        if ys and ys[-1] + tile < pH:
            ys.append(pH - tile)
        if xs and xs[-1] + tile < pW:
            xs.append(pW - tile)
        ys = sorted(set(ys))
        xs = sorted(set(xs))
        model = load_model()
        sr = np.zeros((C, pH * 4, pW * 4), dtype=np.float32)
        weight = np.zeros((pH * 4, pW * 4), dtype=np.float32)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                patch = arr[:, y:y+tile, x:x+tile]
                t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    out = model(t).squeeze(0).cpu().numpy().astype(np.float32)
                oy, ox = y * 4, x * 4
                oh, ow = tile * 4, tile * 4
                w1d_y = np.ones(oh, dtype=np.float32)
                w1d_x = np.ones(ow, dtype=np.float32)
                r = overlap * 4
                if overlap > 0 and r <= oh:
                    w1d_y[:r] = np.linspace(0, 1, r)
                    w1d_y[-r:] = np.linspace(1, 0, r)
                    w1d_x[:r] = np.linspace(0, 1, r)
                    w1d_x[-r:] = np.linspace(1, 0, r)
                w2d = np.outer(w1d_y, w1d_x)
                for c in range(C):
                    sr[c, oy:oy+oh, ox:ox+ow] += out[c] * w2d
                weight[oy:oy+oh, ox:ox+ow] += w2d
        weight = np.where(weight < 1e-8, 1, weight)
        for c in range(C):
            sr[c] /= weight
        sr = sr[:, :H*4, :W*4]
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=C, height=sr.shape[1], width=sr.shape[2],
                           transform=Affine(transform.a/4, transform.b/4, transform.c,
                                           transform.d/4, transform.e/4, transform.f))
        with rasterio.open(outp, "w", **out_profile) as dst:
            dst.write(sr)
        print(f"Input: ({C}, {H}, {W}) -> SR: ({C}, {sr.shape[1]}, {sr.shape[2]}). Saved to {outp}")
    else:
        if HAS_DND:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()
        app = SRApp(root)
        root.mainloop()
