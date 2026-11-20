"""
EDN-S Inference Application
===========================

High-level API for running satellite image super-resolution.
Supports both programmatic API and CLI usage.

Features:
- Super-resolution inference (EDSR-small)
- Optional hallucination detection (spectral, structural, textural)
- Report generation (JSON, optional visualization)
"""

import os
import sys
import threading
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import yaml
import rasterio
from rasterio.transform import Affine
from omegaconf import OmegaConf

from ..models import EDSR, create_edsr
from ..data.normalization import normalize_stack, load_normalization_stats
from ..hallucination import HallucinationDetector


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))

# Model singleton
_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_default_ckpt_path() -> Optional[str]:
    """Get the default checkpoint path."""
    ROOT = Path(__file__).parent.parent.parent.parent
    candidates = [
        ROOT / "experiments" / "checkpoints" / "edsr_small_phase5_best.pt",
        ROOT / "checkpoints" / "edsr_small_phase5_best.pt",
        Path(sys.prefix) / "share" / "edns" / "edsr_small_phase5_best.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def load_model(ckpt_path: Optional[str] = None, config_path: Optional[str] = None) -> EDSR:
    """
    Load the EDSR-small model with pretrained weights.
    
    Args:
        ckpt_path: Path to checkpoint file (uses default if None)
        config_path: Path to model config YAML (uses default if None)
    
    Returns:
        Loaded EDSR model in eval mode
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        
        # Default config: prefer the copy bundled with the package so that a
        # pip-installed CLI works without a source checkout.
        if config_path is None:
            packaged = Path(__file__).parent.parent / "configs" / "models" / "edsr_small.yaml"
            if packaged.exists():
                config_path = packaged
            else:
                ROOT = Path(__file__).parent.parent.parent.parent
                config_path = ROOT / "configs" / "models" / "edsr_small.yaml"
        
        # Load config
        cfg = yaml.safe_load(open(config_path))
        mcfg = OmegaConf.create(cfg["model"])
        
        # Create model
        model = EDSR(
            n_resblocks=mcfg.n_resblocks,
            n_feats=mcfg.n_feats,
            scale=mcfg.scale,
            in_channels=mcfg.in_channels,
            out_channels=mcfg.out_channels,
        )
        
        # Load checkpoint
        if ckpt_path is None:
            ckpt_path = get_default_ckpt_path()
        
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found. Please provide ckpt_path or place "
                f"edsr_small_phase5_best.pt in experiments/checkpoints/"
            )
        
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        model.to(DEVICE)
        
        _MODEL = model
        return model


class SRInference:
    """
    High-level inference class for satellite super-resolution.
    """
    
    def __init__(self, ckpt_path: Optional[str] = None, config_path: Optional[str] = None):
        self.model = load_model(ckpt_path, config_path)
        self.tile_size = 256
        self.overlap = 32
    
    def preprocess(self, arr: np.ndarray) -> np.ndarray:
        """Preprocess input array: z-score normalize using training stats, handle band count."""
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        C = arr.shape[0]

        # Handle different band counts
        if C == 1:
            arr = np.repeat(arr, 4, axis=0)
        elif C == 3:
            # Estimate NIR from RGB
            nir = (arr[0] + arr[1]) / 2.0
            arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
        elif C > 4:
            arr = arr[:4]

        # Z-score normalize using training statistics (model was trained on z-score data)
        stats = load_normalization_stats()
        arr = normalize_stack(arr, band_order=['B2', 'B3', 'B4', 'B8'], stats=stats, method='zscore')
        # Replace NaN (from clouds/shadows) with 0 before inference
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr
    
    def _pad_to_tile(self, arr: np.ndarray, tile: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Pad array to tile size multiple."""
        C, H, W = arr.shape
        pad_h = (tile - H % tile) % tile
        pad_w = (tile - W % tile) % tile
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        return arr, (pad_h, pad_w)
    
    def predict(self, input_arr: np.ndarray) -> np.ndarray:
        """Alias for run()."""
        return self.run(input_arr)

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """
        Run super-resolution on preprocessed array.

        Args:
            input_arr: Input array of shape [C, H, W] with C=4 bands

        Returns:
            Super-resolved array of shape [C, 4H, 4W]
        """
        arr = self.preprocess(input_arr)
        C, H, W = arr.shape
        
        # For small images, single pass
        if H <= self.tile_size and W <= self.tile_size:
            arr, (pad_h, pad_w) = self._pad_to_tile(arr, 4)
            with torch.inference_mode():
                t = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
                sr = self.model(t).squeeze(0).cpu().numpy().astype(np.float32)
            if pad_h or pad_w:
                sr = sr[:, :H*4, :W*4]
            return sr
        
        # For large images, tiled processing
        return self.run_tiled(arr)
    
    def run_tiled(self, arr: np.ndarray) -> np.ndarray:
        """Run tiled super-resolution with overlap blending."""
        C, H, W = arr.shape
        tile = self.tile_size
        overlap = self.overlap
        stride = tile - overlap
        
        # Calculate output dimensions
        pad_h = (tile - H % tile) % tile
        pad_w = (tile - W % tile) % tile
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        
        pH, pW = arr.shape[1], arr.shape[2]
        out_h, out_w = pH * 4, pW * 4
        
        sr = np.zeros((C, out_h, out_w), dtype=np.float32)
        weight = np.zeros((out_h, out_w), dtype=np.float32)
        
        # Generate tile coordinates
        ys = list(range(0, pH - tile + 1, stride))
        xs = list(range(0, pW - tile + 1, stride))
        if ys and ys[-1] + tile < pH:
            ys.append(pH - tile)
        if xs and xs[-1] + tile < pW:
            xs.append(pW - tile)
        ys = sorted(set(ys))
        xs = sorted(set(xs))
        
        for y in ys:
            for x in xs:
                patch = arr[:, y:y+tile, x:x+tile]
                with torch.inference_mode():
                    t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    out = self.model(t).squeeze(0).cpu().numpy().astype(np.float32)
                
                oy, ox = y * 4, x * 4
                oh, ow = tile * 4, tile * 4
                
                # Overlap blending weights
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
        
        # Normalize by weight
        weight = np.where(weight < 1e-8, 1, weight)
        for c in range(C):
            sr[c] /= weight
        
        # Crop to original size * 4
        sr = sr[:, :H*4, :W*4]
        return sr
    
    def run_from_file(self, input_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Run SR from a GeoTIFF file.
        
        Returns:
            Tuple of (sr_array, profile_dict)
        """
        with rasterio.open(input_path) as src:
            arr = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
        
        sr = self.run(arr)
        
        # Update profile for output
        out_profile = profile.copy()
        out_profile.update(
            dtype="float32",
            count=sr.shape[0],
            height=sr.shape[1],
            width=sr.shape[2],
            transform=Affine(transform.a/4, transform.b/4, transform.c,
                            transform.d/4, transform.e/4, transform.f)
        )
        
        return sr, out_profile
    
    def save(self, sr: np.ndarray, output_path: Union[str, Path], profile: dict):
        """Save SR output to GeoTIFF."""
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(sr)


def run_sr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    ckpt_path: Optional[str] = None,
    config_path: Optional[str] = None,
    compute_hallucination: bool = False,
    hr_reference_path: Optional[Union[str, Path]] = None,
    hallucination_report_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function: run SR on a file and save output.
    
    Args:
        input_path: Path to input GeoTIFF
        output_path: Path to save output GeoTIFF
        ckpt_path: Optional checkpoint path
        config_path: Optional config path
        compute_hallucination: Whether to compute hallucination metrics (requires hr_reference_path)
        hr_reference_path: Path to HR reference GeoTIFF for hallucination detection
        hallucination_report_path: Optional path to save hallucination report JSON
    
    Returns:
        Dict with 'output_path', 'hallucination_report' (if computed)
    """
    inferencer = SRInference(ckpt_path, config_path)
    sr, profile = inferencer.run_from_file(input_path)
    inferencer.save(sr, output_path, profile)
    
    result = {"output_path": str(output_path)}
    
    if compute_hallucination:
        if hr_reference_path is None:
            raise ValueError("hr_reference_path is required when compute_hallucination=True")
        
        # Load HR reference
        with rasterio.open(hr_reference_path) as src:
            hr = src.read().astype(np.float32)
        
        # Ensure SR and HR have same shape
        if sr.shape != hr.shape:
            # Resize HR to match SR if needed
            from skimage.transform import resize
            hr_resized = np.zeros_like(sr)
            for c in range(sr.shape[0]):
                hr_resized[c] = resize(hr[c], sr.shape[1:], order=3, anti_aliasing=True, preserve_range=True)
            hr = hr_resized.astype(np.float32)
        
        # Run hallucination detection
        detector = HallucinationDetector()
        hallucination_result = detector.detect_all(sr, hr)
        
        # detect_all() returns the per-category results under "details"
        details = hallucination_result["details"]
        categories = ["spectral", "structural", "textural", "semantic"]

        # Format report
        report = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "hr_reference_path": str(hr_reference_path),
            "hallucination": hallucination_result,
            "summary": {
                "any_hallucination": len(hallucination_result["detected_types"]) > 0,
                "detected_categories": [
                    cat for cat in categories
                    if details.get(cat, {}).get("detected", False)
                ],
                "max_confidence": max(
                    (details.get(cat, {}).get("confidence", 0.0) for cat in categories),
                    default=0.0,
                ),
            },
        }
        
        result["hallucination_report"] = report
        
        if hallucination_report_path:
            with open(hallucination_report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Hallucination report saved to {hallucination_report_path}")
        
        # Print summary to console
        print("\n=== Hallucination Detection Report ===")
        for cat in categories:
            det = details.get(cat, {})
            status = "DETECTED" if det.get("detected", False) else "Clean"
            conf = det.get("confidence", 0.0)
            print(f"  {cat.capitalize():12s}: {status:9s} (confidence: {conf:.4f})")
        any_detected = len(hallucination_result["detected_types"]) > 0
        print(f"  {'Any':12s}: {'YES' if any_detected else 'NO'}")
        print("======================================\n")
    
    return result


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="edns-sr",
        description="EDN-S Super-Resolution CLI - Satellite Image Super-Resolution with Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  edns-sr input.tif                          # Basic SR, saves to input_sr.tif
  edns-sr input.tif output.tif               # Specify output path
  edns-sr input.tif --hallucination --hr reference.tif  # With hallucination detection
  edns-sr input.tif -o out.tif --hallucination --hr ref.tif --report report.json

Environment variables:
  EDNS_CKPT     Override checkpoint path
  EDNS_CONFIG   Override config path
        """
    )
    parser.add_argument("input", help="Path to input GeoTIFF (4-band Sentinel-2)")
    parser.add_argument("-o", "--output", help="Output path (default: input_sr.tif)")
    parser.add_argument("--ckpt", default=os.environ.get("EDNS_CKPT"), help="Checkpoint path")
    parser.add_argument("--config", default=os.environ.get("EDNS_CONFIG"), help="Config path")
    parser.add_argument("--hallucination", action="store_true", help="Enable hallucination detection")
    parser.add_argument("--hr", dest="hr_reference", help="HR reference GeoTIFF path (required with --hallucination)")
    parser.add_argument("--report", help="Path to save hallucination report JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.hallucination and not args.hr_reference:
        parser.error("--hallucination requires --hr reference.tif")
    
    inp = args.input
    outp = args.output or inp.replace(".tif", "_sr.tif").replace(".tiff", "_sr.tiff")
    
    print(f"Processing {inp} -> {outp}")
    if args.hallucination:
        print(f"Hallucination detection: ON (HR reference: {args.hr_reference})")
    t0 = time.time()
    
    result = run_sr(
        inp, outp, 
        ckpt_path=args.ckpt, 
        config_path=args.config,
        compute_hallucination=args.hallucination,
        hr_reference_path=args.hr_reference,
        hallucination_report_path=args.report
    )
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s. Saved to {outp}")
    
    if args.hallucination and args.report:
        print(f"Hallucination report: {args.report}")


if __name__ == "__main__":
    main()