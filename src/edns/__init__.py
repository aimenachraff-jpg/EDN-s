"""
EDN-S: Enhanced Deep Network for Satellite Super-Resolution
==========================================================

A PyTorch implementation of EDSR-small for satellite imagery super-resolution.
Trained on Sentinel-2 simulated ALSAT-2B ×4 dataset.

Author: Aimen Achraff
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Aimen Achraff"

from .models import (
    EDSR,
    EDSRPretrained,
    create_edsr,
    ResBlock,
    Upsampler,
)

from .inference import (
    run_sr,
    load_model,
    SRInference,
)

from .data import (
    load_sentinel2_band,
    normalize_band,
    denormalize_band,
)

from .evaluation import (
    compute_psnr,
    compute_ssim,
    compute_sam,
    compute_ergas,
    compute_all_metrics,
)

__all__ = [
    "EDSR",
    "EDSRPretrained",
    "create_edsr",
    "ResBlock",
    "Upsampler",
    "run_sr",
    "load_model",
    "SRInference",
    "load_sentinel2_band",
    "normalize_band",
    "denormalize_band",
    "compute_psnr",
    "compute_ssim",
    "compute_sam",
    "compute_ergas",
    "compute_all_metrics",
]