"""
EDN-S Evaluation Module
=======================

Metrics for super-resolution quality assessment.
"""

from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_sam,
    compute_ergas,
    compute_all_metrics,
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_sam",
    "compute_ergas",
    "compute_all_metrics",
]