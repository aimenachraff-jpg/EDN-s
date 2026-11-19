"""
EDN-S Inference Module
======================

High-level API for running super-resolution inference.
"""

from .app import run_sr, load_model, SRInference

__all__ = [
    "run_sr",
    "load_model",
    "SRInference",
]