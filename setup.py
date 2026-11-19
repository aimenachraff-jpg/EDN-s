#!/usr/bin/env python3
"""
EDN-S: Enhanced Deep Network for Satellite Super-Resolution
==========================================================

A PyTorch implementation of EDSR-small for satellite imagery super-resolution.
Trained on Sentinel-2 simulated ALSAT-2B ×4 dataset.

Author: Aimen Achraff
License: MIT
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="edns",
    version="1.0.0",
    author="Aimen Achraff",
    author_email="aimenachraff@example.com",
    description="EDN-S: Enhanced Deep Network for Satellite Super-Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimenachraff-jpg/MASTER2-A1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "rasterio>=1.3.0",
        "pillow>=9.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0"],
        "gui": ["tkinter"],
        "cloud": ["kagglehub>=0.3.0", "kaggle>=1.7.0"],
    },
    entry_points={
        "console_scripts": [
            "edns-sr=edns.inference.app:main",
        ],
    },
)