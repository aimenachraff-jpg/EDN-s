#!/usr/bin/env python3
"""Compatibility shim for legacy tooling.

All package metadata lives in ``pyproject.toml``. This file exists only so
that older ``pip``/``setuptools`` versions can still run ``pip install .``.
"""

from setuptools import setup

setup()
