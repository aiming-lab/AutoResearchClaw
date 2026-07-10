"""Deterministic experiment scaffold for bounded Stage 10 plugins."""

from __future__ import annotations

from .main_template import render_main_py, scaffold_sha256
from .plugin_validation import PluginValidationError, validate_detector_plugin

__all__ = [
    "PluginValidationError",
    "render_main_py",
    "scaffold_sha256",
    "validate_detector_plugin",
]
