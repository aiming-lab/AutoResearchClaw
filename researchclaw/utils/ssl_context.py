"""Helpers for creating a consistent HTTPS trust store for urllib clients."""

from __future__ import annotations

import ssl
from functools import lru_cache

import certifi


@lru_cache(maxsize=1)
def get_default_ssl_context() -> ssl.SSLContext:
    """Return an SSL context backed by certifi's CA bundle."""
    return ssl.create_default_context(cafile=certifi.where())
