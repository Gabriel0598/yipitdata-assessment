"""Category standardization.

Maps the raw ``category`` column into a small, consistent taxonomy. The map
lives in :data:`src.config.CATEGORY_MAP` so it is visible in one place.
Unknown categories are preserved as ``UNKNOWN`` rather than silently dropped.
"""
from __future__ import annotations

from .. import config


def standardize_category(raw: str | None) -> str:
    if raw is None:
        return "UNKNOWN"
    key = raw.strip()
    if not key:
        return "UNKNOWN"
    return config.CATEGORY_MAP.get(key, "UNKNOWN")


def is_ai_category(standardized: str) -> bool:
    """True when a standardized category is part of the AI taxonomy."""
    return standardized in config.AI_CATEGORIES
