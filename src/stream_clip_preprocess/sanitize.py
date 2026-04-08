"""Shared filename sanitization utilities."""

from __future__ import annotations

import re

# Characters not allowed in filenames across Windows/macOS/Linux,
# plus whitespace (collapsed into underscores).
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


def sanitize_filename(name: str, fallback: str = "") -> str:
    """Replace unsafe filename characters with underscores.

    :param name: Raw filename string
    :param fallback: Value to return if the sanitized name is empty
    :return: Sanitized filename safe for all platforms
    """
    safe = _UNSAFE_FILENAME_RE.sub("_", name).strip("_")
    return safe or fallback
