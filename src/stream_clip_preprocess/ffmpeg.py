"""Locate the ffmpeg binary, handling PyInstaller frozen bundles."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio_ffmpeg  # type: ignore[import-untyped]


def get_ffmpeg_exe() -> str:
    """Return the path to the ffmpeg executable.

    In a frozen PyInstaller bundle the binary lives under
    ``sys._MEIPASS/imageio_ffmpeg/binaries/``.  When running from
    source we fall back to ``imageio_ffmpeg.get_ffmpeg_exe()``.

    :return: Absolute path to the ffmpeg binary.
    :raises FileNotFoundError: If no ffmpeg binary can be found.
    """
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        binaries_dir = meipass / "imageio_ffmpeg" / "binaries"
        if binaries_dir.is_dir():
            for candidate in binaries_dir.iterdir():
                if candidate.is_file():
                    return str(candidate)
        msg = f"ffmpeg binary not found in frozen bundle: {binaries_dir}"
        raise FileNotFoundError(msg)

    return imageio_ffmpeg.get_ffmpeg_exe()
