"""Tests for ffmpeg binary locator."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from stream_clip_preprocess.ffmpeg import get_ffmpeg_exe

if TYPE_CHECKING:
    from pathlib import Path

_FROZEN = True
_NOT_FROZEN = False


class TestGetFfmpegExe:
    """Tests for get_ffmpeg_exe helper."""

    def test_returns_bundled_binary_when_frozen(self, tmp_path: Path) -> None:
        """In a frozen app, return the binary under _MEIPASS."""
        binaries_dir = tmp_path / "imageio_ffmpeg" / "binaries"
        binaries_dir.mkdir(parents=True)
        fake_binary = binaries_dir / "ffmpeg-test"
        fake_binary.touch()
        fake_binary.chmod(0o755)

        with (
            patch.object(sys, "frozen", _FROZEN, create=True),
            patch.object(sys, "_MEIPASS", str(tmp_path), create=True),
        ):
            result = get_ffmpeg_exe()

        assert result == str(fake_binary)

    def test_falls_back_to_imageio_ffmpeg_when_not_frozen(self) -> None:
        """When not frozen, delegate to imageio_ffmpeg."""
        with (
            patch.object(sys, "frozen", _NOT_FROZEN, create=True),
            patch(
                "stream_clip_preprocess.ffmpeg.imageio_ffmpeg.get_ffmpeg_exe",
                return_value="/usr/bin/ffmpeg",
            ) as mock_get,
        ):
            result = get_ffmpeg_exe()

        mock_get.assert_called_once()
        assert result == "/usr/bin/ffmpeg"

    def test_skips_non_ffmpeg_files_when_frozen(self, tmp_path: Path) -> None:
        """In a frozen app, skip files like README.md and return the ffmpeg binary."""
        binaries_dir = tmp_path / "imageio_ffmpeg" / "binaries"
        binaries_dir.mkdir(parents=True)
        # README.md comes first alphabetically but must be skipped
        readme = binaries_dir / "README.md"
        readme.write_text("docs")
        fake_binary = binaries_dir / "ffmpeg-macos-aarch64-v7.1"
        fake_binary.touch()
        fake_binary.chmod(0o755)

        with (
            patch.object(sys, "frozen", _FROZEN, create=True),
            patch.object(sys, "_MEIPASS", str(tmp_path), create=True),
        ):
            result = get_ffmpeg_exe()

        assert result == str(fake_binary)

    def test_raises_when_frozen_but_no_binary(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError if frozen but binary dir is empty."""
        binaries_dir = tmp_path / "imageio_ffmpeg" / "binaries"
        binaries_dir.mkdir(parents=True)

        with (
            patch.object(sys, "frozen", _FROZEN, create=True),
            patch.object(sys, "_MEIPASS", str(tmp_path), create=True),
            pytest.raises(FileNotFoundError),
        ):
            get_ffmpeg_exe()
