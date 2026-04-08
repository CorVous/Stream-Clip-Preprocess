"""Tests for clip extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from stream_clip_preprocess.clipper import (
    ClipExtractor,
    ClipResult,
    sanitize_clip_filename,
)
from stream_clip_preprocess.models import ClipConfig, Moment

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# sanitize_clip_filename
# ---------------------------------------------------------------------------


class TestSanitizeClipFilename:
    """Tests for sanitize_clip_filename helper."""

    def test_removes_slashes(self) -> None:
        """Test that forward and backward slashes are removed."""
        assert "/" not in sanitize_clip_filename("a/b/c")
        assert "\\" not in sanitize_clip_filename("a\\b\\c")

    def test_replaces_spaces(self) -> None:
        """Test that spaces are replaced with underscores."""
        result = sanitize_clip_filename("hello world test")
        assert " " not in result

    def test_removes_special_chars(self) -> None:
        """Test that special characters are removed."""
        result = sanitize_clip_filename('clip: "name" (v1)')
        assert ":" not in result
        assert '"' not in result

    def test_non_empty_result(self) -> None:
        """Test that sanitize never returns empty string."""
        result = sanitize_clip_filename("///")
        assert len(result) > 0 or result == "clip"


# ---------------------------------------------------------------------------
# ClipResult
# ---------------------------------------------------------------------------


class TestClipResult:
    """Tests for ClipResult dataclass."""

    def test_create_success(self, tmp_path: Path) -> None:
        """Test creating a successful ClipResult."""
        out = tmp_path / "clip.mp4"
        result = ClipResult(success=True, output_path=out, moment_clip_name="test")
        assert result.success is True
        assert result.output_path == out
        assert result.error is None

    def test_create_failure(self) -> None:
        """Test creating a failed ClipResult."""
        result = ClipResult(
            success=False,
            output_path=None,
            moment_clip_name="test",
            error="ffmpeg failed",
        )
        assert result.success is False
        assert result.error == "ffmpeg failed"


# ---------------------------------------------------------------------------
# ClipExtractor
# ---------------------------------------------------------------------------


class TestClipExtractor:
    """Tests for ClipExtractor."""

    def test_clamps_start_to_zero(self, tmp_path: Path) -> None:
        """Test that padded start is clamped to 0."""
        extractor = ClipExtractor()
        config = ClipConfig(output_dir=tmp_path, padding=30)
        moment = Moment(start=10.0, end=60.0, summary="Test", clip_name="test")

        # start - padding = 10 - 30 = -20 -> should clamp to 0
        clamped_start = extractor.compute_padded_start(
            moment, config, video_duration=300
        )
        assert clamped_start == pytest.approx(0.0)

    def test_clamps_end_to_duration(self, tmp_path: Path) -> None:
        """Test that padded end is clamped to video duration."""
        extractor = ClipExtractor()
        config = ClipConfig(output_dir=tmp_path, padding=30)
        moment = Moment(start=250.0, end=280.0, summary="Test", clip_name="test")

        # end + padding = 280 + 30 = 310 → should clamp to 300
        clamped_end = extractor.compute_padded_end(moment, config, video_duration=300)
        assert clamped_end == pytest.approx(300.0)

    def test_normal_padding(self, tmp_path: Path) -> None:
        """Test padding applied normally within video bounds."""
        extractor = ClipExtractor()
        config = ClipConfig(output_dir=tmp_path, padding=30)
        moment = Moment(start=100.0, end=160.0, summary="Test", clip_name="test")

        start = extractor.compute_padded_start(moment, config, video_duration=600)
        end = extractor.compute_padded_end(moment, config, video_duration=600)

        assert start == pytest.approx(70.0)
        assert end == pytest.approx(190.0)

    def test_extract_clip_calls_ffmpeg(self, tmp_path: Path) -> None:
        """Test that extract_clip invokes ffmpeg subprocess."""
        fake_ffmpeg = "/usr/bin/ffmpeg"
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        config = ClipConfig(output_dir=tmp_path, padding=5)
        moment = Moment(
            start=100.0, end=160.0, summary="Test moment", clip_name="test_moment"
        )

        with (
            patch(
                "stream_clip_preprocess.clipper.imageio_ffmpeg.get_ffmpeg_exe",
                return_value=fake_ffmpeg,
            ),
            patch("stream_clip_preprocess.clipper.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            extractor = ClipExtractor()
            result = extractor.extract_clip(
                moment=moment,
                video_path=video_path,
                config=config,
                video_duration=300.0,
            )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert fake_ffmpeg in cmd
        assert result.success is True

    def test_extract_clip_returns_failure_on_ffmpeg_error(self, tmp_path: Path) -> None:
        """Test that ffmpeg non-zero exit code returns failure result."""
        fake_ffmpeg = "/usr/bin/ffmpeg"
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        config = ClipConfig(output_dir=tmp_path, padding=5)
        moment = Moment(start=100.0, end=160.0, summary="Test", clip_name="fail_clip")

        with (
            patch(
                "stream_clip_preprocess.clipper.imageio_ffmpeg.get_ffmpeg_exe",
                return_value=fake_ffmpeg,
            ),
            patch("stream_clip_preprocess.clipper.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")
            extractor = ClipExtractor()
            result = extractor.extract_clip(
                moment=moment,
                video_path=video_path,
                config=config,
                video_duration=300.0,
            )

        assert result.success is False
        assert result.error is not None

    def test_output_filename_format(self, tmp_path: Path) -> None:
        """Test that output filename follows expected format."""
        fake_ffmpeg = "/usr/bin/ffmpeg"
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        config = ClipConfig(output_dir=tmp_path, padding=5)
        moment = Moment(start=100.0, end=160.0, summary="Test", clip_name="my_clip")

        with (
            patch(
                "stream_clip_preprocess.clipper.imageio_ffmpeg.get_ffmpeg_exe",
                return_value=fake_ffmpeg,
            ),
            patch("stream_clip_preprocess.clipper.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            extractor = ClipExtractor()
            result = extractor.extract_clip(
                moment=moment,
                video_path=video_path,
                config=config,
                video_duration=300.0,
            )

        assert result.output_path is not None
        assert "my_clip" in result.output_path.name
        assert result.output_path.suffix == ".mp4"
