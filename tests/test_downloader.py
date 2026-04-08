"""Tests for video downloader."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from stream_clip_preprocess.downloader import (
    DownloadError,
    DownloadProgress,
    VideoDownloader,
)
from stream_clip_preprocess.models import VideoInfo

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# DownloadProgress
# ---------------------------------------------------------------------------


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_create(self) -> None:
        """Test creating DownloadProgress."""
        progress = DownloadProgress(
            percent=50.0,
            speed="1.5 MiB/s",
            eta="0:30",
            status="downloading",
        )
        assert progress.percent == pytest.approx(50.0)
        assert progress.speed == "1.5 MiB/s"
        assert progress.eta == "0:30"
        assert progress.status == "downloading"

    def test_done_status(self) -> None:
        """Test a finished download progress."""
        progress = DownloadProgress(
            percent=100.0,
            speed="",
            eta="",
            status="finished",
        )
        assert progress.percent == pytest.approx(100.0)
        assert progress.status == "finished"


# ---------------------------------------------------------------------------
# VideoDownloader
# ---------------------------------------------------------------------------


class TestVideoDownloader:
    """Tests for VideoDownloader."""

    def test_get_info_returns_video_info(self) -> None:
        """Test get_info returns VideoInfo from yt-dlp metadata."""
        fake_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "duration": 212,
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        with patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert isinstance(info, VideoInfo)
        assert info.video_id == "dQw4w9WgXcQ"
        assert info.title == "Rick Astley - Never Gonna Give You Up"
        assert info.duration == pytest.approx(212)

    def test_get_info_raises_on_failure(self) -> None:
        """Test DownloadError raised when yt-dlp fails."""
        with patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.side_effect = Exception("Network error")
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            with pytest.raises(DownloadError):
                downloader.get_info("https://www.youtube.com/watch?v=invalid")

    def test_download_calls_yt_dlp(self, tmp_path: Path) -> None:
        """Test that download invokes yt-dlp with correct options."""
        fake_info = {
            "id": "abc123",
            "title": "Test Video",
            "duration": 120,
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
        }
        with patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            downloader.download(
                "https://www.youtube.com/watch?v=abc123",
                output_dir=tmp_path,
            )

        assert mock_ydl.called

    def test_progress_callback_invoked(self, tmp_path: Path) -> None:
        """Test that progress callback receives DownloadProgress objects."""
        received: list[DownloadProgress] = []

        def on_progress(p: DownloadProgress) -> None:
            received.append(p)

        fake_info = {
            "id": "abc123",
            "title": "Test Video",
            "duration": 120,
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
        }

        with patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            # Capture the ydl_opts to simulate a hook call
            captured_opts: list[dict] = []  # type: ignore[type-arg]

            def capture_opts(opts: dict) -> MagicMock:  # type: ignore[type-arg]
                captured_opts.append(opts)
                return mock_instance

            mock_ydl.side_effect = capture_opts

            downloader = VideoDownloader()
            downloader.download(
                "https://www.youtube.com/watch?v=abc123",
                output_dir=tmp_path,
                on_progress=on_progress,
            )

            # Simulate a progress hook call
            assert captured_opts, "YoutubeDL was not called"
            hook = captured_opts[0].get("progress_hooks", [None])[0]
            assert hook is not None, "No progress hook registered"
            hook({
                "status": "downloading",
                "_percent_str": "50.0%",
                "_speed_str": "1.0 MiB/s",
                "_eta_str": "0:15",
            })

        assert len(received) == 1
        assert received[0].percent == pytest.approx(50.0)
        assert received[0].status == "downloading"

    def test_sanitize_filename(self) -> None:
        """Test filename sanitization removes invalid characters."""
        downloader = VideoDownloader()
        safe = downloader.sanitize_filename('Test: Video / File "Name"')
        assert "/" not in safe
        assert ":" not in safe
        assert '"' not in safe
