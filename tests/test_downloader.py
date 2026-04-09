"""Tests for video downloader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pytest

from stream_clip_preprocess.downloader import (
    DownloadError,
    DownloadProgress,
    VideoDownloader,
    extract_game_from_youtube,
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
        with (
            patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl,
            patch(
                "stream_clip_preprocess.downloader.extract_game_from_youtube",
                return_value=None,
            ),
        ):
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

    def test_get_info_extracts_game_name(self) -> None:
        """Test get_info extracts game name from yt-dlp metadata."""
        fake_info = {
            "id": "stream123",
            "title": "Playing Minecraft with viewers!",
            "duration": 7200,
            "webpage_url": "https://www.youtube.com/watch?v=stream123",
            "categories": ["Gaming"],
            "game": "Minecraft",
        }
        with patch(
            "stream_clip_preprocess.downloader.yt_dlp.YoutubeDL",
        ) as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(
                return_value=mock_instance,
            )
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.get_info(
                "https://www.youtube.com/watch?v=stream123",
            )

        assert info.game == "Minecraft"

    def test_get_info_game_none_when_absent(self) -> None:
        """Test get_info returns None game when not in metadata."""
        fake_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "duration": 212,
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        with (
            patch(
                "stream_clip_preprocess.downloader.yt_dlp.YoutubeDL",
            ) as mock_ydl,
            patch(
                "stream_clip_preprocess.downloader.extract_game_from_youtube",
                return_value=None,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(
                return_value=mock_instance,
            )
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.get_info(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )

        assert info.game is None

    def test_get_info_extracts_categories(self) -> None:
        """Test get_info extracts categories list from yt-dlp metadata."""
        fake_info = {
            "id": "stream123",
            "title": "Gaming stream",
            "duration": 7200,
            "webpage_url": "https://www.youtube.com/watch?v=stream123",
            "categories": ["Gaming"],
            "game": "Minecraft",
        }
        with patch(
            "stream_clip_preprocess.downloader.yt_dlp.YoutubeDL",
        ) as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.get_info("https://www.youtube.com/watch?v=stream123")

        assert info.categories == ["Gaming"]

    def test_get_info_categories_empty_when_absent(self) -> None:
        """Test get_info returns empty categories when not in metadata."""
        fake_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "duration": 212,
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        with (
            patch(
                "stream_clip_preprocess.downloader.yt_dlp.YoutubeDL",
            ) as mock_ydl,
            patch(
                "stream_clip_preprocess.downloader.extract_game_from_youtube",
                return_value=None,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert info.categories == []

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
        """Test that progress callback receives DownloadProgress from raw yt-dlp fields.

        yt-dlp progress hooks provide raw fields (downloaded_bytes, total_bytes,
        speed, eta) — NOT the formatted ``_*_str`` variants.  The hook must
        compute percent, speed, and ETA from those raw values.
        """
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

            # Simulate a realistic progress hook call with raw yt-dlp fields
            assert captured_opts, "YoutubeDL was not called"
            hook = captured_opts[0].get("progress_hooks", [None])[0]
            assert hook is not None, "No progress hook registered"
            hook({
                "status": "downloading",
                "downloaded_bytes": 50_000_000,
                "total_bytes": 100_000_000,
                "speed": 1_500_000,
                "eta": 33,
                "filename": "abc123.mp4",
                "tmpfilename": "abc123.mp4.part",
                "elapsed": 30,
            })

        assert len(received) == 1
        assert received[0].percent == pytest.approx(50.0)
        assert received[0].status == "downloading"
        assert received[0].speed
        assert received[0].eta

    def test_progress_callback_with_total_bytes_estimate(self, tmp_path: Path) -> None:
        """Test progress with total_bytes_estimate (no total_bytes)."""
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

            hook = captured_opts[0]["progress_hooks"][0]
            hook({
                "status": "downloading",
                "downloaded_bytes": 25_000_000,
                "total_bytes_estimate": 100_000_000,
                "speed": 2_000_000,
                "eta": 37,
                "filename": "abc123.mp4",
                "tmpfilename": "abc123.mp4.part",
                "elapsed": 12,
            })

        assert len(received) == 1
        assert received[0].percent == pytest.approx(25.0)

    def test_progress_callback_unknown_total(self, tmp_path: Path) -> None:
        """Test progress when total size is unknown gives 0%."""
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

            hook = captured_opts[0]["progress_hooks"][0]
            hook({
                "status": "downloading",
                "downloaded_bytes": 25_000_000,
                "speed": 1_000_000,
                "filename": "abc123.mp4",
                "tmpfilename": "abc123.mp4.part",
                "elapsed": 25,
            })

        assert len(received) == 1
        assert received[0].percent == pytest.approx(0.0)

    def test_download_returns_game_when_present(self, tmp_path: Path) -> None:
        """Test that download returns game field from yt-dlp metadata."""
        fake_info = {
            "id": "stream123",
            "title": "Playing Minecraft",
            "duration": 7200,
            "webpage_url": "https://www.youtube.com/watch?v=stream123",
            "game": "Minecraft",
        }
        with patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            downloader = VideoDownloader()
            info = downloader.download(
                "https://www.youtube.com/watch?v=stream123",
                output_dir=tmp_path,
            )

        assert info.game == "Minecraft"

    def test_download_game_none_when_absent(self, tmp_path: Path) -> None:
        """Test that download returns None game when not in metadata."""
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
            info = downloader.download(
                "https://www.youtube.com/watch?v=abc123",
                output_dir=tmp_path,
            )

        assert info.game is None

    def test_sanitize_filename(self) -> None:
        """Test filename sanitization removes invalid characters."""
        downloader = VideoDownloader()
        safe = downloader.sanitize_filename('Test: Video / File "Name"')
        assert "/" not in safe
        assert ":" not in safe
        assert '"' not in safe

    def test_sanitize_filename_empty_string(self) -> None:
        """Test filename sanitization with empty string returns empty."""
        downloader = VideoDownloader()
        result = downloader.sanitize_filename("")
        assert not result

    def test_sanitize_filename_unicode(self) -> None:
        """Test filename sanitization preserves Unicode characters."""
        downloader = VideoDownloader()
        result = downloader.sanitize_filename("video_cafe_moment")
        assert "cafe" in result

    def test_sanitize_filename_long_name(self) -> None:
        """Test filename sanitization with very long name."""
        downloader = VideoDownloader()
        result = downloader.sanitize_filename("a" * 300)
        assert len(result) == 300


# ---------------------------------------------------------------------------
# extract_game_from_youtube
# ---------------------------------------------------------------------------


def _fake_youtube_page(game: str | None = None, year: str = "") -> str:
    """Build minimal HTML mimicking YouTube's ytInitialData with game info."""
    if game is None:
        yt_data = {"engagementPanels": []}
    else:
        yt_data = {
            "engagementPanels": [
                {
                    "engagementPanelSectionListRenderer": {
                        "content": {
                            "structuredDescriptionContentRenderer": {
                                "items": [
                                    {
                                        "videoAttributesSectionViewModel": {
                                            "headerTitle": "Games",
                                            "videoAttributeViewModels": [
                                                {
                                                    "videoAttributeViewModel": {
                                                        "title": game,
                                                        "subtitle": year,
                                                    }
                                                }
                                            ],
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ]
        }
    return f"<script>var ytInitialData = {json.dumps(yt_data)};</script>"


class TestExtractGameFromYoutube:
    """Tests for extract_game_from_youtube."""

    def test_extracts_game_name(self) -> None:
        """Game name is parsed from the videoAttributesSectionViewModel."""
        html = _fake_youtube_page(game="Minecraft", year="2009")
        with patch("stream_clip_preprocess.downloader.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(text=html)
            result = extract_game_from_youtube("https://www.youtube.com/watch?v=x")
        assert result == "Minecraft"

    def test_returns_none_when_no_game_section(self) -> None:
        """Returns None for videos without a Games section."""
        html = _fake_youtube_page(game=None)
        with patch("stream_clip_preprocess.downloader.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(text=html)
            result = extract_game_from_youtube("https://www.youtube.com/watch?v=x")
        assert result is None

    def test_returns_none_when_no_yt_initial_data(self) -> None:
        """Returns None when the page has no ytInitialData."""
        with patch("stream_clip_preprocess.downloader.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(text="<html></html>")
            result = extract_game_from_youtube("https://www.youtube.com/watch?v=x")
        assert result is None

    def test_returns_none_on_network_error(self) -> None:
        """Returns None on any HTTP failure."""
        with patch("stream_clip_preprocess.downloader.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("fail")
            result = extract_game_from_youtube("https://www.youtube.com/watch?v=x")
        assert result is None


class TestGetInfoYouTubeGameFallback:
    """Tests for get_info falling back to YouTube page scraping for game."""

    def test_falls_back_to_youtube_page_when_ytdlp_has_no_game(self) -> None:
        """get_info scrapes game from YouTube page when yt-dlp returns None."""
        fake_info = {
            "id": "test123",
            "title": "Playing Minecraft!",
            "duration": 7200,
            "webpage_url": "https://www.youtube.com/watch?v=test123",
        }
        with (
            patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl,
            patch(
                "stream_clip_preprocess.downloader.extract_game_from_youtube"
            ) as mock_extract,
        ):
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance
            mock_extract.return_value = "Minecraft"

            info = VideoDownloader().get_info("https://www.youtube.com/watch?v=test123")

        assert info.game == "Minecraft"
        mock_extract.assert_called_once_with("https://www.youtube.com/watch?v=test123")

    def test_skips_fallback_when_ytdlp_has_game(self) -> None:
        """get_info does NOT scrape the page when yt-dlp already found a game."""
        fake_info = {
            "id": "twitch_vod",
            "title": "Twitch stream",
            "duration": 3600,
            "webpage_url": "https://www.youtube.com/watch?v=twitch_vod",
            "game": "Valorant",
        }
        with (
            patch("stream_clip_preprocess.downloader.yt_dlp.YoutubeDL") as mock_ydl,
            patch(
                "stream_clip_preprocess.downloader.extract_game_from_youtube"
            ) as mock_extract,
        ):
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.extract_info.return_value = fake_info
            mock_ydl.return_value = mock_instance

            info = VideoDownloader().get_info(
                "https://www.youtube.com/watch?v=twitch_vod"
            )

        assert info.game == "Valorant"
        mock_extract.assert_not_called()
