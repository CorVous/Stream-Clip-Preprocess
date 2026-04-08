"""Tests for transcript fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api import (  # type: ignore[import-untyped]
    NoTranscriptFound,
    VideoUnavailable,
)

from stream_clip_preprocess.models import TranscriptSegment
from stream_clip_preprocess.transcript import (
    NoTranscriptError,
    TranscriptFetcher,
    extract_video_id,
    format_transcript_for_llm,
)

# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    """Tests for extract_video_id helper."""

    def test_standard_watch_url(self) -> None:
        """Test extraction from standard watch URL."""
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        """Test extraction from youtu.be short URL."""
        vid = extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_url_with_timestamp(self) -> None:
        """Test extraction from URL with timestamp parameter."""
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s")
        assert vid == "dQw4w9WgXcQ"

    def test_url_with_playlist(self) -> None:
        """Test extraction from URL with playlist parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=2"
        vid = extract_video_id(url)
        assert vid == "dQw4w9WgXcQ"

    def test_already_a_video_id(self) -> None:
        """Test that bare 11-character video ID is returned as-is."""
        vid = extract_video_id("dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self) -> None:
        """Test that an invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="video ID"):
            extract_video_id("https://www.example.com/not-a-youtube-url")

    def test_embed_url(self) -> None:
        """Test extraction from embed URL."""
        vid = extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# TranscriptFetcher
# ---------------------------------------------------------------------------


class TestTranscriptFetcher:
    """Tests for TranscriptFetcher."""

    def test_fetch_returns_segments(self) -> None:
        """Test that fetch returns a list of TranscriptSegment."""
        mock_data = [
            {"text": "Hello world", "start": 0.0, "duration": 2.5},
            {"text": "How are you", "start": 2.5, "duration": 3.0},
        ]
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.get_transcript.return_value = mock_data
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch("dQw4w9WgXcQ")

        assert len(segments) == 2
        assert isinstance(segments[0], TranscriptSegment)
        assert segments[0].text == "Hello world"
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].duration == pytest.approx(2.5)

    def test_fetch_by_url(self) -> None:
        """Test that fetch_by_url extracts video ID and fetches."""
        mock_data = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.get_transcript.return_value = mock_data
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch_by_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert len(segments) == 1
        mock_api.get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    def test_no_transcript_raises(self) -> None:
        """Test NoTranscriptError when no transcript available."""
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.get_transcript.side_effect = NoTranscriptFound(
                "abc", ["en"], MagicMock()
            )
            fetcher = TranscriptFetcher()
            with pytest.raises(NoTranscriptError):
                fetcher.fetch("abc")

    def test_video_unavailable_raises(self) -> None:
        """Test NoTranscriptError when video is unavailable."""
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.get_transcript.side_effect = VideoUnavailable("abc")
            fetcher = TranscriptFetcher()
            with pytest.raises(NoTranscriptError):
                fetcher.fetch("abc")

    def test_language_preference(self) -> None:
        """Test that language preference is passed to the API."""
        mock_data = [{"text": "Bonjour", "start": 0.0, "duration": 1.5}]
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_api:
            mock_api.get_transcript.return_value = mock_data
            fetcher = TranscriptFetcher(languages=["fr", "en"])
            fetcher.fetch("abc123")

        mock_api.get_transcript.assert_called_once_with(
            "abc123", languages=["fr", "en"]
        )


# ---------------------------------------------------------------------------
# format_transcript_for_llm
# ---------------------------------------------------------------------------


class TestFormatTranscriptForLlm:
    """Tests for format_transcript_for_llm."""

    def test_basic_format(self) -> None:
        """Test basic formatting of transcript segments."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=2.0),
            TranscriptSegment(text="World", start=2.0, duration=3.0),
        ]
        result = format_transcript_for_llm(segments)
        assert "[0:00] Hello" in result
        assert "[0:02] World" in result

    def test_empty_segments(self) -> None:
        """Test formatting empty transcript."""
        result = format_transcript_for_llm([])
        assert not result

    def test_segments_joined_by_newlines(self) -> None:
        """Test that segments are separated by newlines."""
        segments = [
            TranscriptSegment(text="A", start=0.0, duration=1.0),
            TranscriptSegment(text="B", start=1.0, duration=1.0),
        ]
        result = format_transcript_for_llm(segments)
        lines = result.strip().split("\n")
        assert len(lines) == 2
