"""Tests for transcript fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api import (  # type: ignore[import-untyped]
    CouldNotRetrieveTranscript,
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
# Helpers for mocking the v1.x instance-based API
# ---------------------------------------------------------------------------


def _make_fetched_transcript(
    data: list[dict[str, str | int | float]],
) -> MagicMock:
    """Build a mock FetchedTranscript that supports .to_raw_data()."""
    ft = MagicMock()
    ft.to_raw_data.return_value = data
    return ft


def _mock_transcript_list(
    manual_data: list[dict[str, str | int | float]] | None = None,
    generated_data: list[dict[str, str | int | float]] | None = None,
) -> MagicMock:
    """Build a mock TranscriptList returned by api.list().

    :param manual_data: Data for find_manually_created_transcript
    :param generated_data: Data for find_generated_transcript
    """
    tl = MagicMock()

    if manual_data is not None:
        manual_transcript = MagicMock()
        manual_transcript.fetch.return_value = _make_fetched_transcript(manual_data)
        tl.find_manually_created_transcript.return_value = manual_transcript
    else:
        tl.find_manually_created_transcript.side_effect = CouldNotRetrieveTranscript(
            "abc"
        )

    if generated_data is not None:
        gen_transcript = MagicMock()
        gen_transcript.fetch.return_value = _make_fetched_transcript(generated_data)
        tl.find_generated_transcript.return_value = gen_transcript
    else:
        tl.find_generated_transcript.side_effect = CouldNotRetrieveTranscript("abc")

    # Make iterable for the last-resort fallback
    if manual_data is not None:
        first = MagicMock()
        first.fetch.return_value = _make_fetched_transcript(manual_data)
        tl.__iter__ = MagicMock(return_value=iter([first]))
    elif generated_data is not None:
        first = MagicMock()
        first.fetch.return_value = _make_fetched_transcript(generated_data)
        tl.__iter__ = MagicMock(return_value=iter([first]))
    else:
        tl.__iter__ = MagicMock(return_value=iter([]))

    return tl


# ---------------------------------------------------------------------------
# TranscriptFetcher
# ---------------------------------------------------------------------------


class TestTranscriptFetcher:
    """Tests for TranscriptFetcher."""

    def test_fetch_returns_segments_from_manual(self) -> None:
        """Test that fetch returns segments from manual transcript."""
        mock_data = [
            {"text": "Hello world", "start": 0.0, "duration": 2.5},
            {"text": "How are you", "start": 2.5, "duration": 3.0},
        ]
        tl = _mock_transcript_list(manual_data=mock_data)
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_cls.return_value.list.return_value = tl
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch("dQw4w9WgXcQ")

        assert len(segments) == 2
        assert isinstance(segments[0], TranscriptSegment)
        assert segments[0].text == "Hello world"
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].duration == pytest.approx(2.5)

    def test_fetch_falls_back_to_generated(self) -> None:
        """Test fallback to auto-generated transcript."""
        mock_data = [
            {"text": "Auto text", "start": 0.0, "duration": 1.0},
        ]
        tl = _mock_transcript_list(generated_data=mock_data)
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_cls.return_value.list.return_value = tl
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch("abc123")

        assert len(segments) == 1
        assert segments[0].text == "Auto text"

    def test_fetch_by_url(self) -> None:
        """Test that fetch_by_url extracts video ID and fetches."""
        mock_data = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        tl = _mock_transcript_list(manual_data=mock_data)
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.list.return_value = tl
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch_by_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert len(segments) == 1
        mock_instance.list.assert_called_once_with("dQw4w9WgXcQ")

    def test_no_transcript_raises(self) -> None:
        """Test NoTranscriptError when list() itself fails."""
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_cls.return_value.list.side_effect = CouldNotRetrieveTranscript("abc")
            fetcher = TranscriptFetcher()
            with pytest.raises(NoTranscriptError):
                fetcher.fetch("abc")

    def test_no_transcripts_at_all_raises(self) -> None:
        """Test NoTranscriptError when listing succeeds but none match."""
        tl = _mock_transcript_list()  # no manual, no generated, empty iter
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_cls.return_value.list.return_value = tl
            fetcher = TranscriptFetcher()
            with pytest.raises(NoTranscriptError):
                fetcher.fetch("abc")

    def test_language_preference(self) -> None:
        """Test that language preference is used for manual lookup."""
        mock_data = [{"text": "Bonjour", "start": 0.0, "duration": 1.5}]
        tl = _mock_transcript_list(manual_data=mock_data)
        with patch(
            "stream_clip_preprocess.transcript.YouTubeTranscriptApi"
        ) as mock_cls:
            mock_cls.return_value.list.return_value = tl
            fetcher = TranscriptFetcher(languages=["fr", "en"])
            fetcher.fetch("abc123")

        tl.find_manually_created_transcript.assert_called_once_with(
            ["fr", "en"],
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
        assert "[0] Hello" in result
        assert "[2] World" in result

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
