"""Transcript fetcher for YouTube videos."""

from __future__ import annotations

import logging
import re

from youtube_transcript_api import (  # type: ignore[import-untyped]
    NoTranscriptFound,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from stream_clip_preprocess.models import TranscriptSegment

_logger = logging.getLogger(__name__)

# Patterns for extracting YouTube video IDs
_WATCH_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
_SHORT_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})")
_EMBED_RE = re.compile(r"/embed/([A-Za-z0-9_-]{11})")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return the ID if already bare.

    :param url_or_id: YouTube URL or bare video ID
    :return: 11-character YouTube video ID
    :raises ValueError: If no video ID can be extracted
    """
    # Bare video ID
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id

    # Standard watch URL: ?v=ID or &v=ID
    m = _WATCH_RE.search(url_or_id)
    if m:
        return m.group(1)

    # Short URL: youtu.be/ID
    m = _SHORT_RE.search(url_or_id)
    if m:
        return m.group(1)

    # Embed URL: /embed/ID
    m = _EMBED_RE.search(url_or_id)
    if m:
        return m.group(1)

    msg = f"Could not extract video ID from: {url_or_id!r}"
    raise ValueError(msg)


class NoTranscriptError(Exception):
    """Raised when no transcript is available for a video."""


class TranscriptFetcher:
    """Fetches YouTube transcripts using youtube-transcript-api."""

    def __init__(self, languages: list[str] | None = None) -> None:
        """Initialize fetcher with optional language preference.

        :param languages: Ordered list of preferred language codes, e.g. ["en", "fr"]
        """
        self.languages = languages

    def fetch(self, video_id: str) -> list[TranscriptSegment]:
        """Fetch transcript for a given video ID.

        :param video_id: YouTube video ID
        :return: List of TranscriptSegment objects
        :raises NoTranscriptError: If no transcript is available
        """
        _logger.debug("Fetching transcript for video_id=%s", video_id)
        try:
            if self.languages:
                raw = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=self.languages
                )
            else:
                raw = YouTubeTranscriptApi.get_transcript(video_id)
        except (NoTranscriptFound, VideoUnavailable) as exc:
            msg = f"No transcript available for video {video_id!r}"
            raise NoTranscriptError(msg) from exc

        return [
            TranscriptSegment(
                text=entry["text"],
                start=float(entry["start"]),
                duration=float(entry["duration"]),
            )
            for entry in raw
        ]

    def fetch_by_url(self, url: str) -> list[TranscriptSegment]:
        """Fetch transcript by YouTube URL.

        :param url: Full YouTube URL
        :return: List of TranscriptSegment objects
        :raises ValueError: If video ID cannot be extracted
        :raises NoTranscriptError: If no transcript is available
        """
        video_id = extract_video_id(url)
        return self.fetch(video_id)


def format_transcript_for_llm(segments: list[TranscriptSegment]) -> str:
    """Format a list of transcript segments into a string for LLM consumption.

    Each segment is formatted as '[MM:SS] text' on its own line.

    :param segments: List of transcript segments
    :return: Formatted transcript string
    """
    if not segments:
        return ""
    return "\n".join(seg.format_timestamp() for seg in segments)
