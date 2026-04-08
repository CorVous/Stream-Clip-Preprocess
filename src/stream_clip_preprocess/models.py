"""Shared data models for stream-clip-preprocess."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class LLMBackend(enum.Enum):
    """LLM backend selection."""

    LOCAL = "local"
    OPENROUTER = "openrouter"


@dataclass
class VideoInfo:
    """Information about a YouTube video."""

    url: str
    video_id: str
    title: str
    duration: float
    local_path: Path | None = None


@dataclass
class TranscriptSegment:
    """A single segment of a YouTube transcript."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        """Return the end time of this segment."""
        return self.start + self.duration

    def format_timestamp(self) -> str:
        """Format segment as '[MM:SS] text' for LLM consumption."""
        total_seconds = int(self.start)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            ts = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            ts = f"{minutes}:{seconds:02d}"
        return f"[{ts}] {self.text}"


@dataclass
class Moment:
    """A notable moment identified in a stream."""

    start: float
    end: float
    summary: str
    clip_name: str
    youtube_url: str | None = None
    selected: bool = True

    @property
    def duration(self) -> float:
        """Return the duration of this moment."""
        return self.end - self.start

    def build_youtube_url(self, video_id: str) -> str:
        """Build a YouTube URL with timestamp for this moment.

        :param video_id: YouTube video ID
        :return: YouTube URL with timestamp parameter
        """
        start_sec = int(self.start)
        return f"https://www.youtube.com/watch?v={video_id}&t={start_sec}"


@dataclass
class ClipConfig:
    """Configuration for clip extraction."""

    output_dir: Path
    padding: int = 30

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.padding < 0:
            msg = "padding must be non-negative"
            raise ValueError(msg)


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    backend: LLMBackend
    model_path: Path | None = None
    api_key: str | None = None
    model_name: str | None = None
    context_window: int = field(default=8192)
