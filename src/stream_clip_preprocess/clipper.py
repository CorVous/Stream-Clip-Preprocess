"""Clip extractor using ffmpeg."""

from __future__ import annotations

import logging
import subprocess  # noqa: S404
from dataclasses import dataclass
from typing import TYPE_CHECKING

import imageio_ffmpeg  # type: ignore[import-untyped]

from stream_clip_preprocess.sanitize import sanitize_filename

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from stream_clip_preprocess.models import ClipConfig, Moment

_logger = logging.getLogger(__name__)


def sanitize_clip_filename(name: str) -> str:
    """Replace unsafe filename characters with underscores.

    :param name: Raw clip name
    :return: Safe filename string (falls back to 'clip' if empty)
    """
    return sanitize_filename(name, fallback="clip")


@dataclass
class ClipResult:
    """Result of a single clip extraction."""

    success: bool
    output_path: Path | None
    moment_clip_name: str
    error: str | None = None


class ClipExtractor:
    """Extracts video clips using ffmpeg stream copy."""

    def compute_padded_start(
        self,
        moment: Moment,
        config: ClipConfig,
        video_duration: float,
    ) -> float:
        """Compute start time with padding, clamped to [0, video_duration].

        :param moment: The moment to clip
        :param config: Clip configuration with padding
        :param video_duration: Total video duration in seconds
        :return: Clamped start time
        """
        return max(0.0, min(moment.start - config.padding, video_duration))

    def compute_padded_end(
        self,
        moment: Moment,
        config: ClipConfig,
        video_duration: float,
    ) -> float:
        """Compute end time with padding, clamped to [0, video_duration].

        :param moment: The moment to clip
        :param config: Clip configuration with padding
        :param video_duration: Total video duration in seconds
        :return: Clamped end time
        """
        return min(video_duration, max(0.0, moment.end + config.padding))

    def extract_clip(
        self,
        moment: Moment,
        video_path: Path,
        config: ClipConfig,
        video_duration: float,
    ) -> ClipResult:
        """Extract a single clip from a video file using ffmpeg.

        :param moment: Moment to extract
        :param video_path: Path to source video file
        :param config: Clip configuration
        :param video_duration: Total video duration (for clamping)
        :return: ClipResult indicating success or failure
        """
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        start = self.compute_padded_start(moment, config, video_duration)
        end = self.compute_padded_end(moment, config, video_duration)

        safe_name = sanitize_clip_filename(moment.clip_name)
        start_tag = int(moment.start)
        end_tag = int(moment.end)
        filename = f"{safe_name}_{start_tag}-{end_tag}.mp4"
        output_path = config.output_dir / filename

        cmd = [
            ffmpeg,
            "-y",  # overwrite
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(output_path),
        ]

        _logger.debug(
            "Extracting clip: %s -> %s [%.1f-%.1f]",
            moment.clip_name,
            output_path.name,
            start,
            end,
        )

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return ClipResult(
                success=False,
                output_path=None,
                moment_clip_name=moment.clip_name,
                error=str(exc),
            )

        if proc.returncode != 0:
            return ClipResult(
                success=False,
                output_path=None,
                moment_clip_name=moment.clip_name,
                error=proc.stderr or "ffmpeg returned non-zero exit code",
            )

        return ClipResult(
            success=True,
            output_path=output_path,
            moment_clip_name=moment.clip_name,
        )

    def extract_all(
        self,
        moments: list[Moment],
        video_path: Path,
        config: ClipConfig,
        video_duration: float,
        on_clip_done: Callable[[ClipResult], None] | None = None,
    ) -> list[ClipResult]:
        """Extract all selected moments from a video.

        :param moments: List of moments (only selected=True are extracted)
        :param video_path: Path to source video
        :param config: Clip configuration
        :param video_duration: Total video duration
        :param on_clip_done: Optional callback per-clip
        :return: List of ClipResult, one per selected moment
        """
        results = []
        for moment in moments:
            if not moment.selected:
                continue
            result = self.extract_clip(
                moment=moment,
                video_path=video_path,
                config=config,
                video_duration=video_duration,
            )
            results.append(result)
            if on_clip_done is not None:
                on_clip_done(result)
        return results
