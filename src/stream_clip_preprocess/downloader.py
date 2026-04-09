"""Video downloader using yt-dlp."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yt_dlp  # type: ignore[import-untyped]

from stream_clip_preprocess.models import VideoInfo
from stream_clip_preprocess.sanitize import sanitize_filename

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Progress information for an active download."""

    percent: float
    speed: str
    eta: str
    status: str


class DownloadError(Exception):
    """Raised when a download fails."""


class VideoDownloader:
    """Downloads YouTube videos using yt-dlp."""

    def get_info(self, url: str) -> VideoInfo:
        """Fetch video metadata without downloading.

        :param url: YouTube video URL
        :return: VideoInfo with metadata
        :raises DownloadError: If metadata fetch fails
        """
        ydl_opts = {"quiet": True, "no_warnings": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            msg = f"Failed to fetch video info for {url!r}: {exc}"
            raise DownloadError(msg) from exc

        return VideoInfo(
            url=info.get("webpage_url", url),
            video_id=info["id"],
            title=info.get("title", "Unknown"),
            duration=float(info.get("duration", 0)),
            game=info.get("game"),
        )

    def download(
        self,
        url: str,
        output_dir: Path,
        on_progress: Callable[[DownloadProgress], None] | None = None,
    ) -> VideoInfo:
        """Download a video to the specified directory.

        :param url: YouTube video URL
        :param output_dir: Directory to save the video
        :param on_progress: Optional callback for progress updates
        :return: VideoInfo with local_path set
        :raises DownloadError: If download fails
        """
        output_template = str(output_dir / "%(id)s.%(ext)s")

        def _hook(d: dict) -> None:  # type: ignore[type-arg]
            if on_progress is None:
                return
            status = d.get("status", "")

            # yt-dlp progress hooks supply raw numeric fields, not the
            # formatted ``_*_str`` variants (those only exist in the
            # console-display code path).  Compute percentage from
            # downloaded_bytes / total_bytes (or total_bytes_estimate).
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            percent = (downloaded / total * 100.0) if total else 0.0

            raw_speed = d.get("speed")
            if raw_speed is not None and raw_speed > 0:
                if raw_speed >= 1_048_576:
                    speed = f"{raw_speed / 1_048_576:.1f} MiB/s"
                elif raw_speed >= 1024:
                    speed = f"{raw_speed / 1024:.1f} KiB/s"
                else:
                    speed = f"{raw_speed:.0f} B/s"
            else:
                speed = ""

            raw_eta = d.get("eta")
            if raw_eta is not None:
                mins, secs = divmod(int(raw_eta), 60)
                eta = f"{mins}:{secs:02d}"
            else:
                eta = ""

            on_progress(
                DownloadProgress(
                    percent=percent,
                    speed=speed,
                    eta=eta,
                    status=status,
                )
            )

        ydl_opts: dict = {  # type: ignore[type-arg]
            "quiet": True,
            "no_warnings": True,
            "outtmpl": output_template,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "progress_hooks": [_hook],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            msg = f"Failed to download video {url!r}: {exc}"
            raise DownloadError(msg) from exc

        video_id = info["id"]
        local_path = output_dir / f"{video_id}.mp4"

        return VideoInfo(
            url=info.get("webpage_url", url),
            video_id=video_id,
            title=info.get("title", "Unknown"),
            duration=float(info.get("duration", 0)),
            game=info.get("game"),
            local_path=local_path,
        )

    def sanitize_filename(self, name: str) -> str:
        """Remove characters that are unsafe in filenames.

        :param name: Raw filename string
        :return: Sanitized filename safe for all platforms
        """
        return sanitize_filename(name)
