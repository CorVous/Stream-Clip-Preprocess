"""Video downloader using yt-dlp."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yt_dlp  # type: ignore[import-untyped]

from stream_clip_preprocess.models import VideoInfo

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_logger = logging.getLogger(__name__)

# Characters not allowed in filenames on Windows/macOS/Linux
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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
            percent_str = d.get("_percent_str", "0.0%").strip().rstrip("%")
            try:
                percent = float(percent_str)
            except ValueError:
                percent = 0.0
            on_progress(
                DownloadProgress(
                    percent=percent,
                    speed=d.get("_speed_str", ""),
                    eta=d.get("_eta_str", ""),
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
            local_path=local_path,
        )

    def sanitize_filename(self, name: str) -> str:
        """Remove characters that are unsafe in filenames.

        :param name: Raw filename string
        :return: Sanitized filename safe for all platforms
        """
        return _UNSAFE_CHARS_RE.sub("_", name)
