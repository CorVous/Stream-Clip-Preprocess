"""Cache layer for downloaded video and transcript files.

Files are stored in a ``cache/`` subdirectory of the user's chosen output
folder.  The ``video_id`` (YouTube's 11-character identifier) is used as the
cache key.

Layout::

    {output_dir}/
        cache/
            dQw4w9WgXcQ.json   # TranscriptSegment list, JSON
            dQw4w9WgXcQ.mp4    # downloaded video

All public functions accept an optional ``cache_base: Path | None`` argument.
When ``None`` (i.e. the user has not configured an output folder yet):

- Read / existence functions return ``None`` / ``False``.
- Write functions silently skip.

This lets callers invoke cache functions unconditionally without first
checking whether an output directory has been set.
"""

from __future__ import annotations

import json
import logging
import shutil
from typing import TYPE_CHECKING

from stream_clip_preprocess.models import TranscriptSegment

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def cache_dir(output_dir: Path) -> Path:
    """Return the cache directory path (``output_dir / "cache"``).

    This is a pure helper — it does **not** create the directory.

    :param output_dir: The user's chosen output folder.
    :return: Path to the ``cache`` subdirectory.
    """
    return output_dir / "cache"


def transcript_cache_path(video_id: str, cache_base: Path) -> Path:
    """Return the path where a transcript JSON file would be cached.

    :param video_id: YouTube video ID.
    :param cache_base: Cache directory (the ``cache/`` subdirectory).
    :return: ``{cache_base}/{video_id}.json``
    """
    return cache_base / f"{video_id}.json"


def video_cache_path(video_id: str, cache_base: Path) -> Path:
    """Return the path where a cached video file would live.

    :param video_id: YouTube video ID.
    :param cache_base: Cache directory (the ``cache/`` subdirectory).
    :return: ``{cache_base}/{video_id}.mp4``
    """
    return cache_base / f"{video_id}.mp4"


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


def has_cached_transcript(
    video_id: str,
    *,
    cache_base: Path | None = None,
) -> bool:
    """Return ``True`` if a cached transcript JSON exists for *video_id*.

    :param video_id: YouTube video ID.
    :param cache_base: Cache directory; returns ``False`` when ``None``.
    :return: Whether the transcript cache file is present.
    """
    if cache_base is None:
        return False
    return transcript_cache_path(video_id, cache_base).is_file()


def has_cached_video(
    video_id: str,
    *,
    cache_base: Path | None = None,
) -> bool:
    """Return ``True`` if a cached ``.mp4`` exists for *video_id*.

    :param video_id: YouTube video ID.
    :param cache_base: Cache directory; returns ``False`` when ``None``.
    :return: Whether the video cache file is present.
    """
    if cache_base is None:
        return False
    return video_cache_path(video_id, cache_base).is_file()


# ---------------------------------------------------------------------------
# Transcript serialisation
# ---------------------------------------------------------------------------


def save_transcript_to_cache(
    video_id: str,
    segments: list[TranscriptSegment],
    *,
    cache_base: Path | None = None,
) -> None:
    """Serialise *segments* to JSON and write to the cache.

    Creates the cache directory if it does not exist.  Silently no-ops when
    *cache_base* is ``None`` or on any I/O error.

    :param video_id: YouTube video ID (used as the file stem).
    :param segments: Transcript segments to cache.
    :param cache_base: Cache directory; skips write when ``None``.
    """
    if cache_base is None:
        _logger.debug("save_transcript_to_cache: cache_base is None, skipping")
        return
    try:
        cache_base.mkdir(parents=True, exist_ok=True)
        data = [
            {"text": s.text, "start": s.start, "duration": s.duration} for s in segments
        ]
        transcript_cache_path(video_id, cache_base).write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        _logger.debug(
            "Saved transcript cache for %s (%d segments)", video_id, len(segments)
        )
    except OSError as exc:
        _logger.debug("Could not write transcript cache for %s: %s", video_id, exc)


def load_cached_transcript(
    video_id: str,
    *,
    cache_base: Path | None = None,
) -> list[TranscriptSegment] | None:
    """Load a cached transcript from disk.

    Returns ``None`` on cache miss, corrupt file, or when *cache_base* is
    ``None``.  Never raises.

    :param video_id: YouTube video ID.
    :param cache_base: Cache directory; returns ``None`` when ``None``.
    :return: Deserialized transcript segments, or ``None``.
    """
    if cache_base is None:
        return None
    path = transcript_cache_path(video_id, cache_base)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(
                text=str(entry["text"]),
                start=float(entry["start"]),
                duration=float(entry["duration"]),
            )
            for entry in data
        ]
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        _logger.debug("Could not load transcript cache for %s: %s", video_id, exc)
        return None
    else:
        _logger.debug(
            "Loaded transcript cache for %s (%d segments)", video_id, len(segments)
        )
        return segments


# ---------------------------------------------------------------------------
# Video caching
# ---------------------------------------------------------------------------


def store_video_in_cache(
    video_id: str,
    source_path: Path,
    *,
    cache_base: Path | None = None,
) -> Path | None:
    """Move *source_path* into the cache directory.

    Returns the new ``Path`` inside the cache on success, ``None`` when
    *cache_base* is ``None``.

    If *source_path* is already the cached location (i.e. the file was served
    from cache in the first place), returns it as-is without any filesystem
    operation.

    Uses :func:`shutil.move` which performs an atomic rename when source and
    destination are on the same filesystem, and falls back to copy-then-delete
    otherwise.

    :param video_id: YouTube video ID (used as the file stem).
    :param source_path: Current path to the video file.
    :param cache_base: Cache directory; skips when ``None``.
    :return: Path to the cached video, or ``None``.
    """
    if cache_base is None:
        _logger.debug("store_video_in_cache: cache_base is None, skipping")
        return None

    target = video_cache_path(video_id, cache_base)

    # No-op if already in place
    if source_path.resolve() == target.resolve():
        _logger.debug("Video %s already in cache, skipping move", video_id)
        return target

    cache_base.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target))
    _logger.debug("Stored video %s in cache: %s", video_id, target)
    return target


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def cache_summary(*, cache_base: Path | None = None) -> dict[str, int]:
    """Return a summary of cache contents.

    :param cache_base: Cache directory; returns zeros when ``None`` or absent.
    :return: Dict with ``video_count``, ``transcript_count``, ``total_bytes``.
    """
    empty: dict[str, int] = {"video_count": 0, "transcript_count": 0, "total_bytes": 0}
    if cache_base is None or not cache_base.is_dir():
        return empty

    video_count = 0
    transcript_count = 0
    total_bytes = 0

    for f in cache_base.iterdir():
        if not f.is_file():
            continue
        total_bytes += f.stat().st_size
        if f.suffix == ".mp4":
            video_count += 1
        elif f.suffix == ".json":
            transcript_count += 1

    return {
        "video_count": video_count,
        "transcript_count": transcript_count,
        "total_bytes": total_bytes,
    }


def clear_cache(*, cache_base: Path | None = None) -> int:
    """Delete all files in the cache directory.

    Preserves the directory itself.  Returns the number of files removed.
    Returns 0 when *cache_base* is ``None`` or does not exist.

    :param cache_base: Cache directory.
    :return: Number of files deleted.
    """
    if cache_base is None or not cache_base.is_dir():
        return 0

    removed = 0
    for f in cache_base.iterdir():
        if f.is_file():
            try:
                f.unlink()
                removed += 1
            except OSError as exc:
                _logger.debug("Could not delete cache file %s: %s", f, exc)

    _logger.debug("Cleared cache: %d file(s) removed from %s", removed, cache_base)
    return removed
