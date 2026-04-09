"""Tests for cache module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stream_clip_preprocess.cache import (
    cache_dir,
    cache_summary,
    clear_cache,
    has_cached_transcript,
    has_cached_video,
    load_cached_transcript,
    save_transcript_to_cache,
    store_video_in_cache,
    transcript_cache_path,
    video_cache_path,
)
from stream_clip_preprocess.models import TranscriptSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(text: str, start: float = 0.0, duration: float = 2.0) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, duration=duration)


# ---------------------------------------------------------------------------
# TestCacheDir
# ---------------------------------------------------------------------------


class TestCacheDir:
    """Tests for cache_dir() helper."""

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """cache_dir returns a Path."""
        result = cache_dir(tmp_path)
        assert isinstance(result, Path)

    def test_is_cache_subdir(self, tmp_path: Path) -> None:
        """cache_dir returns output_dir / 'cache'."""
        assert cache_dir(tmp_path) == tmp_path / "cache"

    def test_pure_no_side_effects(self, tmp_path: Path) -> None:
        """cache_dir does not create the directory."""
        result = cache_dir(tmp_path)
        assert not result.exists()


# ---------------------------------------------------------------------------
# TestTranscriptCachePath / TestVideoCachePath
# ---------------------------------------------------------------------------


class TestTranscriptCachePath:
    """Tests for transcript_cache_path()."""

    def test_returns_json_extension(self, tmp_path: Path) -> None:
        p = transcript_cache_path("abc123", tmp_path)
        assert p.suffix == ".json"

    def test_uses_video_id_as_stem(self, tmp_path: Path) -> None:
        p = transcript_cache_path("dQw4w9WgXcQ", tmp_path)
        assert p.stem == "dQw4w9WgXcQ"

    def test_lives_in_cache_base(self, tmp_path: Path) -> None:
        p = transcript_cache_path("abc123", tmp_path)
        assert p.parent == tmp_path


class TestVideoCachePath:
    """Tests for video_cache_path()."""

    def test_returns_mp4_extension(self, tmp_path: Path) -> None:
        p = video_cache_path("abc123", tmp_path)
        assert p.suffix == ".mp4"

    def test_uses_video_id_as_stem(self, tmp_path: Path) -> None:
        p = video_cache_path("dQw4w9WgXcQ", tmp_path)
        assert p.stem == "dQw4w9WgXcQ"

    def test_lives_in_cache_base(self, tmp_path: Path) -> None:
        p = video_cache_path("abc123", tmp_path)
        assert p.parent == tmp_path


# ---------------------------------------------------------------------------
# TestHasCachedTranscript
# ---------------------------------------------------------------------------


class TestHasCachedTranscript:
    """Tests for has_cached_transcript()."""

    def test_returns_false_when_none(self) -> None:
        """Returns False when cache_base is None."""
        assert has_cached_transcript("abc123", cache_base=None) is False

    def test_returns_false_when_absent(self, tmp_path: Path) -> None:
        assert has_cached_transcript("abc123", cache_base=tmp_path) is False

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "abc123.json").write_text("[]", encoding="utf-8")
        assert has_cached_transcript("abc123", cache_base=tmp_path) is True


# ---------------------------------------------------------------------------
# TestHasCachedVideo
# ---------------------------------------------------------------------------


class TestHasCachedVideo:
    """Tests for has_cached_video()."""

    def test_returns_false_when_none(self) -> None:
        """Returns False when cache_base is None."""
        assert has_cached_video("abc123", cache_base=None) is False

    def test_returns_false_when_absent(self, tmp_path: Path) -> None:
        assert has_cached_video("abc123", cache_base=tmp_path) is False

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "abc123.mp4").write_bytes(b"fake_video")
        assert has_cached_video("abc123", cache_base=tmp_path) is True


# ---------------------------------------------------------------------------
# TestSaveTranscriptToCache
# ---------------------------------------------------------------------------


class TestSaveTranscriptToCache:
    """Tests for save_transcript_to_cache()."""

    def test_noop_when_cache_base_none(self) -> None:
        """Does nothing when cache_base is None (no error)."""
        # Must not raise
        save_transcript_to_cache("abc123", [_seg("hello")], cache_base=None)

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        save_transcript_to_cache("abc123", [_seg("hello")], cache_base=cache)
        assert cache.is_dir()

    def test_writes_json_file(self, tmp_path: Path) -> None:
        save_transcript_to_cache("abc123", [_seg("hello")], cache_base=tmp_path)
        assert (tmp_path / "abc123.json").is_file()

    def test_json_contains_expected_keys(self, tmp_path: Path) -> None:
        save_transcript_to_cache(
            "abc123", [_seg("hello", 1.0, 3.0)], cache_base=tmp_path
        )
        data = json.loads((tmp_path / "abc123.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert set(data[0].keys()) == {"text", "start", "duration"}

    def test_empty_list_writes_empty_array(self, tmp_path: Path) -> None:
        save_transcript_to_cache("abc123", [], cache_base=tmp_path)
        data = json.loads((tmp_path / "abc123.json").read_text(encoding="utf-8"))
        assert data == []


# ---------------------------------------------------------------------------
# TestLoadCachedTranscript
# ---------------------------------------------------------------------------


class TestLoadCachedTranscript:
    """Tests for load_cached_transcript()."""

    def test_returns_none_when_cache_base_none(self) -> None:
        assert load_cached_transcript("abc123", cache_base=None) is None

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert load_cached_transcript("abc123", cache_base=tmp_path) is None

    def test_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / "abc123.json").write_text("not valid json{{", encoding="utf-8")
        assert load_cached_transcript("abc123", cache_base=tmp_path) is None

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        segments = [_seg("Hello", 0.0, 2.5), _seg("World", 2.5, 3.0)]
        save_transcript_to_cache("abc123", segments, cache_base=tmp_path)
        loaded = load_cached_transcript("abc123", cache_base=tmp_path)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].text == "Hello"
        assert loaded[0].start == pytest.approx(0.0)
        assert loaded[0].duration == pytest.approx(2.5)
        assert loaded[1].text == "World"
        assert loaded[1].start == pytest.approx(2.5)

    def test_returns_transcript_segment_objects(self, tmp_path: Path) -> None:
        save_transcript_to_cache("abc123", [_seg("hi", 1.0, 2.0)], cache_base=tmp_path)
        loaded = load_cached_transcript("abc123", cache_base=tmp_path)

        assert loaded is not None
        assert all(isinstance(s, TranscriptSegment) for s in loaded)

    def test_empty_list_roundtrips(self, tmp_path: Path) -> None:
        save_transcript_to_cache("abc123", [], cache_base=tmp_path)
        loaded = load_cached_transcript("abc123", cache_base=tmp_path)
        assert loaded == []

    def test_returns_none_on_missing_key(self, tmp_path: Path) -> None:
        """Corrupt JSON with wrong keys returns None."""
        (tmp_path / "abc123.json").write_text('[{"foo": "bar"}]', encoding="utf-8")
        assert load_cached_transcript("abc123", cache_base=tmp_path) is None


# ---------------------------------------------------------------------------
# TestStoreVideoInCache
# ---------------------------------------------------------------------------


class TestStoreVideoInCache:
    """Tests for store_video_in_cache()."""

    def test_noop_when_cache_base_none(self, tmp_path: Path) -> None:
        """Returns None and does not move when cache_base is None."""
        src = tmp_path / "vid.mp4"
        src.write_bytes(b"data")
        result = store_video_in_cache("abc123", src, cache_base=None)
        assert result is None
        assert src.exists()  # file not moved

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "source" / "abc123.mp4"
        src.parent.mkdir()
        src.write_bytes(b"data")
        cache = tmp_path / "cache"

        store_video_in_cache("abc123", src, cache_base=cache)
        assert cache.is_dir()

    def test_moves_file_to_cache(self, tmp_path: Path) -> None:
        src = tmp_path / "source" / "abc123.mp4"
        src.parent.mkdir()
        src.write_bytes(b"video_content")
        cache = tmp_path / "cache"

        result = store_video_in_cache("abc123", src, cache_base=cache)

        assert result is not None
        assert result == cache / "abc123.mp4"
        assert result.exists()
        assert result.read_bytes() == b"video_content"
        assert not src.exists()  # original removed

    def test_returns_new_path(self, tmp_path: Path) -> None:
        src = tmp_path / "abc123.mp4"
        src.write_bytes(b"x")
        cache = tmp_path / "cache"

        result = store_video_in_cache("abc123", src, cache_base=cache)
        assert isinstance(result, Path)
        assert result.parent == cache

    def test_noop_when_already_in_cache(self, tmp_path: Path) -> None:
        """Source == target → returns the path without error."""
        cache = tmp_path / "cache"
        cache.mkdir()
        target = cache / "abc123.mp4"
        target.write_bytes(b"cached")

        result = store_video_in_cache("abc123", target, cache_base=cache)
        assert result == target
        assert target.exists()

    def test_overwrites_existing_cached_video(self, tmp_path: Path) -> None:
        """A new source replaces an older cached video."""
        src = tmp_path / "source" / "abc123.mp4"
        src.parent.mkdir()
        src.write_bytes(b"new_data")

        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "abc123.mp4").write_bytes(b"old_data")

        result = store_video_in_cache("abc123", src, cache_base=cache)
        assert result is not None
        assert result.read_bytes() == b"new_data"


# ---------------------------------------------------------------------------
# TestCacheSummary
# ---------------------------------------------------------------------------


class TestCacheSummary:
    """Tests for cache_summary()."""

    def test_zeros_when_none(self) -> None:
        s = cache_summary(cache_base=None)
        assert s["video_count"] == 0
        assert s["transcript_count"] == 0
        assert s["total_bytes"] == 0

    def test_zeros_when_dir_missing(self, tmp_path: Path) -> None:
        s = cache_summary(cache_base=tmp_path / "nonexistent")
        assert s["video_count"] == 0
        assert s["transcript_count"] == 0
        assert s["total_bytes"] == 0

    def test_counts_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_bytes(b"x" * 100)
        (tmp_path / "a.json").write_text("[]", encoding="utf-8")
        (tmp_path / "b.mp4").write_bytes(b"y" * 200)

        s = cache_summary(cache_base=tmp_path)
        assert s["video_count"] == 2
        assert s["transcript_count"] == 1
        assert s["total_bytes"] >= 300

    def test_reports_total_size(self, tmp_path: Path) -> None:
        (tmp_path / "vid.mp4").write_bytes(b"z" * 500)
        s = cache_summary(cache_base=tmp_path)
        assert s["total_bytes"] == 500


# ---------------------------------------------------------------------------
# TestClearCache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for clear_cache()."""

    def test_returns_zero_when_none(self) -> None:
        assert clear_cache(cache_base=None) == 0

    def test_returns_zero_when_empty(self, tmp_path: Path) -> None:
        assert clear_cache(cache_base=tmp_path) == 0

    def test_removes_all_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_bytes(b"a")
        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        clear_cache(cache_base=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_returns_count_of_removed(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_bytes(b"a")
        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        (tmp_path / "c.mp4").write_bytes(b"c")
        assert clear_cache(cache_base=tmp_path) == 3

    def test_preserves_cache_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_bytes(b"a")
        clear_cache(cache_base=tmp_path)
        assert tmp_path.is_dir()

    def test_returns_zero_for_nonexistent_dir(self, tmp_path: Path) -> None:
        assert clear_cache(cache_base=tmp_path / "ghost") == 0
