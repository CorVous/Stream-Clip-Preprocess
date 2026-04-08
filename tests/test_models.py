"""Tests for data models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from stream_clip_preprocess.models import (
    ClipConfig,
    LLMBackend,
    LLMConfig,
    Moment,
    TranscriptSegment,
    VideoInfo,
)

# ---------------------------------------------------------------------------
# VideoInfo
# ---------------------------------------------------------------------------


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_create_basic(self) -> None:
        """Test creating a VideoInfo with all fields."""
        info = VideoInfo(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            video_id="dQw4w9WgXcQ",
            title="Rick Astley - Never Gonna Give You Up",
            duration=212,
        )
        assert info.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert info.video_id == "dQw4w9WgXcQ"
        assert info.title == "Rick Astley - Never Gonna Give You Up"
        assert info.duration == 212

    def test_duration_float(self) -> None:
        """Test VideoInfo accepts float duration."""
        info = VideoInfo(
            url="https://www.youtube.com/watch?v=abc",
            video_id="abc",
            title="Test",
            duration=3661.5,
        )
        assert info.duration == pytest.approx(3661.5)

    def test_optional_local_path_default_none(self) -> None:
        """Test local_path defaults to None."""
        info = VideoInfo(
            url="https://www.youtube.com/watch?v=abc",
            video_id="abc",
            title="Test",
            duration=100,
        )
        assert info.local_path is None

    def test_optional_local_path_set(self, tmp_path: Path) -> None:
        """Test local_path can be set to a Path."""
        video_file = tmp_path / "video.mp4"
        info = VideoInfo(
            url="https://www.youtube.com/watch?v=abc",
            video_id="abc",
            title="Test",
            duration=100,
            local_path=video_file,
        )
        assert info.local_path == video_file


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_create_basic(self) -> None:
        """Test creating a TranscriptSegment."""
        seg = TranscriptSegment(text="Hello, world!", start=10.5, duration=3.2)
        assert seg.text == "Hello, world!"
        assert seg.start == pytest.approx(10.5)
        assert seg.duration == pytest.approx(3.2)

    def test_end_property(self) -> None:
        """Test that end property = start + duration."""
        seg = TranscriptSegment(text="Test", start=10.0, duration=5.0)
        assert seg.end == pytest.approx(15.0)

    def test_format_timestamp_minutes(self) -> None:
        """Test timestamp formatting for LLM prompt (minutes only)."""
        seg = TranscriptSegment(
            text="Something funny happened",
            start=125.0,
            duration=3.0,
        )
        formatted = seg.format_timestamp()
        assert "Something funny happened" in formatted
        assert "2:05" in formatted or "02:05" in formatted

    def test_format_timestamp_hours(self) -> None:
        """Test timestamp formatting for LLM prompt (hours)."""
        seg = TranscriptSegment(text="Late game clip", start=3723.0, duration=2.0)
        formatted = seg.format_timestamp()
        assert "Late game clip" in formatted
        assert "1:02:03" in formatted


# ---------------------------------------------------------------------------
# Moment
# ---------------------------------------------------------------------------


class TestMoment:
    """Tests for Moment dataclass."""

    def test_create_basic(self) -> None:
        """Test creating a Moment."""
        moment = Moment(
            start=120.0,
            end=180.0,
            summary="Streamer falls off a cliff",
            clip_name="cliff_fall",
        )
        assert moment.start == pytest.approx(120.0)
        assert moment.end == pytest.approx(180.0)
        assert moment.summary == "Streamer falls off a cliff"
        assert moment.clip_name == "cliff_fall"

    def test_duration_property(self) -> None:
        """Test duration property."""
        moment = Moment(start=60.0, end=120.0, summary="Test", clip_name="test")
        assert moment.duration == pytest.approx(60.0)

    def test_youtube_url_default_none(self) -> None:
        """Test youtube_url defaults to None."""
        moment = Moment(start=0.0, end=10.0, summary="Test", clip_name="test")
        assert moment.youtube_url is None

    def test_youtube_url_with_video_id(self) -> None:
        """Test building YouTube timestamp URL."""
        moment = Moment(
            start=125.0,
            end=180.0,
            summary="Test",
            clip_name="test",
        )
        url = moment.build_youtube_url("dQw4w9WgXcQ")
        assert "dQw4w9WgXcQ" in url
        assert "t=125" in url

    def test_selected_default_true(self) -> None:
        """Test selected defaults to True."""
        moment = Moment(start=0.0, end=10.0, summary="Test", clip_name="test")
        assert moment.selected is True


# ---------------------------------------------------------------------------
# ClipConfig
# ---------------------------------------------------------------------------


class TestClipConfig:
    """Tests for ClipConfig dataclass."""

    def test_create_with_defaults(self, tmp_path: Path) -> None:
        """Test creating ClipConfig uses sensible defaults."""
        config = ClipConfig(output_dir=tmp_path)
        assert config.padding == 30
        assert config.output_dir == tmp_path

    def test_custom_padding(self, tmp_path: Path) -> None:
        """Test creating ClipConfig with custom padding."""
        config = ClipConfig(output_dir=tmp_path, padding=15)
        assert config.padding == 15

    def test_padding_must_be_non_negative(self, tmp_path: Path) -> None:
        """Test that negative padding raises ValueError."""
        with pytest.raises(ValueError, match="padding"):
            ClipConfig(output_dir=tmp_path, padding=-1)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_create_local(self, tmp_path: Path) -> None:
        """Test creating a local LLM config."""
        model_file = tmp_path / "llama.gguf"
        config = LLMConfig(
            backend=LLMBackend.LOCAL,
            model_path=model_file,
        )
        assert config.backend == LLMBackend.LOCAL
        assert config.model_path == model_file
        assert config.api_key is None
        assert config.model_name is None

    def test_create_openrouter(self) -> None:
        """Test creating an OpenRouter LLM config."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test-key",
            model_name="meta-llama/llama-3.1-8b-instruct",
        )
        assert config.backend == LLMBackend.OPENROUTER
        assert config.api_key == "sk-or-test-key"
        assert config.model_name == "meta-llama/llama-3.1-8b-instruct"
        assert config.model_path is None

    def test_local_without_model_path_raises(self) -> None:
        """Test that LOCAL backend requires model_path."""
        with pytest.raises(ValueError, match="model_path"):
            LLMConfig(backend=LLMBackend.LOCAL)

    def test_openrouter_without_api_key_raises(self) -> None:
        """Test that OPENROUTER backend requires api_key."""
        with pytest.raises(ValueError, match="api_key"):
            LLMConfig(backend=LLMBackend.OPENROUTER, model_name="test-model")

    def test_openrouter_without_model_name_raises(self) -> None:
        """Test that OPENROUTER backend requires model_name."""
        with pytest.raises(ValueError, match="model_name"):
            LLMConfig(backend=LLMBackend.OPENROUTER, api_key="sk-test")

    def test_llm_backend_enum_values(self) -> None:
        """Test LLMBackend enum has expected values."""
        assert LLMBackend.LOCAL is not None
        assert LLMBackend.OPENROUTER is not None
        assert LLMBackend.LOCAL != LLMBackend.OPENROUTER
