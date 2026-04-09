"""Tests for user settings persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from stream_clip_preprocess.settings import (
    Settings,
    load_settings,
    save_settings,
    settings_path,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSettingsPath:
    """Tests for settings file path resolution."""

    def test_returns_path_object(self) -> None:
        """Test that settings_path returns a Path."""
        from pathlib import Path  # noqa: PLC0415

        assert isinstance(settings_path(), Path)

    def test_filename_is_settings_json(self) -> None:
        """Test that the file is named settings.json."""
        assert settings_path().name == "settings.json"


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_defaults(self) -> None:
        """Test that Settings has sensible defaults."""
        s = Settings()
        assert s.backend == "Local"
        assert not s.model_path
        assert not s.api_key
        assert not s.openrouter_model
        assert not s.output_dir

    def test_to_dict(self) -> None:
        """Test that Settings serializes to a dict."""
        s = Settings(backend="OpenRouter", api_key="sk-test")
        d = s.to_dict()
        assert d["backend"] == "OpenRouter"
        assert d["api_key"] == "sk-test"

    def test_from_dict(self) -> None:
        """Test that Settings can be created from a dict."""
        d = {
            "backend": "OpenRouter",
            "model_path": "/some/model.gguf",
            "api_key": "sk-123",
            "openrouter_model": "meta-llama/llama-3",
            "output_dir": "/home/user/clips",
        }
        s = Settings.from_dict(d)
        assert s.backend == "OpenRouter"
        assert s.model_path == "/some/model.gguf"
        assert s.api_key == "sk-123"
        assert s.openrouter_model == "meta-llama/llama-3"
        assert s.output_dir == "/home/user/clips"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test that unknown keys in the dict are ignored."""
        d = {"backend": "Local", "unknown_key": "value"}
        s = Settings.from_dict(d)
        assert s.backend == "Local"

    def test_from_dict_uses_defaults_for_missing(self) -> None:
        """Test that missing keys fall back to defaults."""
        s = Settings.from_dict({"backend": "OpenRouter"})
        assert not s.model_path
        assert not s.api_key


class TestLoadSaveSettings:
    """Tests for loading and saving settings to disk."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test that save_settings creates the JSON file."""
        path = tmp_path / "settings.json"
        s = Settings(backend="Local", model_path="/foo.gguf")
        save_settings(s, path=path)
        assert path.exists()

    def test_load_reads_saved(self, tmp_path: Path) -> None:
        """Test that load_settings reads back what was saved."""
        path = tmp_path / "settings.json"
        original = Settings(
            backend="OpenRouter",
            api_key="sk-test",
            openrouter_model="some-model",
            output_dir="/home/user/clips",
        )
        save_settings(original, path=path)
        loaded = load_settings(path=path)
        assert loaded.backend == "OpenRouter"
        assert loaded.api_key == "sk-test"
        assert loaded.openrouter_model == "some-model"
        assert loaded.output_dir == "/home/user/clips"

    def test_load_missing_file_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that a missing file returns default Settings."""
        path = tmp_path / "nonexistent.json"
        s = load_settings(path=path)
        assert s.backend == "Local"
        assert not s.model_path

    def test_load_corrupt_file_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that a corrupt file returns default Settings."""
        path = tmp_path / "settings.json"
        path.write_text("not json {{{")
        s = load_settings(path=path)
        assert s.backend == "Local"

    def test_save_is_human_readable(self, tmp_path: Path) -> None:
        """Test that the saved JSON is indented for readability."""
        path = tmp_path / "settings.json"
        save_settings(Settings(), path=path)
        text = path.read_text()
        # Indented JSON has newlines
        assert "\n" in text
        parsed = json.loads(text)
        assert "backend" in parsed
