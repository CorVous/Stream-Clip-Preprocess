"""User settings persistence.

Settings are stored as a JSON file in a platform-appropriate config
directory (``~/.stream-clip-preprocess/settings.json``).  The file is
created on first save and read on startup.  A missing or corrupt file
silently falls back to defaults.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_logger = logging.getLogger(__name__)

_APP_DIR_NAME = ".stream-clip-preprocess"
_SETTINGS_FILENAME = "settings.json"

# Fields that are persisted.  Any key not in this set is ignored when
# reading (forward-compatible) and any missing key gets the dataclass default.
_KNOWN_FIELDS = frozenset({
    "backend",
    "model_path",
    "api_key",
    "openrouter_model",
    "output_dir",
})


@dataclasses.dataclass
class Settings:
    """Persisted user settings."""

    backend: str = "Local"
    model_path: str = ""
    api_key: str = ""
    openrouter_model: str = ""
    output_dir: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-friendly dict.

        :return: Dict with all settings fields
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Settings:
        """Create from a dict, ignoring unknown keys.

        Missing keys fall back to the dataclass defaults.

        :param data: Mapping (e.g. parsed from JSON)
        :return: Settings instance
        """
        filtered = {k: str(v) for k, v in data.items() if k in _KNOWN_FIELDS}
        return cls(**filtered)


def settings_path() -> Path:
    """Return the default settings file path.

    :return: ``~/.stream-clip-preprocess/settings.json``
    """
    return Path.home() / _APP_DIR_NAME / _SETTINGS_FILENAME


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from disk, returning defaults on any error.

    :param path: Override path (defaults to :func:`settings_path`)
    :return: Settings instance
    """
    target = path if path is not None else settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return Settings.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        _logger.debug("Could not load settings from %s: %s", target, exc)
        return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Save settings to disk as human-readable JSON.

    Creates parent directories if needed.

    :param settings: Settings to persist
    :param path: Override path (defaults to :func:`settings_path`)
    """
    target = path if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(settings.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
