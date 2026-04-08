"""Shared test fixtures and configuration."""

import importlib.metadata

import pytest


@pytest.fixture
def package_name() -> str:
    """Return the package name (snake_case for imports)."""
    return __package__.split(".")[0] if __package__ else "stream_clip_preprocess"


@pytest.fixture
def cli_name() -> str:
    """Return the CLI command name (kebab-case)."""
    # Dynamically get from package metadata if possible
    try:
        return importlib.metadata.metadata("stream-clip-preprocess")["Name"]
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return "stream-clip-preprocess"
