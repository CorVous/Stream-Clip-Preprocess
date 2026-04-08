"""Integration tests for CLI functionality."""

import re
import subprocess  # noqa: S404
import sys

import pytest

import stream_clip_preprocess

_PACKAGE_NAME: str = stream_clip_preprocess.__package__ or "stream_clip_preprocess"


def test_cli_help() -> None:
    """Test that CLI help works."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_version_flag() -> None:
    """Test --version flag."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert re.search(r"\d+\.\d+\.\d+", result.stdout), "Version not found in output"


def test_cli_version_subcommand() -> None:
    """Test version subcommand."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert re.search(r"\d+\.\d+\.\d+", result.stdout), "Version not found in output"


def test_cli_verbose_flag() -> None:
    """Test that -v flag enables INFO logging."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "-v", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "INFO:" in result.stderr or "INFO:" in result.stdout


def test_cli_very_verbose_flag() -> None:
    """Test that -vv flag enables DEBUG logging."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "-vv", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "DEBUG:" in result.stderr or "DEBUG:" in result.stdout


def test_cli_no_subcommand() -> None:
    """Test that running without subcommand shows help."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_invalid_subcommand() -> None:
    """Test that invalid subcommand shows error."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "invalid"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid" in result.stderr.lower()


def test_version_subcommand_help() -> None:
    """Test that version subcommand has help."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "version", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "version" in result.stdout.lower()


def test_library_import_no_cli_execution() -> None:
    """Test that importing package doesn't execute CLI code."""
    code = """
import sys
import io

old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

from stream_clip_preprocess import __version__

stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

sys.stdout = old_stdout
sys.stderr = old_stderr

assert not stdout_value, f"Unexpected stdout: {stdout_value}"
assert not stderr_value, f"Unexpected stderr: {stderr_value}"
assert __version__ and isinstance(__version__, str)
print("SUCCESS")
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "SUCCESS" in result.stdout


@pytest.mark.skip(reason="Example command not registered - template only")
def test_example_subcommand() -> None:
    """Test example subcommand (when registered)."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "example", "World"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Hello, World!" in result.stdout


@pytest.mark.skip(reason="Example command not registered - template only")
def test_example_subcommand_with_greeting() -> None:
    """Test example subcommand with custom greeting."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            _PACKAGE_NAME,
            "example",
            "World",
            "--greeting",
            "Hi",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Hi, World!" in result.stdout
