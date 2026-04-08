"""Tests for version management and version command."""

import re

from stream_clip_preprocess import __version__


def test_version_import() -> None:
    """Test that __version__ can be imported from package."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    # Should start with X.Y.Z (may have additional dev/git info)
    # Examples: "0.1.0", "0.1.1.dev0+g47aa3f55f.d20260327"

    # Match semantic version at start (X.Y.Z)
    pattern = r"^\d+\.\d+\.\d+"
    assert re.match(pattern, __version__), (
        f"Version {__version__} doesn't start with X.Y.Z"
    )


# NOTE: Subprocess-based version tests (test_version_subcommand, test_version_flag)
# were removed — they duplicated test_integration.py::test_cli_version_subcommand
# and test_integration.py::test_cli_version_flag.
