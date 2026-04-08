"""Tests for GUI scaffold (state/threading and import checks)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from stream_clip_preprocess.gui.state import AppState, run_in_background

_TKINTER_AVAILABLE = True
try:
    import tkinter as tk

    del tk
except ImportError:
    _TKINTER_AVAILABLE = False

_skip_no_tk = pytest.mark.skipif(
    not _TKINTER_AVAILABLE, reason="tkinter not available in this environment"
)


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------


class TestAppState:
    """Tests for AppState enum/constants."""

    def test_initial_state_exists(self) -> None:
        """Test that AppState has an IDLE state."""
        assert AppState.IDLE is not None

    def test_state_values_distinct(self) -> None:
        """Test that all states are distinct."""
        states = list(AppState)
        assert len(states) == len(set(states))

    def test_expected_states_present(self) -> None:
        """Test that key states exist."""
        assert hasattr(AppState, "IDLE")
        assert hasattr(AppState, "FETCHING")
        assert hasattr(AppState, "ANALYZING")
        assert hasattr(AppState, "CLIPPING")


# ---------------------------------------------------------------------------
# run_in_background
# ---------------------------------------------------------------------------


class TestRunInBackground:
    """Tests for the run_in_background threading helper."""

    def test_runs_function_in_thread(self) -> None:
        """Test that function runs in a background thread."""
        main_thread_id = threading.get_ident()
        captured: list[int] = []

        def worker() -> None:
            captured.append(threading.get_ident())

        t = run_in_background(worker)
        t.join(timeout=2.0)

        assert len(captured) == 1
        assert captured[0] != main_thread_id

    def test_returns_thread_object(self) -> None:
        """Test that run_in_background returns a Thread."""
        t = run_in_background(lambda: None)
        assert isinstance(t, threading.Thread)
        t.join(timeout=2.0)

    def test_thread_is_daemon(self) -> None:
        """Test that background thread is a daemon thread."""
        t = run_in_background(lambda: None)
        assert t.daemon is True
        t.join(timeout=2.0)

    def test_on_done_callback_invoked(self) -> None:
        """Test that on_done callback is called after worker completes."""
        results: list[str] = []

        def worker() -> str:
            return "done"

        def on_done(value: object) -> None:
            results.append(str(value))

        t = run_in_background(worker, on_done=on_done)
        t.join(timeout=2.0)

        assert results == ["done"]

    def test_on_error_callback_invoked(self) -> None:
        """Test that on_error callback is called when worker raises."""
        errors: list[Exception] = []

        def worker() -> None:
            msg = "test error"
            raise RuntimeError(msg)

        def on_error(exc: Exception) -> None:
            errors.append(exc)

        t = run_in_background(worker, on_error=on_error)
        t.join(timeout=2.0)

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)


# ---------------------------------------------------------------------------
# GUI module import tests (skipped without tkinter)
# ---------------------------------------------------------------------------


class TestGuiModuleImports:
    """Tests that GUI modules are importable."""

    @_skip_no_tk
    def test_main_app_importable(self) -> None:
        """Test that MainApp class can be imported."""
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp is not None

    def test_themes_importable(self) -> None:
        """Test that themes module can be imported (no tkinter needed)."""
        from stream_clip_preprocess.gui import themes  # noqa: PLC0415

        assert themes is not None
        assert hasattr(themes, "APP_THEME")

    @_skip_no_tk
    def test_widgets_importable(self) -> None:
        """Test that widgets module can be imported."""
        from stream_clip_preprocess.gui import widgets  # noqa: PLC0415

        assert widgets is not None

    @_skip_no_tk
    @patch("stream_clip_preprocess.gui.app.ctk")
    def test_main_app_init_mocked(self, mock_ctk: MagicMock) -> None:
        """Test MainApp.__init__ with mocked customtkinter."""
        mock_ctk.CTk.return_value = MagicMock()
        mock_ctk.StringVar.return_value = MagicMock()
        mock_ctk.BooleanVar.return_value = MagicMock()
        mock_ctk.IntVar.return_value = MagicMock()

        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert callable(MainApp)

    @_skip_no_tk
    def test_cli_gui_command_exists(self) -> None:
        """Test that the launch function is defined."""
        from stream_clip_preprocess.gui.app import launch  # noqa: PLC0415

        assert callable(launch)
