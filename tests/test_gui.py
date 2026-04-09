"""Tests for GUI scaffold (state/threading and import checks)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from stream_clip_preprocess.gui.scroll import normalize_wheel_delta
from stream_clip_preprocess.gui.state import (
    AppState,
    ThrottledCallback,
    run_in_background,
)

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

    @_skip_no_tk
    def test_openrouter_backend_importable(self) -> None:
        """Test that OpenRouterBackend is imported in app module."""
        from stream_clip_preprocess.gui import app as gui_app  # noqa: PLC0415

        assert hasattr(gui_app, "OpenRouterBackend")


# ---------------------------------------------------------------------------
# Mouse-wheel delta normalization
# ---------------------------------------------------------------------------


class TestNormalizeWheelDelta:
    """Tests for normalize_wheel_delta used by the scroll-wheel fix."""

    # -- macOS Tk 9+ (delta ≈ ±120 per notch) --

    def test_mac_tk9_scroll_up(self) -> None:
        """Positive delta on macOS Tk 9 → small negative scroll (up)."""
        units = normalize_wheel_delta(120, platform="darwin")
        assert units < 0
        assert abs(units) <= 6

    def test_mac_tk9_scroll_down(self) -> None:
        """Negative delta on macOS Tk 9 → small positive scroll (down)."""
        units = normalize_wheel_delta(-120, platform="darwin")
        assert units > 0
        assert abs(units) <= 6

    def test_mac_tk9_fast_scroll(self) -> None:
        """Larger delta (trackpad momentum) scales but stays bounded."""
        units = normalize_wheel_delta(360, platform="darwin")
        assert units < 0
        assert abs(units) <= 15

    # -- macOS Tk 8.6 (delta = ±1) --

    def test_mac_tk86_scroll_up(self) -> None:
        """Delta +1 (old Tk 8.6) → -1 scroll."""
        assert normalize_wheel_delta(1, platform="darwin") == -1

    def test_mac_tk86_scroll_down(self) -> None:
        """Delta -1 (old Tk 8.6) → +1 scroll."""
        assert normalize_wheel_delta(-1, platform="darwin") == 1

    # -- zero delta --

    def test_zero_delta_returns_zero(self) -> None:
        """Zero delta produces no scroll."""
        assert normalize_wheel_delta(0, platform="darwin") == 0
        assert normalize_wheel_delta(0, platform="win32") == 0

    # -- Windows (delta = ±120) --

    def test_win_scroll_up(self) -> None:
        """Windows positive delta → negative scroll (up)."""
        units = normalize_wheel_delta(120, platform="win32")
        assert units < 0
        assert abs(units) <= 6

    def test_win_scroll_down(self) -> None:
        """Windows negative delta → positive scroll (down)."""
        units = normalize_wheel_delta(-120, platform="win32")
        assert units > 0

    # -- Direction symmetry --

    def test_mac_opposite_deltas_symmetric(self) -> None:
        """Opposite deltas produce opposite scroll directions."""
        up = normalize_wheel_delta(120, platform="darwin")
        down = normalize_wheel_delta(-120, platform="darwin")
        assert up == -down


# ---------------------------------------------------------------------------
# _format_time
# ---------------------------------------------------------------------------


@_skip_no_tk
class TestFormatTime:
    """Tests for MainApp._format_time static method."""

    def test_format_time_seconds_only(self) -> None:
        """Values under a minute use 00:00:SS."""
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp._format_time(5) == "00:00:05"  # noqa: SLF001

    def test_format_time_minutes_and_seconds(self) -> None:
        """Values under an hour use 00:MM:SS."""
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp._format_time(125) == "00:02:05"  # noqa: SLF001

    def test_format_time_hours(self) -> None:
        """Values over an hour use HH:MM:SS."""
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp._format_time(3723) == "01:02:03"  # noqa: SLF001


# ---------------------------------------------------------------------------
# ThrottledCallback
# ---------------------------------------------------------------------------


class TestThrottledCallback:
    """Tests for ThrottledCallback used to debounce progress updates."""

    def test_first_call_passes_through(self) -> None:
        """First call should always be forwarded immediately."""
        calls: list[int] = []
        throttled = ThrottledCallback(calls.append, min_interval=1.0)
        throttled(42)
        assert calls == [42]

    def test_rapid_calls_are_suppressed(self) -> None:
        """Calls within the min_interval should be dropped."""
        calls: list[int] = []
        throttled = ThrottledCallback(calls.append, min_interval=0.5)
        throttled(1)
        throttled(2)
        throttled(3)
        # Only the first call should have been forwarded
        assert calls == [1]

    def test_call_after_interval_passes_through(self) -> None:
        """A call after the interval elapses should be forwarded."""
        calls: list[int] = []
        throttled = ThrottledCallback(calls.append, min_interval=0.05)
        throttled(1)
        time.sleep(0.08)
        throttled(2)
        assert calls == [1, 2]

    def test_preserves_latest_args(self) -> None:
        """When a call passes through, it uses its own arguments."""
        calls: list[tuple[int, str]] = []

        def record(a: int, b: str) -> None:
            calls.append((a, b))

        throttled = ThrottledCallback(record, min_interval=0.05)
        throttled(1, "a")
        time.sleep(0.08)
        throttled(2, "b")
        assert calls == [(1, "a"), (2, "b")]

    def test_kwargs_forwarded(self) -> None:
        """Keyword arguments should be forwarded to the wrapped callback."""
        calls: list[dict[str, object]] = []

        def record(**kw: object) -> None:
            calls.append(kw)

        throttled = ThrottledCallback(record, min_interval=1.0)
        throttled(x=10, y=20)
        assert calls == [{"x": 10, "y": 20}]

    def test_default_interval(self) -> None:
        """Default min_interval should be a reasonable value (≥100ms)."""
        throttled = ThrottledCallback(lambda: None)
        assert throttled._min_interval >= 0.1  # noqa: SLF001
