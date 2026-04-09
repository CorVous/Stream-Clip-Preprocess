"""Tests for GUI scaffold (state/threading and import checks)."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from stream_clip_preprocess.gui.scroll import normalize_wheel_delta
from stream_clip_preprocess.gui.state import (
    AppState,
    ThrottledCallback,
    run_in_background,
)
from stream_clip_preprocess.models import VideoInfo

if TYPE_CHECKING:
    from stream_clip_preprocess.gui.app import MainApp

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


# ---------------------------------------------------------------------------
# Section dim colors
# ---------------------------------------------------------------------------


class TestSectionDimColors:
    """Tests for section frame dimming color constants in themes."""

    def test_section_fg_normal_exists(self) -> None:
        """SECTION_FG_COLOR_NORMAL should be importable from themes."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_NORMAL,
        )

        assert SECTION_FG_COLOR_NORMAL is not None

    def test_section_fg_disabled_exists(self) -> None:
        """SECTION_FG_COLOR_DISABLED should be importable from themes."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_DISABLED,
        )

        assert SECTION_FG_COLOR_DISABLED is not None

    def test_section_fg_normal_is_two_tuple(self) -> None:
        """Normal section color should be a (light_mode, dark_mode) 2-tuple."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_NORMAL,
        )

        assert isinstance(SECTION_FG_COLOR_NORMAL, tuple)
        assert len(SECTION_FG_COLOR_NORMAL) == 2

    def test_section_fg_disabled_is_two_tuple(self) -> None:
        """Disabled section color should be a (light_mode, dark_mode) 2-tuple."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_DISABLED,
        )

        assert isinstance(SECTION_FG_COLOR_DISABLED, tuple)
        assert len(SECTION_FG_COLOR_DISABLED) == 2

    def test_disabled_darker_than_normal_dark_mode(self) -> None:
        """Disabled gray should be a lower number (darker) than normal in dark mode."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_DISABLED,
            SECTION_FG_COLOR_NORMAL,
        )

        normal_val = int(SECTION_FG_COLOR_NORMAL[1].replace("gray", ""))
        disabled_val = int(SECTION_FG_COLOR_DISABLED[1].replace("gray", ""))
        assert disabled_val < normal_val

    def test_disabled_darker_than_normal_light_mode(self) -> None:
        """Disabled gray should be a lower number (darker) than normal in light mode."""
        from stream_clip_preprocess.gui.themes import (  # noqa: PLC0415
            SECTION_FG_COLOR_DISABLED,
            SECTION_FG_COLOR_NORMAL,
        )

        normal_val = int(SECTION_FG_COLOR_NORMAL[0].replace("gray", ""))
        disabled_val = int(SECTION_FG_COLOR_DISABLED[0].replace("gray", ""))
        assert disabled_val < normal_val

    def test_label_color_constants_in_themes(self) -> None:
        """DISABLED_LABEL_COLOR and NORMAL_LABEL_COLOR should live in themes."""
        from stream_clip_preprocess.gui import themes  # noqa: PLC0415

        assert hasattr(themes, "DISABLED_LABEL_COLOR")
        assert hasattr(themes, "NORMAL_LABEL_COLOR")


# ---------------------------------------------------------------------------
# _section_fg_color helper
# ---------------------------------------------------------------------------


@_skip_no_tk
class TestSectionFgColor:
    """Tests for MainApp._section_fg_color static helper."""

    def test_normal_state_returns_normal_color(self) -> None:
        """'normal' state should return SECTION_FG_COLOR_NORMAL."""
        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp._section_fg_color("normal") == themes.SECTION_FG_COLOR_NORMAL  # noqa: SLF001

    def test_disabled_state_returns_disabled_color(self) -> None:
        """'disabled' state should return SECTION_FG_COLOR_DISABLED."""
        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        assert MainApp._section_fg_color("disabled") == themes.SECTION_FG_COLOR_DISABLED  # noqa: SLF001


# ---------------------------------------------------------------------------
# _set_frame_children_state — text dimming for all widget types
# ---------------------------------------------------------------------------


@_skip_no_tk
class TestFrameChildrenTextDimming:
    """Tests that _set_frame_children_state dims text on all widget types."""

    def test_non_label_child_text_color_dimmed_when_disabled(self) -> None:
        """Any child widget with text_color should be dimmed when disabled."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        import customtkinter as ctk  # noqa: PLC0415

        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        mock_child = MagicMock()
        mock_frame = MagicMock(spec=ctk.CTkFrame)
        mock_frame.winfo_children.return_value = [mock_child]

        MainApp._set_frame_children_state(mock_frame, "disabled")  # noqa: SLF001

        mock_child.configure.assert_any_call(text_color=themes.DISABLED_LABEL_COLOR)

    def test_non_label_child_text_color_restored_when_enabled(self) -> None:
        """Any child widget with text_color should be restored when enabled."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        import customtkinter as ctk  # noqa: PLC0415

        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        mock_child = MagicMock()
        mock_frame = MagicMock(spec=ctk.CTkFrame)
        mock_frame.winfo_children.return_value = [mock_child]

        MainApp._set_frame_children_state(mock_frame, "normal")  # noqa: SLF001

        mock_child.configure.assert_any_call(text_color=themes.NORMAL_LABEL_COLOR)

    def test_text_color_disabled_dimmed_when_section_disabled(self) -> None:
        """Widgets supporting text_color_disabled should also be dimmed."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        import customtkinter as ctk  # noqa: PLC0415

        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        mock_child = MagicMock()
        mock_frame = MagicMock(spec=ctk.CTkFrame)
        mock_frame.winfo_children.return_value = [mock_child]

        MainApp._set_frame_children_state(mock_frame, "disabled")  # noqa: SLF001

        mock_child.configure.assert_any_call(
            text_color_disabled=themes.DISABLED_LABEL_COLOR
        )

    def test_text_color_disabled_restored_when_section_enabled(self) -> None:
        """text_color_disabled should be restored to normal on enable."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        import customtkinter as ctk  # noqa: PLC0415

        from stream_clip_preprocess.gui import themes  # noqa: PLC0415
        from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

        mock_child = MagicMock()
        mock_frame = MagicMock(spec=ctk.CTkFrame)
        mock_frame.winfo_children.return_value = [mock_child]

        MainApp._set_frame_children_state(mock_frame, "normal")  # noqa: SLF001

        mock_child.configure.assert_any_call(
            text_color_disabled=themes.NORMAL_LABEL_COLOR
        )


# ---------------------------------------------------------------------------
# sync_game_field — game field synchronisation
# ---------------------------------------------------------------------------


@pytest.fixture
def app_for_game_sync() -> tuple[MainApp, MagicMock]:
    """Return a bare MainApp and a direct reference to its game-name mock.

    Uses object.__new__ to bypass __init__ (no tkinter needed) and vars() to
    wire up only the two attributes that sync_game_field touches.  The mock
    is returned directly so tests can assert on it without any private
    dotted-attribute access.
    """
    from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

    app: MainApp = object.__new__(MainApp)
    game_name_var: MagicMock = MagicMock()
    vars(app).update({
        "_video_info": None,
        "_game_name_var": game_name_var,
    })
    return app, game_name_var


class TestSyncGameField:
    """Tests for MainApp.sync_game_field."""

    def test_clears_field_when_video_has_no_game(
        self, app_for_game_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Field must be set to empty string when video_info.game is None.

        Without the fix the field was only written when game was truthy, so a
        stale value from a previous fetch would persist across re-fetches.
        """
        app, game_var = app_for_game_sync
        vars(app)["_video_info"] = VideoInfo(
            url="u", video_id="v", title="T", duration=120.0, game=None
        )
        app.sync_game_field()
        game_var.set.assert_called_once_with("")

    def test_sets_field_when_video_has_game(
        self, app_for_game_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Field is populated with the game name from metadata."""
        app, game_var = app_for_game_sync
        vars(app)["_video_info"] = VideoInfo(
            url="u", video_id="v", title="T", duration=120.0, game="Minecraft"
        )
        app.sync_game_field()
        game_var.set.assert_called_once_with("Minecraft")

    def test_no_op_when_no_video_fetched(
        self, app_for_game_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Field is left untouched before any video has been fetched."""
        app, game_var = app_for_game_sync
        app.sync_game_field()
        game_var.set.assert_not_called()


# ---------------------------------------------------------------------------
# sync_stream_type_field — stream type field synchronisation
# ---------------------------------------------------------------------------


@pytest.fixture
def app_for_stream_type_sync() -> tuple[MainApp, MagicMock]:
    """Return a bare MainApp and a direct reference to its stream-type mock.

    Uses object.__new__ to bypass __init__ (no tkinter needed) and vars() to
    wire up only the attributes that sync_stream_type_field touches.
    """
    from stream_clip_preprocess.gui.app import MainApp  # noqa: PLC0415

    app: MainApp = object.__new__(MainApp)
    stream_type_var: MagicMock = MagicMock()
    vars(app).update({
        "_video_info": None,
        "_stream_type_var": stream_type_var,
    })
    return app, stream_type_var


class TestSyncStreamTypeField:
    """Tests for MainApp.sync_stream_type_field."""

    def test_sets_field_from_first_category(
        self, app_for_stream_type_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Stream type is set to the first category from metadata."""
        app, stream_type_var = app_for_stream_type_sync
        vars(app)["_video_info"] = VideoInfo(
            url="u",
            video_id="v",
            title="T",
            duration=120.0,
            categories=["People & Blogs"],
        )
        app.sync_stream_type_field()
        stream_type_var.set.assert_called_once_with("People & Blogs")

    def test_defaults_to_gaming_when_no_categories(
        self, app_for_stream_type_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Stream type defaults to Gaming when the video has no categories."""
        app, stream_type_var = app_for_stream_type_sync
        vars(app)["_video_info"] = VideoInfo(
            url="u",
            video_id="v",
            title="T",
            duration=120.0,
            categories=[],
        )
        app.sync_stream_type_field()
        stream_type_var.set.assert_called_once_with("Gaming")

    def test_uses_first_when_multiple_categories(
        self, app_for_stream_type_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """First category wins when metadata has more than one."""
        app, stream_type_var = app_for_stream_type_sync
        vars(app)["_video_info"] = VideoInfo(
            url="u",
            video_id="v",
            title="T",
            duration=120.0,
            categories=["Entertainment", "Comedy"],
        )
        app.sync_stream_type_field()
        stream_type_var.set.assert_called_once_with("Entertainment")

    def test_no_op_when_no_video_fetched(
        self, app_for_stream_type_sync: tuple[MainApp, MagicMock]
    ) -> None:
        """Field is left untouched before any video has been fetched."""
        app, stream_type_var = app_for_stream_type_sync
        app.sync_stream_type_field()
        stream_type_var.set.assert_not_called()
