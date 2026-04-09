"""Mouse-wheel scroll helpers for cross-platform Tk compatibility.

CustomTkinter's ``CTkScrollableFrame`` binds ``<MouseWheel>`` events
but assumes Tk 8.6 delta conventions.  **Tk 9.0 on macOS completely
dropped ``<MouseWheel>`` event generation** — scroll events are handled
natively by Cocoa views (like NSTextView/NSScrollView) and never reach
Tk's event system.  This makes every Canvas-backed scrollable widget
(including ``CTkScrollableFrame``) completely unable to scroll.

This module provides:

* ``normalize_wheel_delta`` — converts a raw ``event.delta`` into a
  reasonable ``yview_scroll`` unit count (for platforms where
  ``<MouseWheel>`` *does* fire, i.e. Windows / Linux / older macOS Tk).
* ``install_mousewheel_fix`` — the main entry point.  On macOS with
  Tk 9+ it installs an ``NSEvent`` local monitor via PyObjC that
  intercepts native scroll-wheel events and forwards them to the
  correct ``CTkScrollableFrame`` canvas.  On other platforms it patches
  the ``<MouseWheel>`` / ``<Button-4/5>`` bindings.
"""

from __future__ import annotations

import logging
import queue
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk

    import customtkinter as ctk

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Delta normalization (used on non-macOS-Tk9 platforms)
# ---------------------------------------------------------------------------

# The divisor turns the raw platform delta into line-like units.
# ±120 / 40 = ±3 lines, which is a comfortable default.
_DELTA_DIVISOR = 40


def normalize_wheel_delta(delta: int, *, platform: str | None = None) -> int:
    """Convert a raw ``<MouseWheel>`` delta to a ``yview_scroll`` unit count.

    :param delta: ``event.delta`` from a ``<MouseWheel>`` event.
    :param platform: Override ``sys.platform`` (for testing).
    :return: Integer suitable for ``canvas.yview_scroll(n, "units")``.
             Negative = scroll up, positive = scroll down.
    """
    if delta == 0:
        return 0

    plat = platform or sys.platform

    if plat == "darwin":
        # Tk 9+: delta ≈ ±120 per notch (like Windows).
        # Tk 8.6: delta = ±1.
        # Trackpad momentum can produce larger values (±240, ±360 …).
        if abs(delta) > 4:
            # Tk 9+ / momentum — scale down
            return int(-delta / _DELTA_DIVISOR) or (-1 if delta > 0 else 1)
        # Tk 8.6 legacy — just negate
        return -delta

    if plat.startswith("win"):
        # Windows always sends ±120 per notch.
        return int(-delta / _DELTA_DIVISOR) or (-1 if delta > 0 else 1)

    # Linux / fallback — <MouseWheel> is rarely used (Button-4/5 instead),
    # but just in case, treat the same as macOS.
    return -delta


# ---------------------------------------------------------------------------
# macOS Tk 9 native scroll monitor (PyObjC)
# ---------------------------------------------------------------------------

# Pixel-delta divisor for the NSEvent scrollingDeltaY value.
# Higher = slower scroll.  scrollingDeltaY reports pixel-precision deltas;
# dividing by 10 gives comfortable line-by-line scrolling.
_MAC_PIXEL_DIVISOR = 10


def _is_tk9() -> bool:
    """Return True if the running Tk version is 9.x or later."""
    try:
        import tkinter as tk  # noqa: PLC0415

        root = tk._default_root  # noqa: SLF001
        if root is None:
            return False
        version = root.tk.call("info", "patchlevel")
        major_version = 9
        return int(str(version).split(".")[0]) >= major_version
    except Exception:  # noqa: BLE001
        return False


def _install_macos_native_monitor(
    root: ctk.CTk,
    scrollables: list[ctk.CTkScrollableFrame],
) -> bool:
    """Install an NSEvent local monitor for scroll-wheel events.

    Returns True if the monitor was installed, False otherwise.
    """
    try:
        from AppKit import NSEvent  # type: ignore[import-not-found]  # noqa: PLC0415
        from Cocoa import (  # type: ignore[import-not-found]  # noqa: PLC0415
            NSScrollWheelMask,
        )
    except ImportError:
        _logger.warning(
            "pyobjc-framework-Cocoa not installed; "
            "scroll wheel will not work on macOS with Tk 9",
        )
        return False

    scroll_queue: queue.Queue[float] = queue.Queue()

    def _find_scrollable() -> ctk.CTkScrollableFrame | None:
        """Find the scrollable frame under the mouse pointer."""
        try:
            # Get the widget under the mouse via Tk
            x = root.winfo_pointerx()
            y = root.winfo_pointery()
            widget = root.winfo_containing(x, y)
            if widget is None:
                return None
            for sf in scrollables:
                try:
                    if sf.check_if_master_is_canvas(widget):
                        return sf
                except (AttributeError, KeyError, RuntimeError):
                    continue
        except Exception:
            _logger.debug("Error finding scrollable frame", exc_info=True)
        return None

    def _ns_event_handler(ns_event: object) -> object:
        """NSEvent callback — runs on the NSEvent thread.

        Only touches the thread-safe queue — no Tk calls.
        """
        try:
            delta_y = ns_event.scrollingDeltaY()  # type: ignore[union-attr]
            if delta_y:
                scroll_queue.put(delta_y)
        except Exception:
            _logger.debug("NSEvent callback error", exc_info=True)
        return ns_event

    def _poll_queue() -> None:
        """Drain the scroll queue on the Tk main thread."""
        try:
            while True:
                delta_y = scroll_queue.get_nowait()
                target = _find_scrollable()
                if target is None:
                    continue
                canvas = target._parent_canvas  # noqa: SLF001
                if canvas.yview() == (0.0, 1.0):
                    continue
                units = int(-delta_y / _MAC_PIXEL_DIVISOR) or (-1 if delta_y > 0 else 1)
                canvas.yview_scroll(units, "units")
        except queue.Empty:
            pass
        except Exception:
            _logger.debug("Scroll queue poll error", exc_info=True)
        root.after(8, _poll_queue)  # ~120 Hz

    monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        NSScrollWheelMask,
        _ns_event_handler,
    )

    if not monitor:
        _logger.warning("Failed to install NSEvent scroll monitor")
        return False

    def _cleanup() -> None:
        NSEvent.removeMonitor_(monitor)

    root.bind("<Destroy>", lambda _e: _cleanup(), add="+")

    _poll_queue()
    _logger.debug("macOS native scroll monitor installed")
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_mousewheel_fix(
    root: ctk.CTk,
    scrollables: list[ctk.CTkScrollableFrame],
) -> None:
    """Fix scroll-wheel support for all ``CTkScrollableFrame`` children.

    On **macOS with Tk 9+** this installs an ``NSEvent`` local monitor
    via PyObjC (requires ``pyobjc-framework-Cocoa``).

    On **Windows** and **Linux** (and macOS with Tk 8.6) this replaces
    the default ``<MouseWheel>`` / ``<Button-4/5>`` bindings with
    correctly-scaled versions.

    Must be called **after** all scrollable frames have been created.

    :param root: The top-level ``CTk`` window.
    :param scrollables: Scrollable frames to manage, listed **innermost
        first** so nested frames get priority.
    """
    # macOS Tk 9: use native NSEvent monitor
    if sys.platform == "darwin" and _is_tk9():
        if _install_macos_native_monitor(root, scrollables):
            return
        # Fall through to generic fix if monitor failed

    # --- Generic fix (Windows, Linux, macOS Tk 8) ---

    def _find_scrollable(widget: tk.Misc) -> ctk.CTkScrollableFrame | None:
        for sf in scrollables:
            try:
                if sf.check_if_master_is_canvas(widget):
                    return sf
            except (AttributeError, KeyError, RuntimeError):
                continue
        return None

    def _do_scroll(sf: ctk.CTkScrollableFrame, units: int) -> None:
        canvas = sf._parent_canvas  # noqa: SLF001
        if canvas.yview() != (0.0, 1.0):
            canvas.yview_scroll(units, "units")

    if sys.platform == "linux":

        def _on_button_4(event: tk.Event[tk.Misc]) -> None:
            target = _find_scrollable(event.widget)
            if target is not None:
                _do_scroll(target, -3)

        def _on_button_5(event: tk.Event[tk.Misc]) -> None:
            target = _find_scrollable(event.widget)
            if target is not None:
                _do_scroll(target, 3)

        root.bind_all("<Button-4>", _on_button_4)
        root.bind_all("<Button-5>", _on_button_5)

    def _on_mousewheel(event: tk.Event[tk.Misc]) -> None:
        target = _find_scrollable(event.widget)
        if target is None:
            return
        units = normalize_wheel_delta(event.delta)
        if units != 0:
            _do_scroll(target, units)

    root.bind_all("<MouseWheel>", _on_mousewheel)
