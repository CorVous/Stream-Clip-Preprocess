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


def _find_scrollable_candidates(
    widget: tk.Misc | None,
    scrollables: list[ctk.CTkScrollableFrame],
) -> list[ctk.CTkScrollableFrame]:
    """Return ordered candidate list for scrolling, innermost first.

    Enables cascade: when the innermost frame cannot scroll (fully visible),
    callers iterate to the next candidate until one can actually scroll.

    **Pass 1** — ``check_if_master_is_canvas(widget)``: if the widget is a
    direct canvas child of ``scrollables[i]``, return ``scrollables[i:]``.

    **Pass 2** — Tk path-prefix: ``str(widget).startswith(str(sf) + ".")``.
    Catches widgets nested deeper than a direct canvas child (e.g. the
    internal ``Text`` widget inside a ``CTkTextbox``).

    :param widget: The Tk widget under the mouse pointer, or ``None``.
    :param scrollables: Candidate frames ordered **innermost first**.
    :return: Ordered slice of *scrollables* to try; empty when *widget*
        is outside every registered frame.
    """
    if widget is None:
        return []

    # Pass 1: direct canvas-child check (handles normal widgets)
    for i, sf in enumerate(scrollables):
        try:
            if sf.check_if_master_is_canvas(widget):
                return scrollables[i:]
        except (AttributeError, KeyError, RuntimeError):
            continue

    # Pass 2: Tk path-prefix check (catches CTkTextbox and other widgets
    # whose .master chain doesn't reach the canvas directly)
    try:
        widget_path = str(widget)
        for i, sf in enumerate(scrollables):
            try:
                if widget_path.startswith(str(sf) + "."):
                    return scrollables[i:]
            except Exception:
                _logger.debug("Error checking path prefix for %s", sf, exc_info=True)
                continue
    except Exception:
        _logger.debug("Error in scrollable ancestor check", exc_info=True)

    return []


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

    def _find_candidates() -> list[ctk.CTkScrollableFrame]:
        """Return scrollable candidates under the mouse pointer."""
        try:
            x = root.winfo_pointerx()
            y = root.winfo_pointery()
            widget = root.winfo_containing(x, y)
        except Exception:
            _logger.debug("Error getting widget under cursor", exc_info=True)
            return []
        return _find_scrollable_candidates(widget, scrollables)

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
                units = int(-delta_y / _MAC_PIXEL_DIVISOR) or (-1 if delta_y > 0 else 1)
                for sf in _find_candidates():
                    canvas = sf._parent_canvas  # noqa: SLF001
                    if canvas.yview() != (0.0, 1.0):
                        canvas.yview_scroll(units, "units")
                        break
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

    if sys.platform == "linux":

        def _on_button_4(event: tk.Event[tk.Misc]) -> None:
            for sf in _find_scrollable_candidates(event.widget, scrollables):
                canvas = sf._parent_canvas  # noqa: SLF001
                if canvas.yview() != (0.0, 1.0):
                    canvas.yview_scroll(-3, "units")
                    break

        def _on_button_5(event: tk.Event[tk.Misc]) -> None:
            for sf in _find_scrollable_candidates(event.widget, scrollables):
                canvas = sf._parent_canvas  # noqa: SLF001
                if canvas.yview() != (0.0, 1.0):
                    canvas.yview_scroll(3, "units")
                    break

        root.bind_all("<Button-4>", _on_button_4)
        root.bind_all("<Button-5>", _on_button_5)

    def _on_mousewheel(event: tk.Event[tk.Misc]) -> None:
        units = normalize_wheel_delta(event.delta)
        if units == 0:
            return
        for sf in _find_scrollable_candidates(event.widget, scrollables):
            canvas = sf._parent_canvas  # noqa: SLF001
            if canvas.yview() != (0.0, 1.0):
                canvas.yview_scroll(units, "units")
                break

    root.bind_all("<MouseWheel>", _on_mousewheel)
