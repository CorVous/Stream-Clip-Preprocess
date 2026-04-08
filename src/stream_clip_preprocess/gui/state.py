"""Application state and threading helpers (no tkinter dependency)."""

from __future__ import annotations

import enum
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)


class AppState(enum.Enum):
    """States of the main application wizard."""

    IDLE = "idle"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    CLIPPING = "clipping"
    DONE = "done"


def run_in_background(
    fn: Callable[..., object],
    *args: object,
    on_done: Callable[[object], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    **kwargs: object,
) -> threading.Thread:
    """Run a callable in a daemon background thread.

    :param fn: Function to run in background
    :param args: Positional arguments for fn
    :param on_done: Optional callback with fn's return value
    :param on_error: Optional callback called if fn raises
    :param kwargs: Keyword arguments for fn
    :return: The started Thread
    """

    def _target() -> None:
        try:
            result = fn(*args, **kwargs)
            if on_done is not None:
                on_done(result)
        except Exception as exc:
            _logger.exception("Background task failed")
            if on_error is not None:
                on_error(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t
