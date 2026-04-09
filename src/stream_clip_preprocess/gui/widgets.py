"""Custom reusable GUI widgets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)


class LabeledEntry(ctk.CTkFrame):
    """A labeled text entry widget."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        label: str,
        placeholder: str = "",
        **kwargs: object,
    ) -> None:
        """Initialize labeled entry.

        :param master: Parent widget
        :param label: Label text
        :param placeholder: Placeholder text for the entry
        """
        super().__init__(master, **kwargs)

        self._label = ctk.CTkLabel(self, text=label)
        self._label.pack(anchor="w", padx=4)

        self._entry = ctk.CTkEntry(self, placeholder_text=placeholder)
        self._entry.pack(fill="x", padx=4, pady=(0, 4))

    def get(self) -> str:
        """Return current entry value."""
        return self._entry.get()  # type: ignore[no-any-return]

    def set(self, value: str) -> None:
        """Set entry value.

        :param value: Text to set
        """
        self._entry.delete(0, "end")
        self._entry.insert(0, value)


class ProgressSection(ctk.CTkFrame):
    """A progress bar with status label."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        label: str = "Progress",
        **kwargs: object,
    ) -> None:
        """Initialize progress section.

        :param master: Parent widget
        :param label: Section label
        """
        super().__init__(master, **kwargs)

        self._label = ctk.CTkLabel(self, text=label)
        self._label.pack(anchor="w", padx=4)

        self._bar = ctk.CTkProgressBar(self)
        self._bar.set(0)
        self._bar.pack(fill="x", padx=4, pady=2)

        self._status = ctk.CTkLabel(self, text="")
        self._status.pack(anchor="w", padx=4)

    def set_progress(self, percent: float, status: str = "") -> None:
        """Update the progress bar and status label.

        :param percent: Progress value 0.0-1.0
        :param status: Status text to display
        """
        self._bar.set(percent)
        self._status.configure(text=status)


class MomentRow(ctk.CTkFrame):
    """A single moment row in the moments checklist.

    Uses a two-row layout: a top bar with checkbox, time range, and link
    button, followed by a wrapping summary label below.
    """

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        summary: str,
        time_range: str,
        youtube_url: str | None = None,
        on_link_click: Callable[[str], None] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize moment row.

        :param master: Parent widget
        :param summary: Moment summary text
        :param time_range: Formatted time range string
        :param youtube_url: YouTube URL for the timestamp link
        :param on_link_click: Callback when link is clicked
        """
        super().__init__(master, **kwargs)

        # Top bar: checkbox | time range | youtube button
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")

        self._var = ctk.BooleanVar(value=True)
        self._check = ctk.CTkCheckBox(top, text="", variable=self._var)
        self._check.pack(side="left", padx=(4, 0))

        self._time_label = ctk.CTkLabel(top, text=time_range)
        self._time_label.pack(side="left", padx=(2, 8))

        if youtube_url and on_link_click:
            self._link_btn = ctk.CTkButton(
                top,
                text="Open YouTube",
                width=100,
                command=lambda url=youtube_url: on_link_click(url),
            )
            self._link_btn.pack(side="right", padx=4)

        # Summary label: wraps to multiple lines
        self._summary_label = ctk.CTkLabel(
            self,
            text=summary,
            anchor="w",
            justify="left",
            wraplength=600,
        )
        self._summary_label.pack(fill="x", padx=(36, 8), pady=(0, 4))

    @property
    def selected(self) -> bool:
        """Return whether this moment is selected."""
        return bool(self._var.get())
