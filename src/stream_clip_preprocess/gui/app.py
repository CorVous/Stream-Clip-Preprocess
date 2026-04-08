"""Main application window for stream-clip-preprocess."""

from __future__ import annotations

import logging
import webbrowser

import customtkinter as ctk  # type: ignore[import-untyped]

from stream_clip_preprocess.gui import themes
from stream_clip_preprocess.gui.state import AppState, run_in_background

# Re-export for convenience
__all__ = ["AppState", "MainApp", "launch", "run_in_background"]

_logger = logging.getLogger(__name__)


class MainApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        """Initialize the main window and all widgets."""
        super().__init__()
        ctk.set_appearance_mode(themes.APP_THEME)
        ctk.set_default_color_theme(themes.APP_COLOR)

        self.title("Stream Clip Preprocess")
        self.geometry("900x700")
        self.minsize(700, 550)

        self._state = AppState.IDLE
        self._video_info = None
        self._transcript_segments = None
        self._moments: list = []

        self._build_ui()

    def _build_ui(self) -> None:
        """Build all UI sections."""
        self._build_step1_input()
        self._build_step2_context()
        self._build_step3_moments()
        self._build_step4_export()
        self._update_section_states()

    def _build_step1_input(self) -> None:
        """Build Step 1: URL input."""
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(frame, text="Step 1: YouTube URL", font=themes.FONT_TITLE).pack(
            anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0)
        )

        row = ctk.CTkFrame(frame)
        row.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        self._url_var = ctk.StringVar()
        self._url_entry = ctk.CTkEntry(
            row,
            textvariable=self._url_var,
            placeholder_text="https://www.youtube.com/watch?v=...",
        )
        self._url_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._fetch_btn = ctk.CTkButton(row, text="Fetch", command=self._on_fetch)
        self._fetch_btn.pack(side="left")

        self._download_progress = ctk.CTkProgressBar(frame)
        self._download_progress.set(0)
        self._download_progress.pack(
            fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y)
        )

        self._download_status = ctk.CTkLabel(frame, text="")
        self._download_status.pack(anchor="w", padx=themes.PAD_X)

    def _build_step2_context(self) -> None:
        """Build Step 2: Context inputs."""
        self._step2_frame = ctk.CTkFrame(self)
        self._step2_frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(
            self._step2_frame, text="Step 2: Context", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        row1 = ctk.CTkFrame(self._step2_frame)
        row1.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(row1, text="Stream type:").pack(side="left")
        self._stream_type_var = ctk.StringVar(value="Gaming")
        self._stream_type = ctk.CTkOptionMenu(
            row1,
            variable=self._stream_type_var,
            values=["Gaming", "Just Chatting", "IRL", "Creative", "Other"],
        )
        self._stream_type.pack(side="left", padx=8)

        ctk.CTkLabel(row1, text="Game name:").pack(side="left", padx=(16, 0))
        self._game_name_var = ctk.StringVar()
        self._game_entry = ctk.CTkEntry(row1, textvariable=self._game_name_var)
        self._game_entry.pack(side="left", fill="x", expand=True, padx=8)

        ctk.CTkLabel(self._step2_frame, text="Clip prompt:").pack(
            anchor="w", padx=themes.PAD_X
        )
        self._prompt_text = ctk.CTkTextbox(self._step2_frame, height=80)
        default_prompt = (
            "Find the funniest, most exciting, or most notable moments. "
            "Focus on highlights, fails, funny moments, and memorable reactions."
        )
        self._prompt_text.insert("1.0", default_prompt)
        self._prompt_text.pack(fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

        self._analyze_btn = ctk.CTkButton(
            self._step2_frame, text="Find Moments", command=self._on_analyze
        )
        self._analyze_btn.pack(anchor="e", padx=themes.PAD_X, pady=themes.PAD_Y)

    def _build_step3_moments(self) -> None:
        """Build Step 3: Moments checklist."""
        self._step3_frame = ctk.CTkFrame(self)
        self._step3_frame.pack(
            fill="both", expand=True, padx=themes.PAD_X, pady=themes.PAD_Y
        )

        ctk.CTkLabel(
            self._step3_frame, text="Step 3: Moments", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        btn_row = ctk.CTkFrame(self._step3_frame)
        btn_row.pack(fill="x", padx=themes.PAD_X)

        self._select_all_btn = ctk.CTkButton(
            btn_row, text="Select All", width=100, command=self._on_select_all
        )
        self._select_all_btn.pack(side="left", padx=4)

        self._deselect_all_btn = ctk.CTkButton(
            btn_row, text="Deselect All", width=100, command=self._on_deselect_all
        )
        self._deselect_all_btn.pack(side="left", padx=4)

        self._moments_scroll = ctk.CTkScrollableFrame(self._step3_frame)
        self._moments_scroll.pack(
            fill="both", expand=True, padx=themes.PAD_X, pady=themes.PAD_Y
        )

    def _build_step4_export(self) -> None:
        """Build Step 4: Export settings."""
        self._step4_frame = ctk.CTkFrame(self)
        self._step4_frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(
            self._step4_frame, text="Step 4: Export", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        row = ctk.CTkFrame(self._step4_frame)
        row.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(row, text="Padding (seconds):").pack(side="left")
        self._padding_var = ctk.IntVar(value=30)
        self._padding_entry = ctk.CTkEntry(
            row, textvariable=self._padding_var, width=60
        )
        self._padding_entry.pack(side="left", padx=8)

        ctk.CTkLabel(row, text="Output folder:").pack(side="left", padx=(16, 0))
        self._output_dir_var = ctk.StringVar()
        self._output_dir_entry = ctk.CTkEntry(row, textvariable=self._output_dir_var)
        self._output_dir_entry.pack(side="left", fill="x", expand=True, padx=8)

        self._browse_btn = ctk.CTkButton(
            row, text="Browse...", width=80, command=self._on_browse
        )
        self._browse_btn.pack(side="left")

        self._clip_btn = ctk.CTkButton(
            self._step4_frame,
            text="Create Clips",
            command=self._on_create_clips,
        )
        self._clip_btn.pack(anchor="e", padx=themes.PAD_X, pady=themes.PAD_Y)

        self._clip_progress = ctk.CTkProgressBar(self._step4_frame)
        self._clip_progress.set(0)
        self._clip_progress.pack(fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

    def _update_section_states(self) -> None:
        """Enable/disable sections based on current state."""
        step2_state = (
            "normal"
            if self._state not in {AppState.IDLE, AppState.FETCHING}
            else "disabled"
        )
        self._step2_frame.configure(state=step2_state)

        step3_state = (
            "normal"
            if self._state in {AppState.CLIPPING, AppState.DONE}
            else "disabled"
        )
        self._step3_frame.configure(state=step3_state)
        self._step4_frame.configure(state=step3_state)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_fetch(self) -> None:
        """Handle Fetch button click."""
        _logger.debug("Fetch triggered for URL: %s", self._url_var.get())

    def _on_analyze(self) -> None:
        """Handle Find Moments button click."""
        _logger.debug("Analyze triggered")

    def _on_select_all(self) -> None:
        """Handle Select All button click."""
        _logger.debug("Select all triggered")

    def _on_deselect_all(self) -> None:
        """Handle Deselect All button click."""
        _logger.debug("Deselect all triggered")

    def _on_browse(self) -> None:
        """Handle Browse... button click."""
        _logger.debug("Browse triggered")

    def _on_create_clips(self) -> None:
        """Handle Create Clips button click."""
        _logger.debug("Create clips triggered")

    def _open_url(self, url: str) -> None:
        """Open a URL in the default browser.

        :param url: URL to open
        """
        webbrowser.open(url)


def launch() -> None:
    """Launch the main application window."""
    app = MainApp()
    app.mainloop()
