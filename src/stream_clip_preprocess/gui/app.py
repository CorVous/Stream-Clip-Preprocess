"""Main application window for stream-clip-preprocess."""

from __future__ import annotations

import contextlib
import logging
import tempfile
import webbrowser
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, cast

import customtkinter as ctk  # type: ignore[import-untyped]

from stream_clip_preprocess.clipper import ClipExtractor
from stream_clip_preprocess.downloader import DownloadError, VideoDownloader
from stream_clip_preprocess.gui import themes
from stream_clip_preprocess.gui.scroll import install_mousewheel_fix
from stream_clip_preprocess.gui.state import (
    AppState,
    ThrottledCallback,
    run_in_background,
)
from stream_clip_preprocess.gui.widgets import MomentRow
from stream_clip_preprocess.llm.base import LLMError
from stream_clip_preprocess.llm.local import LocalBackend
from stream_clip_preprocess.llm.openrouter import OpenRouterBackend
from stream_clip_preprocess.models import ClipConfig, LLMBackend, LLMConfig, Moment
from stream_clip_preprocess.settings import Settings, load_settings, save_settings
from stream_clip_preprocess.transcript import (
    NoTranscriptError,
    TranscriptFetcher,
    extract_video_id,
)

if TYPE_CHECKING:
    import threading

    from stream_clip_preprocess.clipper import ClipResult
    from stream_clip_preprocess.downloader import DownloadProgress
    from stream_clip_preprocess.models import (
        TranscriptSegment,
        VideoInfo,
    )

# Re-export for convenience
__all__ = ["AppState", "MainApp", "launch", "run_in_background"]

_logger = logging.getLogger(__name__)

# Color used to dim disabled section labels
_DISABLED_LABEL_COLOR = "gray50"
_NORMAL_LABEL_COLOR = ("gray10", "gray90")  # (light_mode, dark_mode)

# YouTube video categories (YouTube Data API v3)
_YOUTUBE_CATEGORIES: list[str] = [
    "Film & Animation",
    "Autos & Vehicles",
    "Music",
    "Pets & Animals",
    "Sports",
    "Travel & Events",
    "Gaming",
    "People & Blogs",
    "Comedy",
    "Entertainment",
    "News & Politics",
    "Howto & Style",
    "Education",
    "Science & Technology",
    "Nonprofits & Activism",
]


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
        self._video_info: VideoInfo | None = None
        self._transcript_segments: list[TranscriptSegment] | None = None
        self._moments: list[Moment] = []
        self._moment_rows: list[MomentRow] = []

        self._download_thread: threading.Thread | None = None

        self._settings_initialized = False
        self._build_ui()
        self._load_persisted_settings()
        self._settings_initialized = True

    def _build_ui(self) -> None:
        """Build all UI sections."""
        self._scroll = ctk.CTkScrollableFrame(self)
        self._scroll.pack(fill="both", expand=True)

        self._build_settings()
        self._build_step1_input()
        self._build_step2_context()
        self._build_step3_moments()
        self._build_step4_export()
        self._update_section_states()

        # Fix scroll-wheel for Tk 9+ on macOS and Button-4/5 on Linux.
        # Inner frame listed first so it takes priority when nested.
        install_mousewheel_fix(self, [self._moments_scroll, self._scroll])

    def _build_settings(self) -> None:
        """Build the settings section for LLM backend selection."""
        frame = ctk.CTkFrame(self._scroll)
        frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(frame, text="Settings", font=themes.FONT_TITLE).pack(
            anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0)
        )

        # Backend toggle row
        toggle_row = ctk.CTkFrame(frame)
        toggle_row.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(toggle_row, text="LLM backend:").pack(side="left")
        self._backend_var = ctk.StringVar(value="Local")
        self._backend_toggle = ctk.CTkSegmentedButton(
            toggle_row,
            values=["Local", "OpenRouter"],
            variable=self._backend_var,
            command=self._on_backend_changed,
        )
        self._backend_toggle.pack(side="left", padx=8)

        # --- Local backend row ---
        self._local_row = ctk.CTkFrame(frame)
        self._local_row.pack(fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

        ctk.CTkLabel(self._local_row, text="Model (.gguf):").pack(side="left")
        self._model_path_var = ctk.StringVar()
        self._model_path_entry = ctk.CTkEntry(
            self._local_row,
            textvariable=self._model_path_var,
            placeholder_text="/path/to/model.gguf",
        )
        self._model_path_entry.pack(side="left", fill="x", expand=True, padx=8)

        self._model_browse_btn = ctk.CTkButton(
            self._local_row,
            text="Browse...",
            width=80,
            command=self._on_model_browse,
        )
        self._model_browse_btn.pack(side="left")

        # --- OpenRouter backend rows ---
        self._openrouter_frame = ctk.CTkFrame(frame)
        # Hidden by default — shown when "OpenRouter" is selected
        self._openrouter_frame.pack(fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

        api_row = ctk.CTkFrame(self._openrouter_frame)
        api_row.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(api_row, text="API key:").pack(side="left")
        self._api_key_var = ctk.StringVar()
        self._api_key_entry = ctk.CTkEntry(
            api_row,
            textvariable=self._api_key_var,
            placeholder_text="sk-or-...",
            show="*",
        )
        self._api_key_entry.pack(side="left", fill="x", expand=True, padx=8)

        model_row = ctk.CTkFrame(self._openrouter_frame)
        model_row.pack(fill="x")

        ctk.CTkLabel(model_row, text="Model:").pack(side="left")
        self._or_model_var = ctk.StringVar()
        self._or_model_entry = ctk.CTkEntry(
            model_row,
            textvariable=self._or_model_var,
            placeholder_text="google/gemini-2.5-flash",
        )
        self._or_model_entry.pack(side="left", fill="x", expand=True, padx=8)

        # Start with correct visibility
        self._on_backend_changed("Local")

    def _load_persisted_settings(self) -> None:
        """Load settings from disk and populate the UI fields."""
        s = load_settings()
        self._backend_var.set(s.backend)
        self._model_path_var.set(s.model_path)
        self._api_key_var.set(s.api_key)
        self._or_model_var.set(s.openrouter_model)
        if s.output_dir:
            self._output_dir_var.set(s.output_dir)
        self._on_backend_changed(s.backend)

    def _save_current_settings(self) -> None:
        """Persist the current UI settings to disk."""
        s = Settings(
            backend=self._backend_var.get(),
            model_path=self._model_path_var.get(),
            api_key=self._api_key_var.get(),
            openrouter_model=self._or_model_var.get(),
            output_dir=self._output_dir_var.get(),
        )
        save_settings(s)

    def _on_backend_changed(self, value: str) -> None:
        """Show/hide settings rows based on backend selection.

        :param value: "Local" or "OpenRouter"
        """
        if self._settings_initialized:
            self._save_current_settings()

        if value == "Local":
            self._local_row.pack(
                fill="x",
                padx=themes.PAD_X,
                pady=(0, themes.PAD_Y),
                after=self._backend_toggle.master,
            )
            self._openrouter_frame.pack_forget()
        else:
            self._local_row.pack_forget()
            self._openrouter_frame.pack(
                fill="x",
                padx=themes.PAD_X,
                pady=(0, themes.PAD_Y),
                after=self._backend_toggle.master,
            )

    def _on_model_browse(self) -> None:
        """Open file dialog to select a GGUF model file."""
        path = filedialog.askopenfilename(
            title="Select GGUF Model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self._model_path_var.set(path)
            self._save_current_settings()

    def _build_step1_input(self) -> None:
        """Build Step 1: URL input."""
        frame = ctk.CTkFrame(self._scroll)
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
        self._step2_frame = ctk.CTkFrame(self._scroll)
        self._step2_frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(
            self._step2_frame, text="Step 2: Context", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        row1 = ctk.CTkFrame(self._step2_frame, fg_color="transparent")
        row1.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(row1, text="Stream type:").pack(side="left")
        self._stream_type_var = ctk.StringVar(value="Gaming")
        self._stream_type = ctk.CTkOptionMenu(
            row1,
            variable=self._stream_type_var,
            values=_YOUTUBE_CATEGORIES,
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

        self._analyze_progress = ctk.CTkProgressBar(self._step2_frame)
        self._analyze_progress.set(0)
        self._analyze_progress.pack(fill="x", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

        self._analyze_status = ctk.CTkLabel(self._step2_frame, text="")
        self._analyze_status.pack(anchor="w", padx=themes.PAD_X)

    def _build_step3_moments(self) -> None:
        """Build Step 3: Moments checklist."""
        self._step3_frame = ctk.CTkFrame(self._scroll)
        self._step3_frame.pack(
            fill="both", expand=True, padx=themes.PAD_X, pady=themes.PAD_Y
        )

        ctk.CTkLabel(
            self._step3_frame, text="Step 3: Moments", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        btn_row = ctk.CTkFrame(self._step3_frame, fg_color="transparent")
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
        self._step4_frame = ctk.CTkFrame(self._scroll)
        self._step4_frame.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(
            self._step4_frame, text="Step 4: Export", font=themes.FONT_TITLE
        ).pack(anchor="w", padx=themes.PAD_X, pady=(themes.PAD_Y, 0))

        row = ctk.CTkFrame(self._step4_frame, fg_color="transparent")
        row.pack(fill="x", padx=themes.PAD_X, pady=themes.PAD_Y)

        ctk.CTkLabel(row, text="Padding (seconds):").pack(side="left")
        self._padding_var = ctk.StringVar(value="30")
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

        self._clip_status = ctk.CTkLabel(self._step4_frame, text="")
        self._clip_status.pack(anchor="w", padx=themes.PAD_X, pady=(0, themes.PAD_Y))

    @staticmethod
    def _section_fg_color(state: str) -> tuple[str, str]:
        """Return the section frame fg_color for the given widget state.

        :param state: Widget state string, either ``"normal"`` or ``"disabled"``.
        :return: A ``(light_mode, dark_mode)`` color tuple for the frame.
        """
        if state == "normal":
            return themes.SECTION_FG_COLOR_NORMAL
        return themes.SECTION_FG_COLOR_DISABLED

    @classmethod
    def _set_frame_children_state(cls, frame: ctk.CTkFrame, state: str) -> None:
        """Recursively enable or disable all interactive children of a frame."""
        color: str | tuple[str, str] = (
            themes.DISABLED_LABEL_COLOR
            if state == "disabled"
            else themes.NORMAL_LABEL_COLOR
        )
        for child in frame.winfo_children():
            # Recurse into sub-frames so nested rows are reached
            if isinstance(child, ctk.CTkFrame):
                cls._set_frame_children_state(child, state)
            with contextlib.suppress(ValueError, TypeError):
                child.configure(state=state)
            # Dim text_color for all CTk widgets (labels, buttons, etc.).
            # Suppress Exception broadly: plain tkinter widgets inside
            # CTkScrollableFrame raise TclError for unknown options.
            with contextlib.suppress(Exception):
                child.configure(text_color=color)
            # Also dim text_color_disabled so entries/option-menus/checkboxes
            # show the dimmed color even while in the "disabled" widget state.
            # (tk.Text inside CTkTextbox uses fg for all states, so text_color
            # above is sufficient for textboxes — no separate handling needed.)
            with contextlib.suppress(Exception):
                child.configure(text_color_disabled=color)

    def _update_section_states(self) -> None:
        """Enable/disable sections based on current state."""
        step2_state = (
            "normal"
            if self._state not in {AppState.IDLE, AppState.FETCHING}
            else "disabled"
        )
        self._set_frame_children_state(self._step2_frame, step2_state)
        self._step2_frame.configure(fg_color=self._section_fg_color(step2_state))

        step3_state = (
            "normal"
            if self._state in {AppState.CLIPPING, AppState.DONE}
            else "disabled"
        )
        self._set_frame_children_state(self._step3_frame, step3_state)
        self._set_frame_children_state(self._step4_frame, step3_state)
        step3_fg = self._section_fg_color(step3_state)
        self._step3_frame.configure(fg_color=step3_fg)
        self._moments_scroll.configure(fg_color=step3_fg)
        self._step4_frame.configure(fg_color=self._section_fg_color(step3_state))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_fetch(self) -> None:
        """Handle Fetch button click."""
        url = self._url_var.get().strip()
        if not url:
            self._download_status.configure(
                text="Please enter a YouTube URL.", text_color=themes.COLOR_ERROR
            )
            return

        _logger.debug("Fetch triggered for URL: %s", url)
        self._state = AppState.FETCHING
        self._fetch_btn.configure(state="disabled")
        self._download_status.configure(
            text="Fetching video info...", text_color="white"
        )
        self._download_progress.set(0)

        def _do_fetch() -> tuple[VideoInfo, list[TranscriptSegment]]:
            downloader = VideoDownloader()
            info = downloader.get_info(url)

            video_id = extract_video_id(url)
            fetcher = TranscriptFetcher()
            segments = fetcher.fetch(video_id)
            return info, segments

        def _on_done(result: object) -> None:
            info, segments = cast("tuple[VideoInfo, list[TranscriptSegment]]", result)
            self._video_info = info
            self._transcript_segments = segments
            self._state = AppState.ANALYZING
            self.after(0, self._after_fetch_success)

        def _on_error(exc: Exception) -> None:
            self.after(0, lambda: self._after_fetch_error(exc))

        run_in_background(_do_fetch, on_done=_on_done, on_error=_on_error)

    def _update_fetch_download_progress(self, prog: DownloadProgress) -> None:
        """Update Step 1 progress bar during video download.

        :param prog: Download progress info
        """
        pct = min(prog.percent / 100.0, 1.0)
        self._download_progress.set(pct)
        self._download_status.configure(
            text=f"Downloading video... {prog.percent:.0f}%",
        )
        # Force Tk to process pending display updates so the progress bar
        # and status label visually repaint immediately.
        self.update_idletasks()

    def _after_fetch_success(self) -> None:
        """Update UI after successful fetch (called on main thread).

        Enables Step 2 immediately and starts the video download in the
        background so the user can configure context and run LLM analysis
        without waiting for the download to finish.
        """
        title = self._video_info.title if self._video_info else "Unknown"
        seg_count = len(self._transcript_segments or [])
        self._download_status.configure(
            text=f"{title} ({seg_count} segments) \u2014 downloading video\u2026",
            text_color="white",
        )
        self._download_progress.set(0)
        self._fetch_btn.configure(state="normal")

        self.sync_game_field()
        self.sync_stream_type_field()
        self._update_section_states()

        # Kick off video download concurrently
        self._start_background_download()

    def sync_game_field(self) -> None:
        """Sync the game name entry from the current video metadata.

        Sets the field to the game name when metadata is available, or clears
        it to an empty string when the fetched video has no game.  Does nothing
        if no video has been fetched yet.
        """
        if self._video_info is not None:
            self._game_name_var.set(self._video_info.game or "")

    def sync_stream_type_field(self) -> None:
        """Sync the stream type dropdown from the current video metadata.

        Sets the dropdown to the first YouTube category when metadata is
        available, or falls back to "Gaming" when no category is present.
        Does nothing if no video has been fetched yet.
        """
        if self._video_info is not None:
            cats = self._video_info.categories
            self._stream_type_var.set(cats[0] if cats else "Gaming")

    def _after_fetch_error(self, exc: Exception) -> None:
        """Update UI after failed fetch (called on main thread)."""
        self._state = AppState.IDLE
        msg = str(exc)
        if isinstance(exc, DownloadError | NoTranscriptError):
            msg = str(exc)
        self._download_status.configure(
            text=f"Error: {msg}", text_color=themes.COLOR_ERROR
        )
        self._fetch_btn.configure(state="normal")
        self._update_section_states()

    # ------------------------------------------------------------------
    # Background video download
    # ------------------------------------------------------------------

    def _start_background_download(self) -> None:
        """Start downloading the video in a background thread."""
        url = self._url_var.get().strip()
        game = self._video_info.game if self._video_info else None

        def _on_dl_progress_raw(prog: DownloadProgress) -> None:
            self.after(
                0,
                lambda p=prog: self._update_fetch_download_progress(p),
            )

        # yt-dlp fires progress hooks on every chunk (potentially hundreds
        # per second).  Throttle so the Tk event queue isn't flooded — this
        # lets the main-thread event loop process idle/repaint events between
        # updates, keeping the progress bar and status label visually current.
        on_dl_progress = ThrottledCallback(_on_dl_progress_raw, min_interval=0.2)

        def _do_download() -> VideoInfo:
            downloader = VideoDownloader()
            tmp_dir = Path(tempfile.mkdtemp(prefix="scp_"))
            return downloader.download(url, tmp_dir, on_progress=on_dl_progress)

        def _on_done(result: object) -> None:
            dl_info = cast("VideoInfo", result)
            if self._video_info:
                self._video_info.local_path = dl_info.local_path
                # Preserve game from initial metadata if download lost it
                if not self._video_info.game and dl_info.game:
                    self._video_info.game = dl_info.game
                elif not self._video_info.game and game:
                    self._video_info.game = game
            self.after(0, self._after_download_success)

        def _on_error(exc: Exception) -> None:
            self.after(0, lambda: self._after_download_error(exc))

        self._download_thread = run_in_background(
            _do_download, on_done=_on_done, on_error=_on_error
        )

    def _after_download_success(self) -> None:
        """Update UI after video download completes (called on main thread)."""
        self._download_progress.set(1.0)
        title = self._video_info.title if self._video_info else "Unknown"
        seg_count = len(self._transcript_segments or [])
        self._download_status.configure(
            text=f"Ready: {title} ({seg_count} segments) \u2014 video downloaded",
            text_color=themes.COLOR_SUCCESS,
        )

    def _after_download_error(self, exc: Exception) -> None:
        """Update UI after video download fails (called on main thread)."""
        self._download_progress.set(0)
        self._download_status.configure(
            text=f"Download failed: {exc}",
            text_color=themes.COLOR_ERROR,
        )

    def _on_analyze(self) -> None:
        """Handle Find Moments button click."""
        _logger.debug("Analyze triggered")

        if not self._transcript_segments:
            return

        is_local = self._backend_var.get() == "Local"

        # Validate settings before starting
        if is_local:
            model_path = self._model_path_var.get().strip()
            if not model_path:
                self._analyze_status.configure(
                    text="Select a .gguf model file in Settings above.",
                    text_color=themes.COLOR_ERROR,
                )
                return
        else:
            api_key = self._api_key_var.get().strip()
            or_model = self._or_model_var.get().strip()
            if not api_key:
                self._analyze_status.configure(
                    text="Enter an OpenRouter API key in Settings.",
                    text_color=themes.COLOR_ERROR,
                )
                return
            if not or_model:
                self._analyze_status.configure(
                    text="Enter an OpenRouter model name in Settings.",
                    text_color=themes.COLOR_ERROR,
                )
                return

        self._save_current_settings()

        self._analyze_btn.configure(state="disabled")
        self._analyze_progress.set(0)
        self._analyze_status.configure(
            text="Analyzing transcript...",
            text_color="white",
        )

        segments = self._transcript_segments
        stream_type = self._stream_type_var.get()
        game_name = self._game_name_var.get()
        clip_desc = self._prompt_text.get("1.0", "end-1c")

        def _on_chunk_progress(current: int, total: int, phase: str) -> None:
            self.after(
                0,
                lambda c=current, t=total, p=phase: self._update_analyze_progress(
                    c, t, p
                ),
            )

        def _do_analyze() -> list[Moment]:
            from pathlib import Path  # noqa: PLC0415

            if is_local:
                config = LLMConfig(
                    backend=LLMBackend.LOCAL,
                    model_path=Path(self._model_path_var.get().strip()),
                )
                backend = LocalBackend(config)
            else:
                config = LLMConfig(
                    backend=LLMBackend.OPENROUTER,
                    api_key=self._api_key_var.get().strip(),
                    model_name=self._or_model_var.get().strip(),
                )
                backend = OpenRouterBackend(config)

            return backend.analyze(
                segments,
                stream_type,
                game_name,
                clip_desc,
                on_progress=_on_chunk_progress,
            )

        def _on_done(result: object) -> None:
            moments = cast("list[Moment]", result)
            self.after(0, lambda: self._after_analyze_success(moments))

        def _on_error(exc: Exception) -> None:
            self.after(0, lambda: self._after_analyze_error(exc))

        run_in_background(_do_analyze, on_done=_on_done, on_error=_on_error)

    def _update_analyze_progress(
        self,
        current: int,
        total: int,
        phase: str,
    ) -> None:
        """Update the analysis progress bar (called on main thread).

        :param current: Current step number (1-based)
        :param total: Total steps in this phase
        :param phase: ``"describe"`` or ``"select"``
        """
        if phase == "describe":
            # Describe phase is 0-80% of the bar
            pct = (current / total) * 0.8
            self._analyze_status.configure(
                text=f"Describing moments: chunk {current}/{total}...",
            )
        else:
            # Select phase is 80-100%
            pct = 0.8 + (current / total) * 0.2
            self._analyze_status.configure(
                text="Selecting best moments...",
            )
        self._analyze_progress.set(pct)

    def _after_analyze_success(self, moments: list[Moment]) -> None:
        """Update UI after successful LLM analysis (called on main thread)."""
        self._moments = moments

        # Populate youtube URLs if we have video info
        if self._video_info:
            for m in self._moments:
                if not m.youtube_url:
                    m.youtube_url = m.build_youtube_url(self._video_info.video_id)

        self._state = AppState.CLIPPING
        self._analyze_progress.set(1.0)
        self._analyze_status.configure(
            text=f"Found {len(moments)} moment(s).",
            text_color=themes.COLOR_SUCCESS,
        )
        self._analyze_btn.configure(state="normal")
        self._display_moments()
        self._update_section_states()

    def _after_analyze_error(self, exc: Exception) -> None:
        """Update UI after failed LLM analysis (called on main thread)."""
        msg = str(exc) if isinstance(exc, LLMError) else f"Analysis failed: {exc}"
        self._analyze_status.configure(text=msg, text_color=themes.COLOR_ERROR)
        self._analyze_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # Moments display
    # ------------------------------------------------------------------

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS.

        :param seconds: Time in seconds
        :return: Formatted time string
        """
        total = int(seconds)
        mins, secs = divmod(total, 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _display_moments(self) -> None:
        """Populate the moments checklist from ``self._moments``."""
        # Clear existing rows
        for row in self._moment_rows:
            row.destroy()
        self._moment_rows.clear()

        for moment in self._moments:
            time_range = (
                f"{self._format_time(moment.start)} - {self._format_time(moment.end)}"
            )
            row = MomentRow(
                self._moments_scroll,
                summary=moment.summary,
                time_range=time_range,
                youtube_url=moment.youtube_url,
                on_link_click=self._open_url,
            )
            row.pack(fill="x", pady=2)
            self._moment_rows.append(row)

    def _on_select_all(self) -> None:
        """Handle Select All button click."""
        for row in self._moment_rows:
            row._var.set(True)  # noqa: SLF001

    def _on_deselect_all(self) -> None:
        """Handle Deselect All button click."""
        for row in self._moment_rows:
            row._var.set(False)  # noqa: SLF001

    def _on_browse(self) -> None:
        """Handle Browse... button click to select output directory."""
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            self._output_dir_var.set(directory)
            self._save_current_settings()

    def _on_create_clips(self) -> None:
        """Handle Create Clips button click."""
        output_dir = self._output_dir_var.get().strip()
        if not output_dir:
            self._clip_status.configure(
                text="Select an output folder first.",
                text_color=themes.COLOR_ERROR,
            )
            return

        # Sync selected state from UI rows to moment objects
        for moment, row in zip(self._moments, self._moment_rows, strict=False):
            moment.selected = row.selected

        selected = [m for m in self._moments if m.selected]
        if not selected:
            self._clip_status.configure(
                text="Select at least one moment to clip.",
                text_color=themes.COLOR_ERROR,
            )
            return

        if not self._video_info or not self._video_info.local_path:
            if self._download_thread and self._download_thread.is_alive():
                self._clip_status.configure(
                    text="Video still downloading, please wait.",
                    text_color=themes.COLOR_ERROR,
                )
            else:
                self._clip_status.configure(
                    text="Video not downloaded. Re-fetch the URL first.",
                    text_color=themes.COLOR_ERROR,
                )
            return

        self._clip_btn.configure(state="disabled")
        self._clip_progress.set(0)
        self._clip_status.configure(text="Creating clips...", text_color="white")

        video_path = self._video_info.local_path
        video_duration = self._video_info.duration
        video_title = self._video_info.title
        video_id = self._video_info.video_id
        output_base = Path(output_dir)
        try:
            padding = int(self._padding_var.get())
        except (ValueError, TypeError):
            padding = 30
        moments = self._moments

        def _do_clip() -> list[ClipResult]:
            from stream_clip_preprocess.sanitize import (  # noqa: PLC0415
                sanitize_filename,
            )

            # Create subfolder: {sanitized_title}_{video_id}
            safe_title = sanitize_filename(video_title, fallback="video")
            clip_dir = output_base / f"{safe_title}_{video_id}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            config = ClipConfig(output_dir=clip_dir, padding=padding)
            extractor = ClipExtractor()
            total_selected = len(selected)
            done_count = [0]

            def _on_clip_done(_result: ClipResult) -> None:
                done_count[0] += 1
                c, t = done_count[0], total_selected
                self.after(0, lambda: self._update_clip_progress(c, t))

            results = extractor.extract_all(
                moments=moments,
                video_path=video_path,
                config=config,
                video_duration=video_duration,
                on_clip_done=_on_clip_done,
            )

            # Delete the downloaded source video
            with contextlib.suppress(OSError):
                video_path.unlink()
                _logger.info("Deleted source video: %s", video_path)

            return results

        def _on_done(result: object) -> None:
            results = cast("list[ClipResult]", result)
            self.after(0, lambda: self._after_clips_done(results))

        def _on_error(exc: Exception) -> None:
            self.after(0, lambda: self._after_clips_error(exc))

        run_in_background(_do_clip, on_done=_on_done, on_error=_on_error)

    def _update_clip_progress(self, current: int, total: int) -> None:
        """Update progress bar during clip extraction (called on main thread).

        :param current: Number of clips extracted so far
        :param total: Total number of clips to extract
        """
        self._clip_progress.set(current / total)
        self._clip_status.configure(text=f"Extracting clip {current}/{total}...")

    def _after_clips_done(self, results: list[ClipResult]) -> None:
        """Update UI after clip extraction completes (called on main thread)."""
        self._clip_btn.configure(state="normal")
        self._clip_progress.set(1.0)

        ok = sum(1 for r in results if r.success)
        fail = len(results) - ok
        if fail:
            self._clip_status.configure(
                text=f"Done: {ok} clip(s) created, {fail} failed.",
                text_color=themes.COLOR_ERROR,
            )
        else:
            self._clip_status.configure(
                text=f"Done! {ok} clip(s) created.",
                text_color=themes.COLOR_SUCCESS,
            )

    def _after_clips_error(self, exc: Exception) -> None:
        """Update UI after clip extraction fails (called on main thread)."""
        self._clip_btn.configure(state="normal")
        msg = str(exc)
        if isinstance(exc, DownloadError):
            msg = f"Download failed: {exc}"
        self._clip_status.configure(text=msg, text_color=themes.COLOR_ERROR)

    def _open_url(self, url: str) -> None:
        """Open a URL in the default browser.

        :param url: URL to open
        """
        webbrowser.open(url)


def launch() -> None:
    """Launch the main application window."""
    app = MainApp()
    app.mainloop()
