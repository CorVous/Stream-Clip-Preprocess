"""Theme and styling constants for the GUI."""

from __future__ import annotations

# Application appearance
APP_THEME = "dark"
APP_COLOR = "blue"

# Fonts
FONT_TITLE = ("Helvetica", 18, "bold")
FONT_LABEL = ("Helvetica", 13)
FONT_SMALL = ("Helvetica", 11)

# Section frame background colors: (light_mode, dark_mode)
SECTION_FG_COLOR_NORMAL: tuple[str, str] = ("gray86", "gray17")
SECTION_FG_COLOR_DISABLED: tuple[str, str] = ("gray78", "gray10")

# Label text colors for disabled/normal states
DISABLED_LABEL_COLOR = "gray50"
NORMAL_LABEL_COLOR: tuple[str, str] = ("gray10", "gray90")

# Colors
COLOR_SUCCESS = "#2ecc71"
COLOR_ERROR = "#e74c3c"
COLOR_WARNING = "#f39c12"

# Layout
PAD_X = 16
PAD_Y = 8
CORNER_RADIUS = 8
