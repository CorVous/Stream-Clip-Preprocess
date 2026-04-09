# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for stream-clip-preprocess."""

import sys
from pathlib import Path

import imageio_ffmpeg

# Locate the bundled ffmpeg binary from imageio-ffmpeg
_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

block_cipher = None

# ---------------------------------------------------------------------------
# Collect llama-cpp-python native libraries (optional)
# ---------------------------------------------------------------------------
# llama_cpp uses ctypes to load .dylib/.so/.dll files from a `lib/`
# subdirectory relative to its own __file__.  We collect every native
# library from that directory so PyInstaller places them at the correct
# path inside the bundle (llama_cpp/lib/*).
# If llama-cpp-python is not installed the build still succeeds — the app
# will fall back to the OpenRouter backend at runtime.
# ---------------------------------------------------------------------------

_llama_binaries: list[tuple[str, str]] = []
_llama_hiddenimports: list[str] = []

try:
    import llama_cpp as _llama_pkg  # noqa: E402

    _llama_dir = Path(_llama_pkg.__file__).parent
    _llama_lib_dir = _llama_dir / "lib"

    if _llama_lib_dir.is_dir():
        _native_suffixes = {".dylib", ".so", ".dll"}
        for _f in _llama_lib_dir.iterdir():
            if _f.suffix in _native_suffixes:
                _llama_binaries.append((str(_f), "llama_cpp/lib"))

    _llama_hiddenimports = [
        "llama_cpp",
        "llama_cpp.llama",
        "llama_cpp.llama_cpp",
        "llama_cpp.llama_chat_format",
        "llama_cpp.llama_grammar",
        "llama_cpp.llama_cache",
        "llama_cpp.llama_tokenizer",
        "llama_cpp.llama_speculative",
        "llama_cpp.llama_types",
        "llama_cpp._internals",
        "llama_cpp._ctypes_extensions",
        "llama_cpp._ggml",
        "llama_cpp._logger",
        "llama_cpp._utils",
    ]
except ImportError:
    pass

a = Analysis(
    ["src/stream_clip_preprocess/cli.py"],
    pathex=["src"],
    binaries=[
        (_ffmpeg_exe, "imageio_ffmpeg/binaries"),
        *_llama_binaries,
    ],
    datas=[],
    hiddenimports=[
        "customtkinter",
        "yt_dlp",
        "youtube_transcript_api",
        "imageio_ffmpeg",
        "httpx",
        *_llama_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="stream-clip-preprocess",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want a terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# macOS app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="Stream Clip Preprocess.app",
        icon=None,
        bundle_identifier="com.corvous.stream-clip-preprocess",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
