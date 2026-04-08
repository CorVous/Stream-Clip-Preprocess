# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for stream-clip-preprocess."""

import sys
from pathlib import Path

import imageio_ffmpeg

# Locate the bundled ffmpeg binary from imageio-ffmpeg
_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
_ffmpeg_name = Path(_ffmpeg_exe).name

block_cipher = None

a = Analysis(
    ["src/stream_clip_preprocess/cli.py"],
    pathex=["src"],
    binaries=[(_ffmpeg_exe, "imageio_ffmpeg/binaries")],
    datas=[],
    hiddenimports=[
        "customtkinter",
        "yt_dlp",
        "youtube_transcript_api",
        "imageio_ffmpeg",
        "httpx",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["llama_cpp"],  # Optional dep; not bundled
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
