# Stream Clip Preprocess — Project Plan

## Overview

A portable desktop application (exe/dmg) that takes a YouTube stream URL, uses an LLM to identify funny/notable moments from the transcript, and creates clips with configurable padding using ffmpeg.

## User Flow

1. Paste a YouTube URL → click "Fetch"
2. App immediately starts video download (background) and fetches transcript (instant)
3. User fills in: stream type, game name (if any), clip prompt (pre-filled, editable)
4. Click "Find Moments" → LLM analyzes transcript, returns timestamped moments
5. User reviews moments as a checklist — each has a summary, duration, and clickable YouTube timestamped link
6. User selects which moments to clip, sets padding (default 30s)
7. Click "Create Clips" → ffmpeg cuts each clip, progress bar shown per clip
8. Done → opens output folder

If the video download isn't finished when the user clicks "Create Clips," the app shows progress and waits.

## Technology Decisions

| Area | Choice | Rationale |
|------|--------|-----------|
| GUI | CustomTkinter | Lightweight, modern look, simple API, small bundle |
| Packaging | PyInstaller | Battle-tested, supports Windows (.exe) and macOS (.app/.dmg) |
| Video download | yt-dlp | The standard — full video download with progress callbacks |
| Transcript | youtube-transcript-api | Instant structured data, no file I/O or parsing needed |
| LLM (local, default) | llama-cpp-python | Bundled in-app, user provides a GGUF model file |
| LLM (cloud) | OpenRouter | API key + model selection, access to many models |
| FFmpeg | imageio-ffmpeg | Bundled ffmpeg binary, no user install needed |
| Clip strategy | ffmpeg stream copy (`-c copy`) | Near-instant, no re-encoding; keyframe imprecision is irrelevant with 30s padding |

## Architecture

```
src/stream_clip_preprocess/
├── cli.py              # Entry point
├── gui/
│   ├── app.py          # Main window
│   ├── widgets.py      # Custom UI components
│   └── themes.py       # Styling
├── downloader.py       # yt-dlp video download
├── transcript.py       # Fetch & parse YouTube transcripts
├── llm/
│   ├── base.py         # Abstract LLM interface
│   ├── local.py        # llama-cpp-python backend
│   └── openrouter.py   # OpenRouter backend
├── clipper.py          # ffmpeg clip extraction
└── models.py           # Shared data models (Moment, ClipConfig, VideoInfo, etc.)
```

## GUI Layout

Single window, wizard-style top-to-bottom flow. Steps 2–4 are disabled until the previous step completes.

### Step 1: Input

- YouTube URL text field + "Fetch" button
- Download progress bar with percentage, speed, and ETA

### Step 2: Context (appears after transcript fetched)

- Stream type dropdown
- Game name text field
- Clip prompt — pre-filled, multi-line, editable
- "Find Moments" button

### Step 3: Moments (appears after LLM analysis)

- Checklist of moments, each with:
  - Checkbox (checked by default)
  - Clip name / summary
  - Time range
  - Link icon → opens YouTube at that timestamp in browser
- Select All / Deselect All buttons

### Step 4: Export

- Padding input (default 30 seconds)
- Output folder selector with browse button (default: `~/StreamClips/{video_title}/`)
- "Create Clips" button
- Per-clip progress bar

### Settings (separate dialog, gear icon)

- LLM backend toggle: Local / OpenRouter
- Local model: dropdown of recommended models (with download button) + "Browse..." for local GGUF files
- OpenRouter: API key field + model dropdown/text field

## LLM Prompt Strategy

- **System prompt**: Sets the role — analyzing a stream transcript to find funny/notable moments
- **Includes**: Stream type, game being played, what kind of clips the user wants
- **User prompt**: Full timestamped transcript
- **Output format**: JSON array of moments, each with: start time, end time, summary, suggested clip name
- **Long transcripts**: Chunk into overlapping windows for models with small context, or send all at once for large-context models
- Prompt is pre-filled but fully editable by the user
- Moments should be up to 3 minutes long, shorter preferred

## Model Selection

- **Dropdown** listing popular/recommended GGUF models with "Download" button for each
- **"Browse..." button** to select a GGUF file already on disk
- Selected model is remembered across sessions

## Clip Extraction Details

- Uses `ffmpeg -ss {start - padding} -to {end + padding} -i video.mp4 -c copy output.mp4`
- Padding clamped to video bounds: `max(0, start - padding)` and `min(duration, end + padding)`
- File naming: `{sanitized_clip_name}_{start_time}-{end_time}.mp4`
- All clips deposited into user-specified output folder

## Build Phases (TDD)

Each phase follows red/green TDD: write a failing test first, then the minimum code to pass, then refactor.

### Phase 1: Project Setup

- Rename template from `python_template` to `stream_clip_preprocess`
- Update `pyproject.toml`: name, scripts entry, dependencies
- Add dependencies: `customtkinter`, `yt-dlp`, `youtube-transcript-api`, `llama-cpp-python`, `imageio-ffmpeg`
- Verify `uv sync --dev`, `uv run ruff check`, `uv run pytest` all pass

### Phase 2: Data Models

- `VideoInfo`: url, video_id, title, duration
- `TranscriptSegment`: text, start, duration
- `Moment`: start, end, summary, clip_name, youtube_url
- `ClipConfig`: padding, output_dir
- `LLMConfig`: backend (local/openrouter), model_path, api_key, model_name

### Phase 3: Transcript Fetcher

- Fetch transcript by video ID using `youtube-transcript-api`
- Return list of `TranscriptSegment`
- Handle errors: no transcript available, video not found
- Format transcript with timestamps for LLM consumption

### Phase 4: Video Downloader

- Download video by URL using `yt-dlp`
- Progress callback for UI updates (percentage, speed, ETA)
- Run in background thread
- Extract video metadata (title, duration)

### Phase 5: LLM Analyzer

- Abstract `LLMBackend` interface with `analyze(transcript, prompt, config) -> list[Moment]`
- `LocalBackend`: llama-cpp-python, loads GGUF model, runs inference
- `OpenRouterBackend`: HTTP API calls
- Prompt construction from user inputs (stream type, game, clip preferences)
- JSON response parsing and validation
- Transcript chunking for long streams

### Phase 6: Clip Extractor

- Find ffmpeg binary via `imageio-ffmpeg`
- Cut clips with `subprocess` + ffmpeg stream copy
- Clamp padding to video bounds
- Sanitize file names
- Progress reporting per clip

### Phase 7: GUI

- Main window with CustomTkinter
- Wire up all components from phases 2–6
- Background threading for downloads and LLM calls
- Settings dialog for LLM configuration
- Error handling and user feedback throughout

### Phase 8: Packaging

- PyInstaller `.spec` files for Windows and macOS
- GitHub Actions CI: matrix build (Windows + macOS ARM)
- Test that built executables work end-to-end
- Create `.dmg` wrapper for macOS (via `create-dmg` or similar)
