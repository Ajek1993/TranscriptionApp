# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TranscriptionApp (v5.0.2) — Python platform for transcribing video/audio (YouTube or local files) into subtitles, with optional machine translation and TTS dubbing. Interface language is Polish.

## Commands

### Run GUI
```bash
python gui.py
# Opens http://127.0.0.1:7860
```

### Run CLI
```bash
python transcribe.py "https://www.youtube.com/watch?v=..."
python transcribe.py "URL" --language en --translate en-pl
python transcribe.py "URL" --dub --tts-voice pl-PL-ZofiaNeural
python transcribe.py --local "video.mp4"
```

### Setup (Windows native)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-windows.txt
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Docker
```bash
docker-compose build
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=..."
```

There is no test suite.

## Architecture

Two entry points: `transcribe.py` (CLI) and `gui.py` (Gradio web UI). All business logic lives in `data/` as 15 specialized modules. GUI components are in `data/gui/`.

### Processing Pipeline (5 stages)

1. **Validation** — `validators.py` checks URLs/files/dependencies; `device_manager.py` detects GPU/CPU
2. **Source Acquisition** — `youtube_processor.py` downloads audio/video via yt-dlp, or extracts audio from local files
3. **Transcription** — `transcription_engines.py` dispatches to WhisperX (default, with speaker diarization) or OpenAI Whisper. Returns `List[Tuple[start_ms, end_ms, text]]`
4. **Post-processing** — `segment_processor.py` splits/merges segments; `translation.py` translates via deep-translator; `srt_writer.py`/`ass_writer.py` generate subtitle files
5. **TTS Dubbing (optional)** — `tts_generator.py` generates speech (Edge TTS cloud or Coqui TTS local); `audio_mixer.py` mixes audio and burns subtitles into video via FFmpeg

### Key Patterns

- **Dispatcher pattern** in `transcription_engines.py` — `transcribe_chunk()` selects engine
- **Command builder pattern** in `command_builders.py` — constructs FFmpeg/yt-dlp commands as lists
- **Singleton-like** `OutputManager` for centralized console output formatting
- `device_manager.py` handles CUDA detection, VRAM monitoring, and cache clearing

### GUI Structure

`gui.py` creates Gradio interface with 3 tabs (Transcription, Dubbing/Subtitles, Download). Configuration constants (models, voices, languages) are in `data/gui/config.py`. Event handlers are in `data/gui/handlers.py`.

## Key Dependencies

- **yt-dlp** (YouTube download), **openai-whisper** + **whisperx** (STT), **deep-translator** (translation)
- **edge-tts** (cloud TTS), **TTS** (Coqui local TTS), **gradio** (web UI)
- **torch 2.3.1 + CUDA 12.1** (GPU acceleration with automatic CPU fallback)
- **FFmpeg** (required system dependency for all audio/video operations)
- `transformers==4.48.0` is pinned for WhisperX compatibility — do not upgrade without testing

## Output

Generated files go to `files/` directory. Temporary files are auto-cleaned after processing (including Gradio temp files).
