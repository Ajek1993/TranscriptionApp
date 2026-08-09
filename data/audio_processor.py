#!/usr/bin/env python3
"""
Audio processing utilities.

This module provides functions for:
- Getting audio duration
- Splitting audio files into chunks
"""

import json
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

from .command_builders import (
    build_ffprobe_duration_cmd,
    build_ffmpeg_audio_split_cmd,
    build_ffprobe_audio_streams_cmd
)


def list_audio_streams(file_path: str) -> List[Dict]:
    """
    List all audio streams of a media file with their language tags.

    Movie rips routinely carry several audio tracks (original + dubbed/voiceover
    versions). Without inspecting them, ffmpeg's default stream selection picks
    whichever track is flagged `default` - and failing that, the one with the
    most channels. On a typical rip both rules point at the dub, so the
    transcription silently happens on the wrong language.

    Args:
        file_path: Path to the media file

    Returns:
        List of dicts with keys: track (audio-relative index used by `-map 0:a:N`),
        codec, channels, language, title. Empty list if probing fails.
    """
    try:
        cmd = build_ffprobe_audio_streams_cmd(file_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []

    streams = []
    for track_idx, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags", {}) or {}
        streams.append({
            "track": track_idx,
            "codec": stream.get("codec_name", "?"),
            "channels": stream.get("channels", 0) or 0,
            "language": (tags.get("language") or "").lower(),
            "title": tags.get("title", "") or ""
        })

    return streams


def format_audio_stream(stream: Dict) -> str:
    """Format a single audio stream description for display."""
    parts = [f"#{stream['track']}"]

    if stream["language"]:
        parts.append(stream["language"])
    else:
        parts.append("brak tagu języka")

    if stream["channels"]:
        parts.append(f"{stream['channels']} kan.")
    if stream["codec"] != "?":
        parts.append(stream["codec"])
    if stream["title"]:
        parts.append(f'"{stream["title"]}"')

    return " | ".join(parts)


# ISO 639-1 (used across the app) -> ISO 639-2/B codes that ffmpeg writes into
# the `language` tag. Only languages offered in the GUI need an entry.
_LANG_ALIASES = {
    "pl": {"pl", "pol"},
    "en": {"en", "eng"},
    "es": {"es", "spa"},
    "fr": {"fr", "fra", "fre"},
    "de": {"de", "deu", "ger"},
    "it": {"it", "ita"},
    "pt": {"pt", "por"},
    "ru": {"ru", "rus"},
    "tr": {"tr", "tur"},
    "nl": {"nl", "nld", "dut"},
    "cs": {"cs", "ces", "cze"},
    "ar": {"ar", "ara"},
    "zh-cn": {"zh", "zho", "chi"},
    "ja": {"ja", "jpn"},
    "hu": {"hu", "hun"},
    "ko": {"ko", "kor"},
    "uk": {"uk", "ukr"},
}


def select_audio_stream(
    streams: List[Dict],
    preferred_language: Optional[str] = None
) -> Tuple[int, str]:
    """
    Pick which audio track to transcribe.

    Selection order:
    1. A track tagged with `preferred_language` (the language the user expects
       to hear), if one exists.
    2. Track #0 - the first track, which is conventionally the original audio.

    Never falls back to ffmpeg's own heuristics, under which a 5.1 dub - or a
    dub flagged as the default track - outranks the 2.0 original.

    Args:
        streams: Output of list_audio_streams()
        preferred_language: ISO 639-1 code (e.g. "en"), or None/"auto"

    Returns:
        Tuple of (track_index, reason_message)
    """
    if not streams:
        return 0, "Nie wykryto ścieżek audio - używam domyślnej (#0)"

    if preferred_language and preferred_language != "auto":
        aliases = _LANG_ALIASES.get(preferred_language, {preferred_language})
        for stream in streams:
            if stream["language"] in aliases:
                return stream["track"], (
                    f"Wybrano ścieżkę {format_audio_stream(stream)} "
                    f"(zgodna z językiem źródłowym '{preferred_language}')"
                )

    chosen = streams[0]
    if len(streams) > 1:
        reason = (
            f"Wybrano ścieżkę {format_audio_stream(chosen)} "
            f"(pierwsza z {len(streams)} - zwykle oryginał)"
        )
    else:
        reason = f"Wybrano ścieżkę {format_audio_stream(chosen)}"

    return chosen["track"], reason


def get_audio_duration(wav_path: str) -> Tuple[bool, float]:
    """
    Get the duration of an audio file in seconds using ffprobe.

    Args:
        wav_path: Path to the WAV file

    Returns:
        Tuple of (success: bool, duration_seconds: float)
    """
    try:
        probe_cmd = build_ffprobe_duration_cmd(wav_path)
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return False, 0.0

        try:
            duration = float(result.stdout.strip())
            return True, duration
        except ValueError:
            return False, 0.0

    except subprocess.TimeoutExpired:
        return False, 0.0
    except Exception as e:
        print(f"Błąd przy odczytaniu długości audio: {str(e)}")
        return False, 0.0


def get_audio_duration_ms(audio_path: str) -> Tuple[bool, int]:
    """
    Get the duration of an audio file in milliseconds using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        Tuple of (success: bool, duration_ms: int)
    """
    success, duration_sec = get_audio_duration(audio_path)
    if success:
        return True, int(duration_sec * 1000)
    return False, 0


def split_audio(wav_path: str, chunk_duration_sec: int = 1800, output_dir: str = ".") -> Tuple[bool, str, List[str]]:
    """
    Split a WAV audio file into chunks of specified duration.

    Args:
        wav_path: Path to the input WAV file
        chunk_duration_sec: Duration of each chunk in seconds (default: 1800 = 30 minutes)
        output_dir: Directory to save the chunks

    Returns:
        Tuple of (success: bool, message: str, chunk_paths: List[str])
    """
    try:
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"Błąd: Plik audio nie istnieje: {wav_path}", []

        # Get audio duration
        success, duration = get_audio_duration(wav_path)
        if not success:
            return False, "Błąd: Nie można odczytać długości audio", []

        if duration == 0:
            return False, "Błąd: Audio jest puste (duration = 0)", []

        print(f"Długość audio: {duration:.1f} sekund ({duration/60:.1f} minut)")

        # If audio is shorter than chunk duration, return the original file
        if duration <= chunk_duration_sec:
            print(f"Audio krótsze niż {chunk_duration_sec} sekund - brak podziału")
            return True, "Audio nie wymaga podziału", [str(wav_path)]

        # Create output directory for chunks
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate number of chunks needed
        num_chunks = (int(duration) + chunk_duration_sec - 1) // chunk_duration_sec
        print(f"Dzielenie audio na {num_chunks} chunki po {chunk_duration_sec} sekund...")

        chunk_paths = []
        stem = wav_file.stem

        for i in tqdm(range(num_chunks), desc="Splitting audio", unit="chunk"):
            chunk_num = i + 1
            start_time = i * chunk_duration_sec
            chunk_file = output_path / f"{stem}_chunk_{chunk_num:03d}.wav"

            # Use ffmpeg to extract chunk
            cmd = build_ffmpeg_audio_split_cmd(wav_path, chunk_file, start_time, chunk_duration_sec)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return False, f"Błąd ffmpeg przy tworzeniu chunk {chunk_num}: {result.stderr}", []

            if not chunk_file.exists():
                return False, f"Błąd: Chunk {chunk_num} nie został utworzony", []

            chunk_paths.append(str(chunk_file))

        return True, f"Audio podzielone na {num_chunks} chunki", chunk_paths

    except subprocess.TimeoutExpired:
        return False, "Błąd: Dzielenie audio przerwane (timeout)", []
    except Exception as e:
        return False, f"Błąd przy dzieleniu audio: {str(e)}", []
