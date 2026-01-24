#!/usr/bin/env python3
"""
Audio processing utilities.

This module provides functions for:
- Getting audio duration
- Splitting audio files into chunks
"""

import subprocess
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

from .command_builders import build_ffprobe_duration_cmd, build_ffmpeg_audio_split_cmd


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
