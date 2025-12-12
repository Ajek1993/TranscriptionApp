#!/usr/bin/env python3
"""
YouTube to SRT Transcription Tool
MVP Stage 1: Validate and download audio from YouTube
MVP Stage 2: Split audio into chunks (~30 minutes each)
"""

import re
import subprocess
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, List


def validate_youtube_url(url: str) -> bool:
    """
    Validate if the provided URL is a valid YouTube URL.

    Args:
        url: The URL to validate

    Returns:
        True if valid YouTube URL, False otherwise
    """
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    return bool(re.match(youtube_regex, url))


def check_dependencies() -> Tuple[bool, str]:
    """
    Check if required dependencies are available (ffmpeg and yt-dlp).

    Returns:
        Tuple of (success: bool, message: str)
    """
    missing_deps = []

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'],
                      capture_output=True,
                      timeout=5,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        missing_deps.append('ffmpeg')

    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'],
                      capture_output=True,
                      timeout=5,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        missing_deps.append('yt-dlp')

    if missing_deps:
        error_msg = "Błąd: Brakuje wymaganych narzędzi:\n\n"

        if 'ffmpeg' in missing_deps:
            error_msg += "ffmpeg nie jest zainstalowany. Zainstaluj:\n"
            error_msg += "  - winget install FFmpeg\n"
            error_msg += "  - lub choco install ffmpeg\n"
            error_msg += "  - lub pobierz z https://ffmpeg.org/download.html\n\n"

        if 'yt-dlp' in missing_deps:
            error_msg += "yt-dlp nie jest zainstalowany. Zainstaluj: pip install yt-dlp\n"

        return False, error_msg

    return True, "Wszystkie zależności dostępne"


def download_audio(url: str, output_dir: str = ".") -> Tuple[bool, str, str]:
    """
    Download audio from YouTube and save as WAV (mono, 16kHz).

    Args:
        url: YouTube URL
        output_dir: Directory to save the audio file

    Returns:
        Tuple of (success: bool, message: str, audio_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract video ID for naming
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
    if not video_id_match:
        return False, "Błąd: Nie udało się wyodrębnić ID wideo z URL", ""

    video_id = video_id_match.group(1)
    audio_file = output_path / f"{video_id}.wav"

    try:
        print(f"Pobieranie audio z YouTube... ({url})")

        cmd = [
            'yt-dlp',
            '-f', 'bestaudio/best',
            '-x',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '-o', str(audio_file),
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd yt-dlp"
            if "private" in error_msg.lower() or "not available" in error_msg.lower():
                return False, "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie.", ""
            else:
                return False, f"Błąd yt-dlp: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        print(f"Audio pobrane: {audio_file}")

        # Verify audio format with ffprobe
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries',
                     'stream=channels,sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1:noescapes=1',
                     str(audio_file)]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            channels = int(output_info[0]) if len(output_info) > 0 else 0
            sample_rate = int(output_info[1]) if len(output_info) > 1 else 0
            print(f"Format audio: {channels} kanał(y), {sample_rate} Hz")

        return True, f"Audio pobrane pomyślnie: {audio_file}", str(audio_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Pobieranie przerwane (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy pobieraniu: {str(e)}", ""


def get_audio_duration(wav_path: str) -> Tuple[bool, float]:
    """
    Get the duration of an audio file in seconds using ffprobe.

    Args:
        wav_path: Path to the WAV file

    Returns:
        Tuple of (success: bool, duration_seconds: float)
    """
    try:
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(wav_path)
        ]

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

        for i in range(num_chunks):
            chunk_num = i + 1
            start_time = i * chunk_duration_sec
            chunk_file = output_path / f"{stem}_chunk_{chunk_num:03d}.wav"

            # Use ffmpeg to extract chunk
            cmd = [
                'ffmpeg',
                '-i', str(wav_path),
                '-ss', str(start_time),
                '-t', str(chunk_duration_sec),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(chunk_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return False, f"Błąd ffmpeg przy tworzeniu chunk {chunk_num}: {result.stderr}", []

            if not chunk_file.exists():
                return False, f"Błąd: Chunk {chunk_num} nie został utworzony", []

            chunk_paths.append(str(chunk_file))
            print(f"Stworzony chunk {chunk_num}/{num_chunks}: {chunk_file.name}")

        return True, f"Audio podzielone na {num_chunks} chunki", chunk_paths

    except subprocess.TimeoutExpired:
        return False, "Błąd: Dzielenie audio przerwane (timeout)", []
    except Exception as e:
        return False, f"Błąd przy dzieleniu audio: {str(e)}", []


def main():
    parser = argparse.ArgumentParser(
        description='Pobierz audio z YouTube i konwertuj na transkrypcję SRT'
    )
    parser.add_argument('url', nargs='?', help='URL YouTube do transkrypcji')
    parser.add_argument('--only-download', action='store_true',
                       help='Tylko pobierz audio, nie transkrybuj (developerski)')
    parser.add_argument('--only-chunk', action='store_true',
                       help='Tylko podziel audio, nie transkrybuj (developerski)')

    args = parser.parse_args()

    # Check if URL was provided
    if not args.url:
        print("Błąd: Podaj URL YouTube")
        print("Użycie: python transcribe.py \"https://www.youtube.com/watch?v=VIDEO_ID\"")
        return 1

    # Validate URL
    if not validate_youtube_url(args.url):
        print("Błąd: Niepoprawny URL YouTube. Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID")
        return 1

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(deps_msg)
        return 1

    # Download audio
    success, message, audio_path = download_audio(args.url)
    if not success:
        print(message)
        return 1

    print(message)

    if args.only_download:
        return 0

    # Stage 2: Split audio into chunks
    success, message, chunk_paths = split_audio(audio_path)
    if not success:
        print(message)
        return 1

    print(message)
    print(f"Chunks: {chunk_paths}")

    if args.only_chunk:
        return 0

    # Future stages will go here
    print("Etapy 3-5 będą wdrażane w następnych fazach.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
