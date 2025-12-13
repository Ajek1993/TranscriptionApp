#!/usr/bin/env python3
"""
YouTube Video Transcription to SRT - Stage 1: Validation and Audio Download
"""

import re
import sys
import subprocess
import argparse
import os
import shutil
from pathlib import Path


def validate_youtube_url(url):
    """
    Validate if URL is a valid YouTube URL.

    Returns:
        str: Video ID if valid, None otherwise
    """
    # Pattern for YouTube URLs
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def check_dependencies():
    """
    Check if required dependencies are installed (ffmpeg, yt-dlp).

    Raises:
        SystemExit: If dependencies are missing with helpful error messages
    """
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Błąd: ffmpeg nie jest zainstalowany.")
        if sys.platform == 'darwin':
            print("Zainstaluj: brew install ffmpeg")
        elif sys.platform == 'win32':
            print("Zainstaluj: choco install ffmpeg")
        else:
            print("Zainstaluj: sudo apt-get install ffmpeg")
        sys.exit(1)

    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Błąd: yt-dlp nie jest zainstalowany.")
        print("Zainstaluj: pip install yt-dlp")
        sys.exit(1)


def download_audio(url, output_dir):
    """
    Download audio from YouTube URL and save as mono WAV at 16kHz.

    Args:
        url (str): YouTube URL
        output_dir (str): Directory to save the audio file

    Returns:
        str: Path to downloaded audio file

    Raises:
        RuntimeError: If download fails
    """
    video_id = validate_youtube_url(url)
    if not video_id:
        raise ValueError(
            f"Błąd: Niepoprawny URL YouTube. "
            f"Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID"
        )

    output_path = os.path.join(output_dir, f"{video_id}.wav")

    # yt-dlp command to extract audio
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio only
        '--audio-format', 'wav',
        '--audio-quality', '192',
        '-o', output_path,
        url
    ]

    try:
        print(f"Pobieranie audio z YouTube...")
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Błąd: Nie można pobrać filmu. "
            f"Film może być prywatny, usunięty lub niedostępny w Twoim regionie."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Błąd: Pobieranie trwało za długo (timeout)")

    # Convert to mono 16kHz using ffmpeg
    temp_path = os.path.join(output_dir, f"{video_id}_temp.wav")
    os.rename(output_path, temp_path)

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', temp_path,
        '-ac', '1',  # Convert to mono
        '-ar', '16000',  # Resample to 16kHz
        '-y',  # Overwrite without asking
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        os.remove(temp_path)
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Błąd: Nie można przetworzyć audio: {e}")

    return output_path


def main():
    """Main entry point for the transcription pipeline."""
    parser = argparse.ArgumentParser(
        description='Convert YouTube video to SRT subtitles'
    )
    parser.add_argument(
        'url',
        nargs='?',
        help='YouTube URL to transcribe'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output SRT filename'
    )
    parser.add_argument(
        '--only-download',
        action='store_true',
        help=argparse.SUPPRESS  # Developer flag
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.url:
        parser.print_help()
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Validate YouTube URL
    video_id = validate_youtube_url(args.url)
    if not video_id:
        print(
            f"Błąd: Niepoprawny URL YouTube. "
            f"Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID"
        )
        sys.exit(1)

    # Create temp directory
    output_dir = Path(os.getcwd())

    try:
        # Download audio
        audio_path = download_audio(args.url, str(output_dir))
        print(f"Pobrano audio: {audio_path}")

        if args.only_download:
            print(f"Sukces: Audio zostało pobrane do {audio_path}")
            return

        # TODO: Add other stages (chunking, transcription, etc.)

    except Exception as e:
        print(f"{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
