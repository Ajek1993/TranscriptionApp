#!/usr/bin/env python3
"""
SRT file reader for parsing existing subtitle files.

This module provides functionality for reading SRT subtitle files
and converting them to the internal segment format.
"""

import re
from pathlib import Path
from typing import List, Tuple


def parse_srt_timestamp(timestamp: str) -> int:
    """
    Convert SRT timestamp to milliseconds.

    Args:
        timestamp: SRT timestamp in format "HH:MM:SS,mmm"

    Returns:
        Time in milliseconds
    """
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp.strip())
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        ms = int(match.group(4))
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + ms
    return 0


def parse_srt_file(file_path: str) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Parse an SRT file and extract segments.

    Args:
        file_path: Path to the SRT file

    Returns:
        Tuple of (success: bool, message: str, segments: List[(start_ms, end_ms, text)])
    """
    segments = []

    try:
        # Try UTF-8 first, then UTF-8 with BOM, then Latin-1
        content = None
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return False, f"Blad: Nie mozna odczytac pliku SRT (nieznane kodowanie): {file_path}", []

        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Split by double newlines (subtitle blocks)
        blocks = re.split(r'\n\n+', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue

            # Find timestamp line (can be line 1 or 2 depending on whether index is present)
            timestamp_line_idx = None
            for i, line in enumerate(lines):
                if '-->' in line:
                    timestamp_line_idx = i
                    break

            if timestamp_line_idx is None:
                continue

            # Parse timestamp line
            timestamp_match = re.match(
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                lines[timestamp_line_idx]
            )

            if timestamp_match:
                start_ms = parse_srt_timestamp(timestamp_match.group(1))
                end_ms = parse_srt_timestamp(timestamp_match.group(2))

                # Text is everything after the timestamp line
                text_lines = lines[timestamp_line_idx + 1:]
                text = ' '.join(line.strip() for line in text_lines if line.strip())

                # Remove HTML-like tags (e.g., <i>, </i>, <b>, </b>)
                text = re.sub(r'<[^>]+>', '', text)

                if text:
                    segments.append((start_ms, end_ms, text))

        if not segments:
            return False, f"Blad: Nie znaleziono segmentow w pliku SRT: {file_path}", []

        return True, f"Zaladowano {len(segments)} segmentow", segments

    except FileNotFoundError:
        return False, f"Blad: Plik SRT nie istnieje: {file_path}", []
    except Exception as e:
        return False, f"Blad: Nie mozna odczytac pliku SRT: {file_path}\n  {str(e)}", []


def validate_srt_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the provided path is a valid SRT file.

    Checks the provided path directly, then falls back to 'files/' folder.

    Args:
        file_path: Path to the SRT file (can be absolute, relative, or just filename)

    Returns:
        Tuple of (is_valid: bool, message_or_path: str)
        - If valid: (True, absolute_path_to_file)
        - If invalid: (False, error_message)
    """
    path = Path(file_path)

    # Check if file extension is .srt
    if path.suffix.lower() != '.srt':
        return False, f"Blad: Plik musi miec rozszerzenie .srt: {file_path}"

    # First, check the provided path directly
    if path.exists() and path.is_file():
        return True, str(path.resolve())

    # If not found, try 'files/' folder
    files_path = Path('files') / path.name
    if files_path.exists() and files_path.is_file():
        return True, str(files_path.resolve())

    # File not found
    return False, f"Blad: Plik SRT nie istnieje.\n  Sprawdzono: {path}\n  Sprawdzono: {files_path}"


def detect_language_from_segments(segments: List[Tuple[int, int, str]]) -> str:
    """
    Detect language from segment text content.

    Uses langdetect library to determine language from subtitle text.

    Args:
        segments: List of (start_ms, end_ms, text) tuples

    Returns:
        ISO language code (e.g., "pl", "en", "de")
    """
    try:
        from langdetect import detect, DetectorFactory

        # Make detection deterministic
        DetectorFactory.seed = 0

        # Combine text from first N segments (enough for reliable detection)
        sample_text = ' '.join([text for _, _, text in segments[:30]])

        if len(sample_text) < 20:
            return "pl"  # Default fallback for very short text

        detected = detect(sample_text)

        # Map langdetect codes to TTS-compatible codes
        language_map = {
            'pl': 'pl',
            'en': 'en',
            'de': 'de',
            'fr': 'fr',
            'es': 'es',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ja': 'ja',
            'zh-cn': 'zh-cn',
            'zh-tw': 'zh-cn',
            'ko': 'ko',
            'uk': 'uk',
            'cs': 'cs',
            'nl': 'nl',
            'tr': 'tr',
            'ar': 'ar',
            'hu': 'hu',
        }

        return language_map.get(detected, detected)

    except ImportError:
        # langdetect not installed - return default
        print("Ostrzezenie: Biblioteka langdetect nie jest zainstalowana. Uzywam domyslnego jezyka: pl")
        return "pl"
    except Exception:
        # Detection failed - return default
        return "pl"
