"""
Validators Module
Input validation and dependency checking.
"""

import re
import subprocess
from pathlib import Path
from typing import Tuple

# Supported video formats for local files
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov'}

# Default output directory for downloads and generated files
FILES_DIR = Path.cwd() / "files"

# Translation support
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

# Edge TTS support
try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Coqui TTS support
try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False


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


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the provided path is a valid, supported video file.

    Automatically searches in "files" folder if file not found at provided path.

    Args:
        file_path: Path to the video file (can be absolute, relative, or just filename)

    Returns:
        Tuple of (is_valid: bool, message_or_path: str)
        - If valid: (True, absolute_path_to_file)
        - If invalid: (False, error_message)
    """
    # Try original path first (backward compatibility)
    path = Path(file_path)

    if path.exists():
        # Found at provided path - validate extension
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            supported = ', '.join(SUPPORTED_VIDEO_EXTENSIONS)
            return False, f"Błąd: Nieobsługiwany format. Wspierane: {supported}"
        return True, str(path.absolute())

    # If not found and not an absolute path, try "files" folder
    if not path.is_absolute():
        files_path = FILES_DIR / file_path
        if files_path.exists():
            # Found in files folder - validate extension
            if files_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                supported = ', '.join(SUPPORTED_VIDEO_EXTENSIONS)
                return False, f"Błąd: Nieobsługiwany format. Wspierane: {supported}"
            return True, str(files_path.absolute())

    # Not found in either location - provide helpful error
    if not path.is_absolute():
        return False, (f"Błąd: Plik nie istnieje.\n"
                      f"  Sprawdzono: {path.absolute()}\n"
                      f"  Sprawdzono: {FILES_DIR / file_path}")
    else:
        return False, f"Błąd: Plik nie istnieje: {file_path}"


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


def check_edge_tts_dependency() -> Tuple[bool, str]:
    """
    Check if edge-tts is available for TTS dubbing.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not EDGE_TTS_AVAILABLE:
        return False, "Błąd: edge-tts nie jest zainstalowany. Zainstaluj: pip install edge-tts"
    return True, "edge-tts dostępny"


def check_coqui_tts_dependency() -> Tuple[bool, str]:
    """
    Check if Coqui TTS is available for TTS dubbing.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not COQUI_TTS_AVAILABLE:
        return False, "Błąd: Coqui TTS nie jest zainstalowany. Zainstaluj: pip install TTS"
    return True, "Coqui TTS dostępny"
