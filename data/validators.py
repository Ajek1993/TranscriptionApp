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

# Supported target languages for translation
SUPPORTED_TARGET_LANGUAGES = {
    'pl': 'polski',
    'en': 'angielski',
    'de': 'niemiecki',
    'fr': 'francuski',
    'es': 'hiszpański',
    'it': 'włoski',
    'pt': 'portugalski',
    'ru': 'rosyjski',
    'uk': 'ukraiński',
    'cs': 'czeski',
    'nl': 'holenderski',
    'ja': 'japoński',
    'zh-cn': 'chiński',
    'ko': 'koreański',
    'tr': 'turecki',
    'ar': 'arabski',
    'hu': 'węgierski',
}

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


def validate_youtube_url_with_message(url: str) -> Tuple[bool, str]:
    """
    Validate YouTube URL and return user-friendly error message.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid: bool, message: str)
        - If valid: (True, "✓ URL YouTube jest poprawny")
        - If invalid: (False, error_message)
    """
    if not url or not url.strip():
        return False, "URL jest pusty"

    url = url.strip()

    # Check if it starts with http/https
    if not url.startswith(('http://', 'https://')):
        return False, "URL musi zaczynać się od http:// lub https://"

    # Validate YouTube format
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+$'
    if not re.match(youtube_regex, url):
        return False, "Nieprawidłowy format URL YouTube"

    return True, "✓ URL YouTube jest poprawny"


def validate_file_extension(file_path: str, allowed_extensions: set) -> Tuple[bool, str]:
    """
    Validate file extension and return user-friendly error message.

    Args:
        file_path: Path to the file
        allowed_extensions: Set of allowed extensions (e.g., {'.mp4', '.mkv'})

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not file_path:
        return False, "Ścieżka pliku jest pusta"

    path = Path(file_path)
    ext = path.suffix.lower()

    if not ext:
        return False, "Plik nie ma rozszerzenia"

    if ext not in allowed_extensions:
        allowed_list = ', '.join(sorted(allowed_extensions))
        return False, f"Nieobsługiwany format: {ext}. Wspierane: {allowed_list}"

    return True, f"✓ Plik {ext} jest obsługiwany"


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the provided path is a valid, supported audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
    return validate_file_extension(file_path, SUPPORTED_AUDIO_EXTENSIONS)


def validate_srt_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the provided path is a valid SRT file.

    Args:
        file_path: Path to the SRT file

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not file_path:
        return False, "Ścieżka pliku jest pusta"

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext != '.srt':
        return False, "Plik musi mieć rozszerzenie .srt"

    return True, "✓ Plik SRT jest poprawny"


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


def validate_translate_format(translate_spec: str) -> Tuple[bool, str, str, str]:
    """
    Validate the --translate flag format.

    Supported formats:
    - "auto-XX" - auto-detect source, translate to XX (e.g., auto-pl, auto-en)
    - "XX-YY" - translate from XX to YY (e.g., pl-en, en-pl)

    Args:
        translate_spec: Translation specification string

    Returns:
        Tuple of (is_valid: bool, error_message: str, source_lang: str, target_lang: str)
        - If valid: (True, "", source_lang, target_lang)
        - If invalid: (False, error_message, "", "")
    """
    if not translate_spec:
        return False, "Błąd: Pusty format tłumaczenia", "", ""

    parts = translate_spec.split('-')

    # Handle special case: zh-cn as target (3 parts)
    if len(parts) == 3 and parts[1] == 'zh' and parts[2] == 'cn':
        source_lang = parts[0]
        target_lang = 'zh-cn'
    elif len(parts) == 2:
        source_lang, target_lang = parts
    else:
        supported_list = ', '.join(SUPPORTED_TARGET_LANGUAGES.keys())
        return False, (
            f"Błąd: Nieprawidłowy format flagi --translate: '{translate_spec}'\n"
            f"Poprawne formaty:\n"
            f"  --translate auto-pl    # autodetekcja → polski\n"
            f"  --translate auto-en    # autodetekcja → angielski\n"
            f"  --translate pl-en      # polski → angielski\n"
            f"  --translate en-pl      # angielski → polski\n"
            f"Obsługiwane języki docelowe: {supported_list}"
        ), "", ""

    # Validate source language (can be 'auto' or a language code)
    if source_lang != 'auto' and source_lang not in SUPPORTED_TARGET_LANGUAGES:
        supported_list = ', '.join(SUPPORTED_TARGET_LANGUAGES.keys())
        return False, (
            f"Błąd: Nieobsługiwany język źródłowy: '{source_lang}'\n"
            f"Użyj 'auto' dla automatycznej detekcji lub jeden z: {supported_list}"
        ), "", ""

    # Validate target language
    if target_lang not in SUPPORTED_TARGET_LANGUAGES:
        supported_list = ', '.join(SUPPORTED_TARGET_LANGUAGES.keys())
        return False, (
            f"Błąd: Nieobsługiwany język docelowy: '{target_lang}'\n"
            f"Obsługiwane języki: {supported_list}"
        ), "", ""

    return True, "", source_lang, target_lang
