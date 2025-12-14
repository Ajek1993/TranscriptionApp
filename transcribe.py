#!/usr/bin/env python3
"""
YouTube to SRT Transcription Tool
MVP Stage 1: Validate and download audio from YouTube
MVP Stage 2: Split audio into chunks (~30 minutes each)
MVP Stage 3: Transcribe audio using faster-whisper
MVP Stage 4: Merge segments and generate SRT file
MVP Stage 5: Complete pipeline with CLI and automatic cleanup
"""

import re
import subprocess
import sys
import argparse
import json
import tempfile
import shutil
import time
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

# Supported video formats for local files
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov'}

# Translation support
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


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

    Args:
        file_path: Path to the video file

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"Błąd: Plik nie istnieje: {file_path}"

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ', '.join(SUPPORTED_VIDEO_EXTENSIONS)
        return False, f"Błąd: Nieobsługiwany format. Wspierane: {supported}"

    return True, "OK"


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


def extract_audio_from_video(video_path: str, output_dir: str = ".") -> Tuple[bool, str, str]:
    """
    Extract audio from a local video file and convert to WAV (mono, 16kHz, PCM 16-bit).

    Args:
        video_path: Path to the local video file
        output_dir: Directory to save the extracted audio

    Returns:
        Tuple of (success: bool, message: str, audio_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on input video filename
    video_file = Path(video_path)
    audio_file = output_path / f"{video_file.stem}.wav"

    try:
        print(f"Ekstrakcja audio z pliku wideo... ({video_path})")

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd ffmpeg"
            return False, f"Błąd ffmpeg: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        print(f"Audio wyekstrahowane: {audio_file}")

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

        return True, f"Audio wyekstrahowane pomyślnie: {audio_file}", str(audio_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Ekstrakcja audio przerwana (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy ekstrakcji audio: {str(e)}", ""


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


def detect_device() -> Tuple[str, str]:
    """
    Detect available device for faster-whisper (CUDA GPU or CPU).

    Returns:
        Tuple of (device: str, device_info: str)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", f"NVIDIA GPU ({gpu_name})"
    except ImportError:
        pass
    except Exception:
        pass

    return "cpu", "CPU"


def transcribe_chunk(wav_path: str, model_size: str = "base", language: str = "pl", segment_progress_bar: tqdm = None) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transcribe a WAV audio file using faster-whisper.

    Args:
        wav_path: Path to the WAV file
        model_size: Model size to use (tiny, base, small, medium, large)
        language: Language code (default: pl for Polish)
        segment_progress_bar: Optional tqdm progress bar for segment updates

    Returns:
        Tuple of (success: bool, message: str, segments: List[(start_ms, end_ms, text)])
    """
    try:
        # Check if faster-whisper is installed
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return False, "Błąd: faster-whisper nie jest zainstalowany. Zainstaluj: pip install faster-whisper", []

        # Check if file exists
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"Błąd: Plik audio nie istnieje: {wav_path}", []

        # Detect device
        device, device_info = detect_device()
        tqdm.write(f"Używane urządzenie: {device_info}")

        # Initialize model
        tqdm.write(f"Ładowanie modelu {model_size}...")
        try:
            model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU. Błąd: {e}")
                device = "cpu"
                device_info = "CPU (fallback)"
                model = WhisperModel(model_size, device=device, compute_type="int8")
            else:
                raise

        tqdm.write(f"Transkrypcja: {wav_file.name}...")

        # Transcribe
        segments_generator, info = model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=False,
            vad_filter=True
        )

        tqdm.write(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

        # Parse segments to list
        segments = []
        for segment in segments_generator:
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            text = segment.text.strip()
            segments.append((start_ms, end_ms, text))

            # Update progress bar if provided
            if segment_progress_bar:
                segment_progress_bar.set_postfix_str(f"{len(segments)} segments")

        if not segments:
            return False, "Błąd: Transkrypcja nie zwróciła żadnych segmentów (puste audio lub brak mowy)", []

        return True, f"Transkrypcja zakończona pomyślnie: {len(segments)} segmentów", segments

    except ImportError as e:
        return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}. Zainstaluj: pip install faster-whisper", []
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []


def format_srt_timestamp(ms: int) -> str:
    """
    Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted timestamp string in SRT format
    """
    total_seconds = ms // 1000
    milliseconds = ms % 1000

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def translate_segments(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    batch_size: int = 50
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Translate text content of segments while preserving timestamps.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        source_lang: Source language code ('pl' or 'en')
        target_lang: Target language code ('pl' or 'en')
        batch_size: Number of segments to translate in one batch

    Returns:
        Tuple of (success: bool, message: str, translated_segments: List)
    """
    if not TRANSLATOR_AVAILABLE:
        return False, "Błąd: deep-translator nie jest zainstalowany. Zainstaluj: pip install deep-translator", []

    if not segments:
        return True, "Brak segmentów do tłumaczenia", []

    try:
        # Użyj bezpośrednio skrótów języków - GoogleTranslator je obsługuje
        translator = GoogleTranslator(source=source_lang, target=target_lang)

        translated_segments = []

        print(f"Tłumaczenie {len(segments)} segmentów ({source_lang} -> {target_lang})...")

        # Process in batches for efficiency
        with tqdm(total=len(segments), desc="Translating", unit="seg") as pbar:
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]

                # Extract texts for batch translation
                texts = [text for _, _, text in batch]

                # Translate texts
                try:
                    translated_texts = []
                    for text in texts:
                        try:
                            translated_text = translator.translate(text)
                            translated_texts.append(translated_text)
                        except Exception as e:
                            # Keep original text on translation error
                            tqdm.write(f"Ostrzeżenie: Nie udało się przetłumaczyć tekstu: {text[:50]}... ({str(e)})")
                            translated_texts.append(text)
                except Exception as e:
                    return False, f"Błąd podczas tłumaczenia: {str(e)}", []

                # Rebuild segments with translated text
                for j, (start_ms, end_ms, _) in enumerate(batch):
                    translated_text = translated_texts[j] if j < len(translated_texts) else batch[j][2]
                    translated_segments.append((start_ms, end_ms, translated_text))

                pbar.update(len(batch))

        return True, f"Przetłumaczono {len(translated_segments)} segmentów", translated_segments

    except Exception as e:
        return False, f"Błąd podczas tłumaczenia: {str(e)}", []


def write_srt(segments: List[Tuple[int, int, str]], output_path: str) -> Tuple[bool, str]:
    """
    Write segments to an SRT file with UTF-8 encoding.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        output_path: Path to the output SRT file

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not segments:
            print("Ostrzeżenie: Brak segmentów do zapisania")
            # Create empty SRT file with warning comment
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Pusty plik SRT - brak segmentów transkrypcji\n")
            return True, f"Zapisano pusty plik SRT (brak segmentów): {output_path}"

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, (start_ms, end_ms, text) in enumerate(segments, 1):
                # SRT format:
                # 1
                # 00:00:00,000 --> 00:00:05,000
                # Text content
                # [blank line]

                f.write(f"{idx}\n")
                f.write(f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}\n")
                f.write(f"{text}\n")
                f.write("\n")

        return True, f"Zapisano {len(segments)} segmentów do pliku SRT: {output_path}"

    except Exception as e:
        return False, f"Błąd przy zapisie pliku SRT: {str(e)}"


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

        return True, f"Audio podzielone na {num_chunks} chunki", chunk_paths

    except subprocess.TimeoutExpired:
        return False, "Błąd: Dzielenie audio przerwane (timeout)", []
    except Exception as e:
        return False, f"Błąd przy dzieleniu audio: {str(e)}", []


def cleanup_temp_files(temp_dir: str, retries: int = 3, delay: float = 0.2) -> None:
    """
    Clean up temporary files and directories with retry mechanism for Windows file locks.

    Args:
        temp_dir: Path to the temporary directory to remove
        retries: Number of retry attempts (default: 3)
        delay: Delay between retries in seconds (default: 0.2)
    """
    if not temp_dir or not Path(temp_dir).exists():
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(temp_dir)
            print(f"Pliki tymczasowe usunięte: {temp_dir}")
            return
        except PermissionError as e:
            if attempt < retries - 1:
                # Wait a bit for Windows to release file locks
                time.sleep(delay)
            else:
                print(f"Ostrzeżenie: Nie można usunąć plików tymczasowych: {temp_dir}")
                print(f"Błąd: {e}")
                print("Możesz usunąć je ręcznie później.")
        except Exception as e:
            print(f"Ostrzeżenie: Błąd przy usuwaniu plików tymczasowych: {e}")
            return


def main():
    parser = argparse.ArgumentParser(
        description='Transkrypcja z YouTube lub lokalnych plików wideo na SRT'
    )
    parser.add_argument('url', nargs='?', help='URL YouTube do transkrypcji')
    parser.add_argument('-l', '--local', type=str,
                       help='Ścieżka do lokalnego pliku wideo (MP4, MKV, AVI, MOV)')
    parser.add_argument('--only-download', action='store_true',
                       help='Tylko pobierz audio, nie transkrybuj (developerski)')
    parser.add_argument('--only-chunk', action='store_true',
                       help='Tylko podziel audio, nie transkrybuj (developerski)')
    parser.add_argument('--only-transcribe', action='store_true',
                       help='Tylko transkrybuj chunki, nie generuj SRT (developerski)')
    parser.add_argument('--test-merge', action='store_true',
                       help='Test generowania SRT z hardcoded danymi (developerski)')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Rozmiar modelu Whisper (domyślnie: base)')
    parser.add_argument('--language', type=str, default='pl',
                       help='Język transkrypcji (domyślnie: pl)')
    parser.add_argument('-t', '--translate', type=str, choices=['pl-en', 'en-pl'],
                       help='Tłumaczenie (pl-en: polski->angielski, en-pl: angielski->polski)')
    parser.add_argument('-o', '--output', type=str,
                       help='Nazwa pliku wyjściowego SRT (domyślnie: video_id.srt lub nazwa_pliku.srt)')

    args = parser.parse_args()

    # Test merge functionality with hardcoded data
    if args.test_merge:
        print("Test generowania SRT z przykładowymi danymi...")

        # Hardcoded test segments simulating 2 chunks
        # Chunk 1: 0-30 seconds
        chunk1_segments = [
            (0, 5000, "Witam w przykładowym filmie o transkrypcji."),
            (5500, 12000, "To jest pierwszy segment z pierwszego chunka."),
            (12500, 20000, "Tutaj sprawdzamy czy polskie znaki działają: ąćęłńóśźż."),
            (20500, 30000, "Koniec pierwszego chunka.")
        ]

        # Chunk 2: 30-60 seconds (timestamps will be adjusted)
        chunk2_segments = [
            (0, 8000, "To jest początek drugiego chunka."),
            (8500, 18000, "Timestampy powinny być przesunięte o 30 sekund."),
            (18500, 28000, "Sprawdzamy merge i scalanie segmentów."),
            (28500, 35000, "Koniec testu.")
        ]

        # Simulate merging process
        all_segments = []
        chunk_offsets = [0, 30000]  # Second chunk starts at 30 seconds

        # Add chunk 1 segments with offset
        for start_ms, end_ms, text in chunk1_segments:
            all_segments.append((start_ms + chunk_offsets[0], end_ms + chunk_offsets[0], text))

        # Add chunk 2 segments with offset
        for start_ms, end_ms, text in chunk2_segments:
            all_segments.append((start_ms + chunk_offsets[1], end_ms + chunk_offsets[1], text))

        # Write to test output file
        test_output = "test_output.srt"
        success, message = write_srt(all_segments, test_output)

        if success:
            print(message)
            print(f"\nPodgląd zawartości {test_output}:")
            print("-" * 60)
            with open(test_output, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:500])  # Print first 500 chars
                if len(content) > 500:
                    print("...")
            print("-" * 60)
            print(f"\nOtwórz plik '{test_output}' w VLC lub innym odtwarzaczu aby sprawdzić napisy.")
            return 0
        else:
            print(message)
            return 1

    # Check input source (YouTube URL or local file)
    if not args.url and not args.local:
        print("Błąd: Podaj URL YouTube lub ścieżkę do lokalnego pliku wideo")
        print("Użycie:")
        print("  - YouTube: python transcribe.py \"https://www.youtube.com/watch?v=VIDEO_ID\"")
        print("  - Lokalny: python transcribe.py --local \"C:\\path\\to\\video.mp4\"")
        return 1

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(deps_msg)
        return 1

    # Create temporary directory for intermediate files
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="transcribe_")
        print(f"Katalog tymczasowy: {temp_dir}")

        # Process input source: local file or YouTube URL
        if args.local:
            # Local file mode
            is_valid, error_msg = validate_video_file(args.local)
            if not is_valid:
                print(error_msg)
                return 1

            # Extract audio from local video
            success, message, audio_path = extract_audio_from_video(args.local, output_dir=temp_dir)
            if not success:
                print(message)
                return 1

            print(message)
            input_stem = Path(args.local).stem
        else:
            # YouTube mode
            # Validate URL
            if not validate_youtube_url(args.url):
                print("Błąd: Niepoprawny URL YouTube. Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID")
                return 1

            # Download audio from YouTube
            success, message, audio_path = download_audio(args.url, output_dir=temp_dir)
            if not success:
                print(message)
                return 1

            print(message)

            # Extract video ID for naming
            video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', args.url)
            input_stem = video_id_match.group(1) if video_id_match else "output"

        if args.only_download:
            # For --only-download, copy file to current directory before cleanup
            audio_file = Path(audio_path)
            dest_path = Path.cwd() / audio_file.name
            shutil.copy2(audio_path, dest_path)
            print(f"Plik audio skopiowany do: {dest_path}")
            return 0

        # Stage 2: Split audio into chunks (in temp directory)
        success, message, chunk_paths = split_audio(audio_path, output_dir=temp_dir)
        if not success:
            print(message)
            return 1

        print(message)

        if args.only_chunk:
            # For --only-chunk, copy chunks to current directory before cleanup
            for chunk_path in chunk_paths:
                chunk_file = Path(chunk_path)
                dest_path = Path.cwd() / chunk_file.name
                shutil.copy2(chunk_path, dest_path)
                print(f"Chunk skopiowany do: {dest_path}")
            return 0

        # Stage 3: Transcribe each chunk
        all_segments = []
        chunk_offsets = []
        current_offset_ms = 0

        # Create overall progress bar for chunks
        with tqdm(total=len(chunk_paths), desc="Overall Progress", unit="chunk", position=0) as chunk_pbar:
            for idx, chunk_path in enumerate(chunk_paths, 1):
                chunk_file_name = Path(chunk_path).name

                # Update chunk progress bar description
                chunk_pbar.set_description(f"Chunk {idx}/{len(chunk_paths)}")

                # Create segment progress bar for this chunk (indeterminate - no total)
                segment_pbar = tqdm(
                    desc=f"  └─ Transcribing {chunk_file_name}",
                    unit=" seg",
                    position=1,
                    leave=False,  # Don't leave segment bar after completion
                    total=0,  # Set total to 0 for indeterminate progress
                    bar_format='{desc}: {postfix}'  # Custom format without progress bar
                )

                # Transcribe with progress callback
                success, message, segments = transcribe_chunk(
                    chunk_path,
                    model_size=args.model,
                    language=args.language,
                    segment_progress_bar=segment_pbar
                )

                # Close segment progress bar
                segment_pbar.close()

                if not success:
                    print(message)
                    return 1

                # Store offset for this chunk
                chunk_offsets.append(current_offset_ms)

                # Adjust timestamps and add to all_segments
                adjusted_segments = [(start_ms + current_offset_ms, end_ms + current_offset_ms, text)
                                    for start_ms, end_ms, text in segments]
                all_segments.extend(adjusted_segments)

                # Update offset for next chunk (based on last segment end time)
                if segments:
                    last_segment_end = segments[-1][1]  # end_ms of last segment
                    current_offset_ms += last_segment_end

                # Update chunk progress bar with segment count
                chunk_pbar.set_postfix_str(f"{len(segments)} segments, total: {len(all_segments)}")
                chunk_pbar.update(1)

        # Translation step (if requested)
        if args.translate:
            print(f"\n=== Etap: Tłumaczenie ===")

            # Parse translation direction
            src_lang, tgt_lang = args.translate.split('-')

            # Validate source language matches transcription language
            if args.language != src_lang:
                print(f"Ostrzeżenie: Język transkrypcji ({args.language}) różni się od źródłowego języka tłumaczenia ({src_lang})")

            success, message, translated_segments = translate_segments(
                all_segments,
                source_lang=src_lang,
                target_lang=tgt_lang
            )

            if not success:
                print(message)
                return 1

            print(message)
            all_segments = translated_segments

        if args.only_transcribe:
            print(f"\nŁącznie transkrybowanych segmentów: {len(all_segments)}")
            return 0

        # Stage 4: Generate SRT file
        print(f"\n=== Etap 4: Generowanie pliku SRT ===")
        print(f"Łącznie segmentów: {len(all_segments)}")

        # Determine output filename
        if args.output:
            srt_filename = args.output
            if not srt_filename.endswith('.srt'):
                srt_filename += '.srt'
        else:
            srt_filename = f"{input_stem}.srt"

        # Write SRT file to current directory
        success, message = write_srt(all_segments, srt_filename)
        if not success:
            print(message)
            return 1

        print(message)
        print(f"\n[OK] Transkrypcja zakończona pomyślnie!")
        print(f"[OK] Plik SRT zapisany: {srt_filename}")
        print(f"\nMożesz otworzyć plik SRT w VLC lub innym odtwarzaczu wideo.")

        return 0

    finally:
        # Always cleanup temporary files, even on error
        if temp_dir:
            cleanup_temp_files(temp_dir)


if __name__ == '__main__':
    sys.exit(main())
