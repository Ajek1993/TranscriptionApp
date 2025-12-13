#!/usr/bin/env python3
"""
YouTube Video Transcription to SRT - Complete Pipeline
"""

import re
import sys
import subprocess
import argparse
import os
import shutil
import tempfile
from pathlib import Path

# Parakeet-mlx will be imported when needed to provide better error messages
try:
    from parakeet_mlx import from_pretrained
    PARAKEET_AVAILABLE = True
except ImportError:
    PARAKEET_AVAILABLE = False


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


def get_audio_duration(wav_path):
    """
    Get duration of audio file in seconds using ffprobe.

    Args:
        wav_path (str): Path to WAV file

    Returns:
        float: Duration in seconds

    Raises:
        RuntimeError: If duration cannot be determined
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        wav_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        duration = float(result.stdout.strip())

        if duration <= 0:
            raise RuntimeError(f"Błąd: Plik audio ma nieprawidłową długość: {duration}s")

        return duration
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Błąd: Nie można odczytać długości pliku audio: {e}")
    except ValueError:
        raise RuntimeError("Błąd: ffprobe zwrócił nieprawidłową wartość długości")


def split_audio(wav_path, chunk_duration_sec, output_dir):
    """
    Split audio file into chunks of specified duration.

    Args:
        wav_path (str): Path to input WAV file
        chunk_duration_sec (int): Duration of each chunk in seconds
        output_dir (str): Directory to save chunks

    Returns:
        list: List of paths to chunk files

    Raises:
        RuntimeError: If splitting fails
    """
    # Get audio duration
    duration = get_audio_duration(wav_path)

    # If audio is shorter than chunk duration, no splitting needed
    if duration <= chunk_duration_sec:
        print(f"Audio ({duration:.1f}s) jest krótsze niż chunk ({chunk_duration_sec}s) - pomijanie podziału")
        return [wav_path]

    # Calculate number of chunks
    num_chunks = int(duration / chunk_duration_sec) + (1 if duration % chunk_duration_sec > 0 else 0)
    print(f"Dzielenie audio ({duration:.1f}s) na {num_chunks} chunków po {chunk_duration_sec}s...")

    chunk_paths = []
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1:03d}.wav")

        cmd = [
            'ffmpeg',
            '-i', wav_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration_sec),
            '-c', 'copy',  # Copy codec without re-encoding
            '-y',  # Overwrite without asking
            chunk_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            chunk_paths.append(chunk_path)
            print(f"  Utworzono chunk {i+1}/{num_chunks}: {os.path.basename(chunk_path)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Błąd: Nie można utworzyć chunku {i+1}: {e}")

    return chunk_paths


def transcribe_chunk(wav_path):
    """
    Transcribe audio chunk using parakeet-mlx.

    Args:
        wav_path (str): Path to WAV file (mono, 16kHz)

    Returns:
        list: List of segments [(start_ms, end_ms, "text"), ...]

    Raises:
        RuntimeError: If transcription fails
    """
    if not PARAKEET_AVAILABLE:
        raise RuntimeError(
            "Błąd: parakeet-mlx nie jest zainstalowany.\n"
            "Zainstaluj: pip install parakeet-mlx"
        )

    try:
        print(f"  Ładowanie modelu parakeet-mlx...")
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

        print(f"  Transkrypcja pliku: {os.path.basename(wav_path)}")
        result = model.transcribe(wav_path)

        # Check if we got any results
        if not hasattr(result, 'sentences') or not result.sentences:
            raise RuntimeError("Błąd: Transkrypcja nie zwróciła żadnych wyników")

        # Convert sentences to our format
        segments = []
        for sentence in result.sentences:
            start_ms = int(sentence.start * 1000)
            end_ms = int(sentence.end * 1000)
            text = sentence.text.strip()
            if text:
                segments.append((start_ms, end_ms, text))

        if not segments:
            raise RuntimeError("Błąd: Transkrypcja nie zwróciła żadnych segmentów tekstowych")

        return segments

    except ImportError as e:
        raise RuntimeError(
            f"Błąd: Nie można zaimportować parakeet-mlx: {e}\n"
            f"Zainstaluj: pip install parakeet-mlx"
        )
    except Exception as e:
        raise RuntimeError(f"Błąd podczas transkrypcji: {e}")


def format_srt_timestamp(ms):
    """
    Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        ms (int): Time in milliseconds

    Returns:
        str: Formatted timestamp (e.g., "00:01:23,456")
    """
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments, output_path):
    """
    Write segments to SRT file.

    Args:
        segments (list): List of tuples [(start_ms, end_ms, "text"), ...]
        output_path (str): Path to output SRT file

    Raises:
        RuntimeError: If writing fails
    """
    if not segments:
        print("Ostrzeżenie: Brak segmentów do zapisania")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (start_ms, end_ms, text) in enumerate(segments, 1):
                # SRT format:
                # 1
                # 00:00:00,000 --> 00:00:05,000
                # Text content
                # (blank line)
                f.write(f"{i}\n")
                f.write(f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}\n")
                f.write(f"{text}\n")
                f.write("\n")
    except Exception as e:
        raise RuntimeError(f"Błąd podczas zapisywania pliku SRT: {e}")


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
    parser.add_argument(
        '--only-chunk',
        action='store_true',
        help=argparse.SUPPRESS  # Developer flag
    )
    parser.add_argument(
        '--only-transcribe',
        action='store_true',
        help=argparse.SUPPRESS  # Developer flag
    )
    parser.add_argument(
        '--test-merge',
        action='store_true',
        help=argparse.SUPPRESS  # Developer flag
    )

    args = parser.parse_args()

    # Test merge with hardcoded data
    if args.test_merge:
        print("Testowanie funkcji scalania i generowania SRT...")
        test_segments = [
            (0, 2000, "Pierwszy segment transkrypcji."),
            (2000, 5500, "Drugi segment z różnym czasem."),
            (5500, 10000, "Trzeci segment na końcu."),
        ]
        test_output = "test_output.srt"
        write_srt(test_segments, test_output)
        print(f"Sukces: Utworzono {test_output}")
        print("\nZawartość pliku:")
        with open(test_output, 'r', encoding='utf-8') as f:
            print(f.read())
        return

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
    temp_dir = tempfile.mkdtemp(prefix="transcribe_")

    try:
        # Download audio
        audio_path = download_audio(args.url, temp_dir)
        print(f"Pobrano audio: {audio_path}")

        if args.only_download:
            print(f"Sukces: Audio zostało pobrane do {audio_path}")
            return

        # Chunk audio
        chunk_duration_sec = 30 * 60  # 30 minutes in seconds
        chunk_paths = split_audio(audio_path, chunk_duration_sec, temp_dir)

        if args.only_chunk:
            print(f"Sukces: Audio podzielone na {len(chunk_paths)} chunków")
            for i, chunk_path in enumerate(chunk_paths, 1):
                print(f"  Chunk {i}: {os.path.basename(chunk_path)}")
            return

        # Transcribe chunks
        print(f"Rozpoczynanie transkrypcji {len(chunk_paths)} chunk(ów)...")
        all_segments = []

        for i, chunk_path in enumerate(chunk_paths, 1):
            print(f"Transkrypcja chunk {i}/{len(chunk_paths)}...")
            segments = transcribe_chunk(chunk_path)

            # If this is not the first chunk, offset timestamps
            if i > 1:
                offset_ms = (i - 1) * chunk_duration_sec * 1000
                segments = [(start + offset_ms, end + offset_ms, text)
                           for start, end, text in segments]

            all_segments.extend(segments)
            print(f"  Chunk {i}: {len(segments)} segmentów")

        if args.only_transcribe:
            print(f"\nSukces: Transkrypcja zakończona")
            print(f"Łącznie segmentów: {len(all_segments)}")
            print(f"\nPrzykładowe segmenty:")
            for start_ms, end_ms, text in all_segments[:3]:
                print(f"  [{start_ms}ms - {end_ms}ms] {text}")
            return

        # Generate SRT file
        output_filename = args.output if args.output else f"{video_id}.srt"
        print(f"\nGenerowanie pliku SRT: {output_filename}...")
        write_srt(all_segments, output_filename)
        print(f"Sukces: Zapisano {output_filename}")
        print(f"Łącznie segmentów: {len(all_segments)}")

    except Exception as e:
        print(f"{e}")
        sys.exit(1)
    finally:
        # Cleanup temporary files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\nUsunięto pliki tymczasowe: {temp_dir}")
            except Exception as e:
                print(f"\nOstrzeżenie: Nie można usunąć plików tymczasowych: {e}")


if __name__ == '__main__':
    main()
