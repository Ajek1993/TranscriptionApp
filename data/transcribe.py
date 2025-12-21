#!/usr/bin/env python3
"""
YouTube to SRT Transcription Tool
MVP Stage 1: Validate and download audio from YouTube
MVP Stage 2: Split audio into chunks (~30 minutes each)
MVP Stage 3: Transcribe audio using faster-whisper
MVP Stage 4: Merge segments and generate SRT file
MVP Stage 5: Complete pipeline with CLI and automatic cleanup
MVP Stage 6: TTS dubbing with Edge TTS
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
import threading
import logging
from queue import Queue

# Supported video formats for local files
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov'}

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


# ===== LOGGING SETUP =====

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===== OUTPUT MANAGEMENT =====

class OutputManager:
    """Centralized output management for user-facing messages."""

    @staticmethod
    def stage_header(stage_num: int, stage_name: str) -> None:
        """Print stage header: === Etap X: NAME ==="""
        print(f"\n=== Etap {stage_num}: {stage_name} ===")

    @staticmethod
    def info(message: str, use_tqdm_safe: bool = False) -> None:
        """Print info message (use tqdm.write if progress bars active)."""
        if use_tqdm_safe:
            tqdm.write(message)
        else:
            print(message)

    @staticmethod
    def success(message: str) -> None:
        """Print success message with checkmark."""
        print(f"\n[OK] {message}")

    @staticmethod
    def warning(message: str, use_tqdm_safe: bool = False) -> None:
        """Print warning message."""
        msg = f"Ostrzeżenie: {message}"
        if use_tqdm_safe:
            tqdm.write(msg)
        else:
            print(msg)

    @staticmethod
    def error(message: str) -> None:
        """Print error message."""
        print(f"Błąd: {message}")

    @staticmethod
    def detail(message: str, use_tqdm_safe: bool = False) -> None:
        """Print detailed info (indented, secondary importance)."""
        msg = f"  {message}"
        if use_tqdm_safe:
            tqdm.write(msg)
        else:
            print(msg)

    @staticmethod
    def mode_header(mode_name: str, details: dict = None) -> None:
        """Print mode header with configuration details."""
        print(f"\n=== {mode_name} ===")
        if details:
            for key, value in details.items():
                print(f"{key}: {value}")


# ===== COMMAND BUILDERS =====

def build_ffprobe_audio_info_cmd(file_path: str) -> list:
    """Build ffprobe command to get audio info (channels, sample_rate)."""
    return [
        'ffprobe', '-v', 'error', '-show_entries',
        'stream=channels,sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1:noescapes=1',
        str(file_path)
    ]


def build_ffprobe_video_info_cmd(file_path: str) -> list:
    """Build ffprobe command to get video info (width, height, codec)."""
    return [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]


def build_ffprobe_duration_cmd(file_path: str) -> list:
    """Build ffprobe command to get file duration."""
    return [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]


def build_ffmpeg_audio_extraction_cmd(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1
) -> list:
    """Build ffmpeg command to extract audio as WAV."""
    return [
        'ffmpeg',
        '-i', str(input_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', str(sample_rate),  # Sample rate
        '-ac', str(channels),  # Channels
        '-y',  # Overwrite output
        str(output_path)
    ]


def build_ffmpeg_audio_split_cmd(
    input_path: str,
    output_path: str,
    start_time: int,
    duration: int
) -> list:
    """Build ffmpeg command to split audio into chunks."""
    return [
        'ffmpeg',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        str(output_path)
    ]


def build_ffmpeg_video_merge_cmd(
    video_path: str,
    audio_path: str,
    output_path: str
) -> list:
    """Build ffmpeg command to merge video with audio track."""
    return [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        str(output_path)
    ]


def build_ffmpeg_subtitle_burn_cmd(
    video_path: str,
    srt_path: str,
    output_path: str,
    subtitle_style: str
) -> list:
    """Build ffmpeg command to burn subtitles into video."""
    # Convert paths to absolute and escape for ffmpeg
    srt_path_abs = str(Path(srt_path).resolve())
    srt_path_filter = srt_path_abs.replace('\\', '/').replace(':', '\\:')

    # Build subtitles filter with custom style
    subtitles_filter = f"subtitles='{srt_path_filter}':force_style='{subtitle_style}':charenc=UTF-8"

    return [
        'ffmpeg', '-y',
        '-i', str(Path(video_path).resolve()),
        '-vf', subtitles_filter,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'copy',
        str(output_path)
    ]


def build_ytdlp_audio_download_cmd(url: str, output_file: str) -> list:
    """Build yt-dlp command to download audio only."""
    return [
        'yt-dlp',
        '-f', 'bestaudio/best',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', str(output_file),
        url
    ]


def build_ytdlp_video_download_cmd(url: str, output_file: str, quality: str = "1080") -> list:
    """Build yt-dlp command to download video."""
    format_str = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"

    return [
        'yt-dlp',
        '-f', format_str,
        '--merge-output-format', 'mp4',
        '-o', str(output_file),
        url
    ]


# ===== DUBBING PIPELINE =====

def run_dubbing_pipeline(
    segments: List[Tuple[int, int, str]],
    audio_path: str,
    original_video_path: str,
    input_stem: str,
    args,
    temp_dir: str
) -> Tuple[bool, str]:
    """
    Run complete TTS dubbing pipeline.

    Handles:
    - TTS generation for all segments
    - TTS track combination
    - Audio mixing
    - Video creation (if not audio-only)

    Returns:
        Tuple of (success, error_msg)
    """
    OutputManager.stage_header(4, "Generowanie dubbingu TTS")

    # For full dubbing, we need video file
    if args.dub and not original_video_path:
        return False, "Błąd: Dubbing wymaga pliku wideo"

    # Create TTS directory in temp
    tts_dir = Path(temp_dir) / "tts"
    tts_dir.mkdir(exist_ok=True)

    # Generate TTS for each segment
    success, message, tts_files = generate_tts_segments(
        segments,
        str(tts_dir),
        voice=args.tts_voice,
        engine=args.tts_engine,
        coqui_model=args.coqui_model,
        coqui_speaker=args.coqui_speaker,
        speaker_wav=audio_path
    )

    if not success:
        return False, message

    print(message)

    # Get total duration of original audio
    success, total_duration_ms = get_audio_duration_ms(audio_path)
    if not success:
        return False, "Błąd: Nie można odczytać długości audio"

    # Combine TTS segments into single track
    tts_combined_path = Path(temp_dir) / "tts_combined.wav"
    success, message = create_tts_audio_track(
        tts_files,
        total_duration_ms,
        str(tts_combined_path)
    )

    if not success:
        return False, message

    print(message)

    # Mix original audio with TTS
    mixed_audio_path = Path(temp_dir) / "mixed_audio.wav"
    success, message = mix_audio_tracks(
        audio_path,
        str(tts_combined_path),
        str(mixed_audio_path),
        original_volume=args.original_volume,
        tts_volume=args.tts_volume
    )

    if not success:
        return False, message

    print(message)

    # Audio-only mode: just save the mixed audio
    if args.dub_audio_only:
        if args.dub_output:
            dubbed_audio_filename = args.dub_output
            if not dubbed_audio_filename.endswith('.wav'):
                dubbed_audio_filename += '.wav'
        else:
            dubbed_audio_filename = f"{input_stem}_dubbed.wav"

        # Copy mixed audio to output
        shutil.copy2(str(mixed_audio_path), dubbed_audio_filename)

        OutputManager.success("Dubbing audio zakończony pomyślnie!")
        OutputManager.success(f"Audio z dubbingiem: {dubbed_audio_filename}")
    else:
        # Full video dubbing mode
        if args.dub_output:
            dubbed_video_filename = args.dub_output
            if not dubbed_video_filename.endswith('.mp4'):
                dubbed_video_filename += '.mp4'
        else:
            dubbed_video_filename = f"{input_stem}_dubbed.mp4"

        success, message = create_dubbed_video(
            original_video_path,
            str(mixed_audio_path),
            dubbed_video_filename
        )

        if not success:
            return False, message

        OutputManager.info(message)
        OutputManager.success("Dubbing zakończony pomyślnie!")
        OutputManager.success(f"Wideo z dubbingiem: {dubbed_video_filename}")

    return True, ""


# ===== SRT GENERATION =====

def generate_srt_output(segments: List[Tuple[int, int, str]], input_stem: str, args, temp_dir: str) -> Tuple[bool, str, str]:
    """
    Generate SRT file from segments.

    Returns:
        Tuple of (success, error_msg, srt_filename)
    """
    OutputManager.stage_header(3, "Generowanie pliku SRT")
    print(f"Łącznie segmentów: {len(segments)}")

    # Determine output filename
    if args.output:
        # User explicitly specified output - always honor it
        srt_filename = args.output
        if not srt_filename.endswith('.srt'):
            srt_filename += '.srt'
    elif args.burn_subtitles:
        # Burning subtitles without explicit output - use temp directory
        srt_filename = str(Path(temp_dir) / f"{input_stem}.srt")
    else:
        # Normal case: write to current directory
        srt_filename = f"{input_stem}.srt"

    # Write SRT file
    success, message = write_srt(segments, srt_filename)
    if not success:
        return False, message, ""

    print(message)
    return True, "", srt_filename


# ===== SUBTITLE BURNING PIPELINE =====

def burn_subtitles_to_video_pipeline(original_video_path: str, srt_filename: str, input_stem: str, args, temp_dir: str) -> Tuple[bool, str]:
    """
    Handle complete subtitle burning pipeline.

    Returns:
        Tuple of (success, error_msg)
    """
    OutputManager.stage_header(5, "Wgrywanie napisów do wideo")

    # Need video file
    if not original_video_path:
        # Try to download video if we have URL
        if args.url:
            print("Pobieranie wideo dla napisów...")
            success, message, video_path = download_video(
                args.url,
                output_dir=temp_dir,
                quality=args.video_quality
            )
            if not success:
                return False, message

            original_video_path = video_path
        else:
            return False, "Błąd: --burn-subtitles wymaga wideo (YouTube URL lub --local)"

    # Determine output filename
    if args.burn_output:
        burn_output_filename = args.burn_output
        if not burn_output_filename.endswith('.mp4'):
            burn_output_filename += '.mp4'
    else:
        burn_output_filename = f"{input_stem}_subtitled.mp4"

    # Burn subtitles
    success, message = burn_subtitles_to_video(
        original_video_path,
        srt_filename,
        burn_output_filename,
        subtitle_style=args.subtitle_style
    )

    if not success:
        return False, message

    print(message)
    OutputManager.success(f"Wideo z napisami: {burn_output_filename}")
    return True, ""


# ===== TRANSCRIPTION PIPELINE =====

def _transcribe_all_chunks(chunk_paths: List[str], args) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transcribe all chunks with progress tracking.

    Returns:
        Tuple of (success, error_msg, all_segments)
    """
    all_segments = []
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
                leave=False,
                total=0,
                bar_format='{desc}: {postfix}'
            )

            # Transcribe with progress callback
            success, message, segments = transcribe_chunk(
                chunk_path,
                model_size=args.model,
                language=args.language,
                engine=args.engine,
                segment_progress_bar=segment_pbar,
                timeout_seconds=args.transcription_timeout,
                # WhisperX parameters
                whisperx_align=getattr(args, 'whisperx_align', False),
                whisperx_diarize=getattr(args, 'whisperx_diarize', False),
                whisperx_min_speakers=getattr(args, 'whisperx_min_speakers', None),
                whisperx_max_speakers=getattr(args, 'whisperx_max_speakers', None),
                hf_token=getattr(args, 'hf_token', None)
            )

            # Close segment progress bar
            segment_pbar.close()

            if not success:
                return False, message, []

            # Split long segments for better dubbing sync
            if args.dub or args.dub_audio_only:
                original_count = len(segments)

                segments = split_long_segments(
                    segments,
                    max_duration_ms=args.max_segment_duration * 1000,
                    max_words=args.max_segment_words
                )

                if len(segments) != original_count:
                    tqdm.write(f"Podzielono długie segmenty: {original_count} → {len(segments)} segmentów")

                # Fill gaps in timestamps if requested
                if args.fill_gaps:
                    segments = fill_timestamp_gaps(
                        segments,
                        max_gap_to_fill_ms=args.max_gap_fill,
                        min_pause_ms=args.min_pause
                    )
                    tqdm.write(f"Wypełniono małe luki w timestampach")

            # Adjust timestamps and add to all_segments
            adjusted_segments = [(start_ms + current_offset_ms, end_ms + current_offset_ms, text)
                                for start_ms, end_ms, text in segments]
            all_segments.extend(adjusted_segments)

            # Update offset for next chunk (based on last segment end time)
            if segments:
                last_segment_end = segments[-1][1]
                current_offset_ms += last_segment_end

            # Update chunk progress bar with segment count
            chunk_pbar.set_postfix_str(f"{len(segments)} segments, total: {len(all_segments)}")
            chunk_pbar.update(1)

    return True, "", all_segments


def _translate_segments_if_requested(segments: List[Tuple[int, int, str]], args) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Handle translation step if requested.

    Returns:
        Tuple of (success, error_msg, segments)
    """
    if not args.translate:
        return True, "", segments

    OutputManager.stage_header(2, "Tłumaczenie")

    # Parse translation direction
    src_lang, tgt_lang = args.translate.split('-')

    # Validate source language matches transcription language
    if args.language and args.language != src_lang:
        print(f"Ostrzeżenie: Język transkrypcji ({args.language}) różni się od źródłowego języka tłumaczenia ({src_lang})")

    success, message, translated_segments = translate_segments(
        segments,
        source_lang=src_lang,
        target_lang=tgt_lang
    )

    if not success:
        return False, message, []

    print(message)
    return True, "", translated_segments


def run_transcription_pipeline(audio_path: str, args, temp_dir: str) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Run complete transcription pipeline.

    Handles:
    - Audio splitting
    - Chunk transcription
    - Segment splitting (for dubbing)
    - Gap filling
    - Translation (if requested)

    Returns:
        Tuple of (success, error_msg, segments)
    """
    # Stage 2: Split audio into chunks
    success, message, chunk_paths = split_audio(audio_path, output_dir=temp_dir)
    if not success:
        return False, message, []

    print(message)

    if args.only_chunk:
        # For --only-chunk, copy chunks to current directory before cleanup
        for chunk_path in chunk_paths:
            chunk_file = Path(chunk_path)
            dest_path = Path.cwd() / chunk_file.name
            shutil.copy2(chunk_path, dest_path)
            print(f"Chunk skopiowany do: {dest_path}")
        return False, "ONLY_CHUNK_MODE", []

    # Stage 3: Transcribe all chunks
    success, message, all_segments = _transcribe_all_chunks(chunk_paths, args)
    if not success:
        return False, message, []

    # Translation step (if requested)
    success, message, all_segments = _translate_segments_if_requested(all_segments, args)
    if not success:
        return False, message, []

    if args.only_transcribe:
        print(f"\nŁącznie transkrybowanych segmentów: {len(all_segments)}")
        return False, "ONLY_TRANSCRIBE_MODE", []

    return True, "", all_segments


# ===== INPUT SOURCE PROCESSING =====

def _process_local_file(file_path: str, temp_dir: str) -> Tuple[bool, str, str, str, str]:
    """
    Process local video file.

    Returns:
        Tuple of (success, error_msg, audio_path, video_path, input_stem)
    """
    is_valid, error_msg = validate_video_file(file_path)
    if not is_valid:
        return False, error_msg, "", "", ""

    video_path = file_path

    # Extract audio from local video
    success, message, audio_path = extract_audio_from_video(file_path, output_dir=temp_dir)
    if not success:
        return False, message, "", "", ""

    print(message)
    input_stem = Path(file_path).stem

    return True, "", audio_path, video_path, input_stem


def _process_youtube_url(url: str, temp_dir: str, args) -> Tuple[bool, str, str, str, str]:
    """
    Process YouTube URL.

    Returns:
        Tuple of (success, error_msg, audio_path, video_path, input_stem)
    """
    # Validate URL
    if not validate_youtube_url(url):
        return False, "Błąd: Niepoprawny URL YouTube. Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID", "", "", ""

    # Extract video ID for naming
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
    input_stem = video_id_match.group(1) if video_id_match else "output"

    video_path = ""

    # If dubbing is requested, download full video
    if args.dub and not args.dub_audio_only:
        print(f"\n=== Dubbing włączony: Pobieranie pełnego wideo ===")
        success, message, video_path = download_video(url, output_dir=temp_dir, quality=args.video_quality)
        if not success:
            return False, message, "", "", ""

        print(message)

        # Extract audio from downloaded video
        success, message, audio_path = extract_audio_from_video(video_path, output_dir=temp_dir)
        if not success:
            return False, message, "", "", ""

        print(message)
    elif args.dub_audio_only:
        # Audio-only dubbing mode - just download audio
        print(f"\n=== Dubbing audio-only włączony: Pobieranie audio ===")
        success, message, audio_path = download_audio(url, output_dir=temp_dir)
        if not success:
            return False, message, "", "", ""

        print(message)
    else:
        # Only audio needed - download audio only
        success, message, audio_path = download_audio(url, output_dir=temp_dir)
        if not success:
            return False, message, "", "", ""

        print(message)

    return True, "", audio_path, video_path, input_stem


def process_input_source(args, temp_dir: str) -> Tuple[bool, str, str, str, str]:
    """
    Process input source (local file or YouTube URL).

    Returns:
        Tuple of (success, error_msg, audio_path, video_path, input_stem)
    """
    if args.local:
        return _process_local_file(args.local, temp_dir)
    else:
        return _process_youtube_url(args.url, temp_dir, args)


# ===== MODE HANDLERS =====

def handle_video_download_mode(args) -> int:
    """Handle --download mode (download video only, no transcription)."""
    if not validate_youtube_url(args.download):
        OutputManager.error("Niepoprawny URL YouTube.")
        return 1

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(deps_msg)
        return 1

    OutputManager.mode_header("Tryb pobierania wideo", {
        "Pobieranie z": args.download,
        "Jakość": f"{args.video_quality}p"
    })

    # Download to current directory
    success, message, video_path = download_video(
        args.download,
        output_dir=".",
        quality=args.video_quality
    )

    if not success:
        print(message)
        return 1

    OutputManager.info(message)
    OutputManager.success(f"Wideo pobrane: {video_path}")
    return 0


def handle_audio_download_mode(args) -> int:
    """Handle --download-audio-only mode (download audio only, no transcription)."""
    if not validate_youtube_url(args.download_audio_only):
        OutputManager.error("Niepoprawny URL YouTube.")
        return 1

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(deps_msg)
        return 1

    OutputManager.mode_header("Tryb pobierania audio", {
        "Pobieranie z": args.download_audio_only,
        "Jakość": args.audio_quality
    })

    # Download to current directory
    success, message, audio_path = download_audio(
        args.download_audio_only,
        output_dir="."
    )

    if not success:
        print(message)
        return 1

    OutputManager.info(message)
    OutputManager.success(f"Audio pobrane: {audio_path}")
    return 0


def handle_test_merge_mode(args) -> int:
    """Handle --test-merge mode (test SRT generation with hardcoded data)."""
    print("Test generowania SRT z przykładowymi danymi...")

    # Hardcoded test segments simulating 2 chunks
    chunk1_segments = [
        (0, 5000, "Witam w przykładowym filmie o transkrypcji."),
        (5500, 12000, "To jest pierwszy segment z pierwszego chunka."),
        (12500, 20000, "Tutaj sprawdzamy czy polskie znaki działają: ąćęłńóśźż."),
        (20500, 30000, "Koniec pierwszego chunka.")
    ]

    chunk2_segments = [
        (0, 8000, "To jest początek drugiego chunka."),
        (8500, 18000, "Timestampy powinny być przesunięte o 30 sekund."),
        (18500, 28000, "Sprawdzamy merge i scalanie segmentów."),
        (28500, 35000, "Koniec testu.")
    ]

    # Simulate merging process
    all_segments = []
    chunk_offsets = [0, 30000]

    for start_ms, end_ms, text in chunk1_segments:
        all_segments.append((start_ms + chunk_offsets[0], end_ms + chunk_offsets[0], text))

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
            print(content[:500])
            if len(content) > 500:
                print("...")
        print("-" * 60)
        print(f"\nOtwórz plik '{test_output}' w VLC lub innym odtwarzaczu aby sprawdzić napisy.")
        return 0
    else:
        print(message)
        return 1


# ===== VALIDATION FUNCTIONS =====


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
        OutputManager.info(f"Pobieranie audio z YouTube... ({url})")

        cmd = build_ytdlp_audio_download_cmd(url, str(audio_file))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd yt-dlp"
            if "private" in error_msg.lower() or "not available" in error_msg.lower():
                return False, "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie.", ""
            else:
                return False, f"Błąd yt-dlp: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        OutputManager.info(f"Audio pobrane: {audio_file}")

        # Verify audio format with ffprobe
        probe_cmd = build_ffprobe_audio_info_cmd(audio_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            channels = int(output_info[0]) if len(output_info) > 0 else 0
            sample_rate = int(output_info[1]) if len(output_info) > 1 else 0
            OutputManager.detail(f"Format audio: {channels} kanał(y), {sample_rate} Hz")

        return True, f"Audio pobrane pomyślnie: {audio_file}", str(audio_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Pobieranie przerwane (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy pobieraniu: {str(e)}", ""

def download_video(url: str, output_dir: str = ".", quality: str = "1080") -> Tuple[bool, str, str]:
    """
    Download video from YouTube in specified quality.

    Args:
        url: YouTube URL
        output_dir: Directory to save the video file
        quality: Preferred video quality (default: "1080")

    Returns:
        Tuple of (success: bool, message: str, video_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract video ID for naming
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
    if not video_id_match:
        return False, "Błąd: Nie udało się wyodrębnić ID wideo z URL", ""

    video_id = video_id_match.group(1)
    video_file = output_path / f"{video_id}.mp4"

    try:
        print(f"Pobieranie wideo z YouTube w jakości {quality}p... ({url})")

        cmd = build_ytdlp_video_download_cmd(url, str(video_file), quality)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd yt-dlp"
            if "private" in error_msg.lower() or "not available" in error_msg.lower():
                return False, "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie.", ""
            else:
                return False, f"Błąd yt-dlp: {error_msg}", ""

        if not video_file.exists():
            return False, f"Błąd: Plik wideo nie został utworzony: {video_file}", ""

        # Get video info with ffprobe
        probe_cmd = build_ffprobe_video_info_cmd(video_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            if len(output_info) >= 3:
                width = output_info[0]
                height = output_info[1]
                codec = output_info[2]
                print(f"Wideo pobrane: {width}x{height}, codec: {codec}")
            else:
                print(f"Wideo pobrane: {video_file}")
        else:
            print(f"Wideo pobrane: {video_file}")

        return True, f"Wideo pobrane pomyślnie: {video_file}", str(video_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Pobieranie przerwane (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy pobieraniu wideo: {str(e)}", ""



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

        cmd = build_ffmpeg_audio_extraction_cmd(video_path, audio_file)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd ffmpeg"
            return False, f"Błąd ffmpeg: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        print(f"Audio wyekstrahowane: {audio_file}")

        # Verify audio format with ffprobe
        probe_cmd = build_ffprobe_audio_info_cmd(audio_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            channels = int(output_info[0]) if len(output_info) > 0 else 0
            sample_rate = int(output_info[1]) if len(output_info) > 1 else 0
            OutputManager.detail(f"Format audio: {channels} kanał(y), {sample_rate} Hz")

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



def _run_transcription_with_timeout(
    model, audio_path: str, language: str,
    timeout_seconds: int, segment_progress_bar, vad_parameters: dict
) -> Tuple[bool, str, List[Tuple[int, int, str]], object]:
    """Run model.transcribe() with timeout protection using threading."""
    import time

    if timeout_seconds <= 0:
        # No timeout - run directly
        segments_generator, info = model.transcribe(
            audio_path, language=language, word_timestamps=True,
            vad_filter=True, vad_parameters=vad_parameters
        )
        segments = []
        for segment in segments_generator:
            segments.append((int(segment.start * 1000), int(segment.end * 1000), segment.text.strip()))
            if segment_progress_bar:
                segment_progress_bar.set_postfix_str(f"{len(segments)} segments")
        return True, "", segments, info

    # Run with timeout
    result_queue = Queue()
    exception_queue = Queue()

    def transcribe_thread():
        try:
            start_time = time.time()
            last_log = start_time

            segments_generator, info = model.transcribe(
                audio_path, language=language, word_timestamps=True,
                vad_filter=True, vad_parameters=vad_parameters
            )

            segments = []
            for segment in segments_generator:
                segments.append((int(segment.start * 1000), int(segment.end * 1000), segment.text.strip()))

                # Log progress every 30 seconds
                current_time = time.time()
                elapsed = current_time - start_time
                if current_time - last_log >= 30:
                    tqdm.write(f"  Progress: {len(segments)} segments, {elapsed:.0f}s elapsed")
                    last_log = current_time

                if segment_progress_bar:
                    segment_progress_bar.set_postfix_str(f"{len(segments)} seg, {elapsed:.0f}s")

            result_queue.put(("success", segments, info))
        except Exception as e:
            exception_queue.put(e)

    thread = threading.Thread(target=transcribe_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Timeout occurred
        error_msg = f"TIMEOUT: Transcription exceeded {timeout_seconds}s ({timeout_seconds/60:.1f} min)."
        error_msg += "\nSolutions:"
        error_msg += "\n  1. Use smaller model: --model base"
        error_msg += f"\n  2. Increase timeout: --transcription-timeout {timeout_seconds * 2}"
        error_msg += "\n  3. Disable timeout: --transcription-timeout 0"
        return False, error_msg, [], None

    if not exception_queue.empty():
        raise exception_queue.get()

    if not result_queue.empty():
        status, segments, info = result_queue.get()
        return True, "", segments, info

    return False, "Unknown error during transcription", [], None


# ===== PHASE 3: GPU MEMORY MONITORING =====

WHISPER_MODEL_MEMORY_REQUIREMENTS = {
    'tiny': 1, 'base': 1, 'small': 2, 'medium': 5, 'large': 10,
    'large-v2': 10, 'large-v3': 10
}


def get_gpu_memory_info() -> str:
    """Get GPU memory usage information."""
    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total_mem - allocated
            return f"GPU Memory: {allocated:.2f}GB/{total_mem:.2f}GB ({free:.2f}GB free)"
    except:
        pass
    return ""


def detect_device() -> Tuple[str, str]:
    """Detect available device with cuDNN validation."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            if cudnn_version and cudnn_version >= 90000:  # cuDNN 9.x
                device_info = f"NVIDIA GPU ({gpu_name}, CUDA {cuda_version}, cuDNN {cudnn_version // 1000}.{(cudnn_version % 1000) // 100})"
                return "cuda", device_info
            else:
                tqdm.write(f"Warning: cuDNN {cudnn_version} too old. Need >=9.0 for CUDA 12.8. Falling back to CPU")
                return "cpu", f"CPU (cuDNN incompatible: {cudnn_version})"
    except ImportError:
        pass
    except Exception as e:
        tqdm.write(f"Warning: GPU detection failed: {e}")

    return "cpu", "CPU"


def transcribe_with_whisper(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z OpenAI Whisper (automatyczne GPU/CUDA).

    Whisper automatycznie wykorzystuje GPU jeśli dostępne.
    """
    try:
        import whisper
    except ImportError:
        return False, "Błąd: whisper nie jest zainstalowany. Zainstaluj: pip install openai-whisper", []

    # Detekcja urządzenia (GPU preferred)
    device, device_info = detect_device()
    tqdm.write(f"Używane urządzenie: {device_info}")

    # Ładowanie modelu
    tqdm.write(f"Ładowanie modelu OpenAI Whisper {model_size}...")
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        if device == "cuda":
            tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU. Błąd: {e}")
            device = "cpu"
            model = whisper.load_model(model_size, device="cpu")
        else:
            return False, f"Błąd podczas ładowania modelu whisper: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja: {Path(wav_path).name}...")

    # OpenAI Whisper nie używa timeout wrapper (prostsza implementacja)
    try:
        result = model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda")
        )
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []

    # Parsowanie segmentów
    segments = []
    for segment in result["segments"]:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        text = segment["text"].strip()
        segments.append((start_ms, end_ms, text))

        if segment_progress_bar:
            segment_progress_bar.set_postfix_str(f"{len(segments)} segments")

    tqdm.write(f"Wykryty język: {result['language']}")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments


def transcribe_with_faster_whisper(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z Faster-Whisper (zawsze CPU).

    Faster-Whisper wymusza użycie CPU ze względu na problemy z CTranslate2.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return False, "Błąd: faster-whisper nie jest zainstalowany. Zainstaluj: pip install faster-whisper", []

    # WYMUSZENIE CPU dla Faster-Whisper
    device = "cpu"
    device_info = "CPU (faster-whisper - wymuszony CPU)"
    tqdm.write(f"Używane urządzenie: {device_info}")
    tqdm.write("  INFO: Faster-Whisper używa CPU ze względu na kompatybilność CTranslate2")

    # Inicjalizacja modelu (zawsze CPU)
    tqdm.write(f"Ładowanie modelu {model_size}...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        return False, f"Błąd podczas ładowania modelu faster-whisper: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja: {Path(wav_path).name}...")

    if timeout_seconds > 0:
        tqdm.write(f"Timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} min)")
    else:
        tqdm.write("Timeout: disabled")

    vad_params = dict(
        threshold=0.5, min_speech_duration_ms=250, max_speech_duration_s=15,
        min_silence_duration_ms=500, speech_pad_ms=400
    )

    # Transcribe with timeout
    success, error_msg, segments, info = _run_transcription_with_timeout(
        model, str(wav_path), language, timeout_seconds, segment_progress_bar, vad_params
    )

    if not success:
        return False, error_msg, []

    if info:
        tqdm.write(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments


def transcribe_with_whisperx(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int,
    align: bool = False,
    diarize: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    hf_token: str = None
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z WhisperX (GPU/CPU, alignment, diarization).

    WhisperX oferuje:
    - Lepszą dokładność timestampów
    - Word-level alignment
    - Speaker diarization (rozpoznawanie mówców)
    """
    try:
        import whisperx
        import torch
    except ImportError:
        return False, "Błąd: whisperx nie jest zainstalowany. Zainstaluj: pip install whisperx", []

    # Detekcja urządzenia
    device, device_info = detect_device()
    tqdm.write(f"Używane urządzenie: {device_info}")

    # Compute type
    compute_type = "float16" if device == "cuda" else "int8"

    # Ładowanie modelu
    tqdm.write(f"Ładowanie modelu WhisperX {model_size}...")
    try:
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language
        )
    except Exception as e:
        if device == "cuda":
            tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU. Błąd: {e}")
            device = "cpu"
            compute_type = "int8"
            model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
        else:
            return False, f"Błąd podczas ładowania modelu WhisperX: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja WhisperX: {Path(wav_path).name}...")

    # Załaduj audio
    try:
        audio = whisperx.load_audio(str(wav_path))
    except Exception as e:
        return False, f"Błąd podczas ładowania audio: {str(e)}", []

    # Transkrypcja
    try:
        result = model.transcribe(audio, batch_size=16)
    except Exception as e:
        return False, f"Błąd podczas transkrypcji WhisperX: {str(e)}", []

    # Word-level alignment (opcjonalnie)
    if align:
        tqdm.write("Wykonywanie word-level alignment...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
        except Exception as e:
            tqdm.write(f"Ostrzeżenie: Alignment nie powiódł się: {e}")

    # Speaker diarization (opcjonalnie)
    if diarize:
        if not hf_token:
            tqdm.write("Ostrzeżenie: Speaker diarization wymaga HuggingFace token (--hf-token)")
        else:
            tqdm.write("Wykonywanie speaker diarization...")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                tqdm.write(f"Ostrzeżenie: Diarization nie powiódł się: {e}")

    # Konwersja segmentów do formatu (start_ms, end_ms, text)
    segments = []
    for segment in result.get("segments", []):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        text = segment["text"].strip()

        # Dodaj speaker info jeśli dostępne
        if "speaker" in segment:
            text = f"[{segment['speaker']}] {text}"

        segments.append((start_ms, end_ms, text))

        if segment_progress_bar:
            segment_progress_bar.set_postfix_str(f"{len(segments)} segments")

    tqdm.write(f"Wykryty język: {result.get('language', language)}")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments


def transcribe_chunk(
    wav_path: str,
    model_size: str = "base",
    language: str = "pl",
    engine: str = "whisper",
    segment_progress_bar: tqdm = None,
    timeout_seconds: int = 1800,
    # WhisperX options
    whisperx_align: bool = False,
    whisperx_diarize: bool = False,
    whisperx_min_speakers: int = None,
    whisperx_max_speakers: int = None,
    hf_token: str = None
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja pliku WAV przy użyciu wybranego silnika.

    Args:
        wav_path: Ścieżka do pliku WAV
        model_size: Rozmiar modelu (tiny, base, small, medium, large)
        language: Kod języka (pl, en, etc.)
        engine: Silnik transkrypcji ('whisper', 'faster-whisper', 'whisperx')
        segment_progress_bar: Progress bar dla segmentów
        timeout_seconds: Timeout dla transkrypcji (0 = bez limitu)
        whisperx_align: Włącz word-level alignment (tylko WhisperX)
        whisperx_diarize: Włącz speaker diarization (tylko WhisperX)
        whisperx_min_speakers: Min liczba mówców (tylko WhisperX)
        whisperx_max_speakers: Max liczba mówców (tylko WhisperX)
        hf_token: HuggingFace token (tylko WhisperX diarization)

    Returns:
        Tuple (success: bool, message: str, segments: List[(start_ms, end_ms, text)])
    """
    try:
        # Sprawdź czy plik istnieje
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"Błąd: Plik audio nie istnieje: {wav_path}", []

        # Wybór silnika transkrypcji
        if engine == "whisper":
            return transcribe_with_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds
            )

        elif engine == "faster-whisper":
            return transcribe_with_faster_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds
            )

        elif engine == "whisperx":
            return transcribe_with_whisperx(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                align=whisperx_align,
                diarize=whisperx_diarize,
                min_speakers=whisperx_min_speakers,
                max_speakers=whisperx_max_speakers,
                hf_token=hf_token
            )

        else:
            return False, f"Błąd: Nieobsługiwany silnik transkrypcji: {engine}", []

    except ImportError as e:
        return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}", []
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []


def split_long_segments(
    segments: List[Tuple[int, int, str]],
    max_duration_ms: int = 10000,
    max_words: int = 15
) -> List[Tuple[int, int, str]]:
    """
    Split long segments into smaller ones based on duration and word count.
    
    Args:
        segments: List of (start_ms, end_ms, text) tuples
        max_duration_ms: Maximum segment duration in milliseconds
        max_words: Maximum words per segment
    
    Returns:
        List of split segments
    """
    result = []
    
    for start_ms, end_ms, text in segments:
        duration_ms = end_ms - start_ms
        words = text.split()
        
        # If segment is short enough, keep as is
        if duration_ms <= max_duration_ms and len(words) <= max_words:
            result.append((start_ms, end_ms, text))
            continue
        
        # Split long segment by sentences or words
        import re
        # Split by sentence-ending punctuation
        sentences = re.split(r'([.!?]+\s+)', text)
        
        if len(sentences) <= 1:
            # No sentence breaks, split by words
            words_per_chunk = max(1, max_words)
            word_chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
            
            time_per_word = duration_ms / len(words)
            current_time = start_ms
            
            for chunk in word_chunks:
                chunk_text = ' '.join(chunk)
                chunk_duration = int(len(chunk) * time_per_word)
                chunk_end = min(current_time + chunk_duration, end_ms)
                
                result.append((current_time, chunk_end, chunk_text))
                current_time = chunk_end
        else:
            # Split by sentences
            current_time = start_ms
            time_per_char = duration_ms / len(text)
            
            for i in range(0, len(sentences), 2):  # Sentences + punctuation pairs
                if i >= len(sentences):
                    break
                    
                sentence = sentences[i]
                punct = sentences[i + 1] if i + 1 < len(sentences) else ''
                full_sentence = sentence + punct
                
                if not full_sentence.strip():
                    continue
                
                sentence_duration = int(len(full_sentence) * time_per_char)
                sentence_end = min(current_time + sentence_duration, end_ms)
                
                result.append((current_time, sentence_end, full_sentence.strip()))
                current_time = sentence_end
    
    return result


def fill_timestamp_gaps(
    segments: List[Tuple[int, int, str]],
    max_gap_to_fill_ms: int = 2000,
    min_pause_ms: int = 200
) -> List[Tuple[int, int, str]]:
    """
    Fill only small gaps in timestamps by extending segments.
    Large gaps (silence/music) are preserved.
    
    Args:
        segments: List of (start_ms, end_ms, text) tuples
        max_gap_to_fill_ms: Maximum gap size to fill (larger gaps preserved)
        min_pause_ms: Minimum pause to leave between segments
    
    Returns:
        List of segments with small gaps filled
    """
    if not segments:
        return segments
    
    result = []
    
    for i, (start_ms, end_ms, text) in enumerate(segments):
        if i < len(segments) - 1:
            next_start_ms = segments[i + 1][0]
            gap_ms = next_start_ms - end_ms
            
            # Only fill small gaps
            if 0 < gap_ms <= max_gap_to_fill_ms:
                new_end_ms = next_start_ms - min_pause_ms
                result.append((start_ms, new_end_ms, text))
            else:
                # Large gap - preserve it
                result.append((start_ms, end_ms, text))
        else:
            result.append((start_ms, end_ms, text))
    
    return result




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


async def generate_tts_for_segment(
    text: str,
    output_path: str,
    voice: str = "pl-PL-MarekNeural",
    rate: str = "+0%"
) -> Tuple[bool, str, float]:
    """
    Generate TTS audio for a single text segment with speed control (Edge TTS).

    Args:
        text: Text to convert to speech
        output_path: Path to save the MP3 file
        voice: Voice name (default: pl-PL-MarekNeural)
        rate: Speech rate adjustment (e.g., "+0%", "+20%", "+50%")

    Returns:
        Tuple of (success: bool, message: str, duration_seconds: float)
    """
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)

        # Get duration of generated audio
        success, duration_sec = get_audio_duration(output_path)
        if not success:
            return False, "Błąd: Nie można odczytać długości wygenerowanego TTS", 0.0

        return True, "TTS wygenerowany", duration_sec

    except Exception as e:
        return False, f"Błąd generowania TTS: {str(e)}", 0.0


def generate_tts_coqui_for_segment(
    text: str,
    output_path: str,
    model_name: str = "tts_models/pl/mai_female/vits",
    speaker: str = None,
    speaker_wav: str = None, 
    speed: float = 1.0,
    tts_instance: 'TTS' = None
) -> Tuple[bool, str, float]:
    """
    Generate TTS audio for a single text segment with Coqui TTS.

    Args:
        text: Text to convert to speech
        output_path: Path to save the MP3 file
        model_name: Coqui TTS model name (default: tts_models/pl/mai_female/vits)
        speaker: Speaker ID for multi-speaker models (optional)
        speed: Speech speed multiplier (default: 1.0, range: 0.5-2.0)
        tts_instance: Reusable TTS instance (optional, for performance)

    Returns:
        Tuple of (success: bool, message: str, duration_seconds: float)
    """
    try:
        # Generate temporary WAV file
        temp_wav = str(Path(output_path).with_suffix('.wav.tmp'))

        # Initialize TTS if not provided
        if tts_instance is None:
            tts = TTS(model_name=model_name)
        else:
            tts = tts_instance

        # ✅ JEDYNE wywołanie tts_to_file
        kwargs = {"text": text, "file_path": temp_wav, "language": "pl"}
        if "xtts" in model_name.lower():
            if speaker_wav:                    # ✅ PRIORYTET 1: WAV plik
                kwargs["speaker_wav"] = speaker_wav
            elif speaker:                      # ✅ PRIORYTET 2: nazwa speakera
                kwargs["speaker"] = speaker
            else:
                return False, "XTTS v2 wymaga speaker_wav LUB speaker", 0.0
        elif speaker:
            kwargs["speaker"] = speaker
        
        tts.tts_to_file(**kwargs)

        # Apply speed adjustment and convert to MP3 if needed
        if speed != 1.0:
            # Use ffmpeg to adjust speed and convert to MP3
            cmd = [
                'ffmpeg', '-y', '-i', temp_wav,
                '-filter:a', f'atempo={speed}',
                '-b:a', '192k',
                output_path
            ]
        else:
            # Just convert WAV to MP3
            cmd = [
                'ffmpeg', '-y', '-i', temp_wav,
                '-b:a', '192k',
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Clean up temporary WAV
        if Path(temp_wav).exists():
            Path(temp_wav).unlink()

        if result.returncode != 0:
            return False, f"Błąd konwersji audio: {result.stderr}", 0.0

        # Get duration of generated audio
        success, duration_sec = get_audio_duration(output_path)
        if not success:
            return False, "Błąd: Nie można odczytać długości wygenerowanego TTS", 0.0

        return True, "TTS wygenerowany (Coqui)", duration_sec

    except Exception as e:
        # Clean up temporary WAV on error
        if 'temp_wav' in locals() and Path(temp_wav).exists():
            Path(temp_wav).unlink()
        return False, f"Błąd generowania TTS (Coqui): {str(e)}", 0.0


def generate_tts_segments(
    segments: List[Tuple[int, int, str]],
    output_dir: str,
    voice: str = "pl-PL-MarekNeural",
    engine: str = "edge",
    coqui_model: str = "tts_models/pl/mai_female/vits",
    coqui_speaker: str = None,
    speaker_wav: str = None
) -> Tuple[bool, str, List[Tuple[int, str, float]]]:
    """
    Generate TTS for all segments with automatic speed adjustment.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        output_dir: Directory to save TTS files
        voice: Voice name (for Edge TTS)
        engine: TTS engine ('edge' or 'coqui')
        coqui_model: Coqui TTS model name (for Coqui TTS)
        coqui_speaker: Speaker ID for multi-speaker Coqui models (optional)

    Returns:
        Tuple of (success: bool, message: str, tts_files: List[(start_ms, path, duration_sec)])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tts_files = []

    # Initialize Coqui TTS instance once if using coqui engine
    coqui_tts_instance = None
    if engine == "coqui":
        print(f"Ładowanie modelu Coqui TTS: {coqui_model}...")
        try:
            coqui_tts_instance = TTS(model_name=coqui_model)
            print(f"Model Coqui TTS załadowany pomyślnie")
        except Exception as e:
            return False, f"Błąd ładowania modelu Coqui TTS: {str(e)}", []

    print(f"Generowanie TTS dla {len(segments)} segmentów (engine: {engine})...")

    with tqdm(total=len(segments), desc="Generating TTS", unit="seg") as pbar:
        for idx, (start_ms, end_ms, text) in enumerate(segments):
            # Skip empty text
            if not text.strip():
                pbar.update(1)
                continue

            slot_duration_ms = end_ms - start_ms
            slot_duration_sec = slot_duration_ms / 1000.0

            tts_file = output_path / f"tts_{idx:04d}.mp3"

            # Try generating with normal speed first
            speed = 1.0  # For Coqui TTS
            rate = "+0%"  # For Edge TTS
            max_retries = 3

            for retry in range(max_retries):
                try:
                    # Generate TTS based on engine
                    if engine == "edge":
                        success, message, tts_duration = asyncio.run(
                            generate_tts_for_segment(text, str(tts_file), voice, rate)
                        )
                    elif engine == "coqui":
                        success, message, tts_duration = generate_tts_coqui_for_segment(
                            text, str(tts_file), coqui_model, coqui_speaker, speed, coqui_tts_instance, speaker_wav=speaker_wav
                        )
                    else:
                        tqdm.write(f"Błąd: Nieznany silnik TTS: {engine}")
                        pbar.update(1)
                        break

                    if not success:
                        if retry < max_retries - 1:
                            time.sleep(0.5 * (retry + 1))  # Exponential backoff
                            continue
                        else:
                            tqdm.write(f"Ostrzeżenie: Nie udało się wygenerować TTS dla segmentu {idx}: {message}")
                            pbar.update(1)
                            break

                    # Check if TTS is too long for the slot
                    if tts_duration > slot_duration_sec * 1.5:
                        # Calculate required speed increase (max +50%)
                        speed_multiplier = min(tts_duration / slot_duration_sec, 1.5)

                        if engine == "edge":
                            rate_percent = int((speed_multiplier - 1.0) * 100)
                            rate_percent = min(rate_percent, 50)  # Cap at +50%
                            rate = f"+{rate_percent}%"
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {slot_duration_sec:.2f}s), przyspieszam do {rate}")
                        elif engine == "coqui":
                            speed = min(speed_multiplier, 1.5)  # Cap at 1.5x
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {slot_duration_sec:.2f}s), przyspieszam do {speed:.2f}x")

                        # Regenerate with adjusted speed
                        if engine == "edge":
                            success, message, tts_duration = asyncio.run(
                                generate_tts_for_segment(text, str(tts_file), voice, rate)
                            )
                        elif engine == "coqui":
                            success, message, tts_duration = generate_tts_coqui_for_segment(
                                text, str(tts_file), coqui_model, coqui_speaker, speed, coqui_tts_instance
                            )

                        if not success:
                            if retry < max_retries - 1:
                                time.sleep(0.5 * (retry + 1))
                                continue
                            else:
                                tqdm.write(f"Ostrzeżenie: Nie udało się wygenerować TTS dla segmentu {idx}: {message}")
                                pbar.update(1)
                                break

                    tts_files.append((start_ms, str(tts_file), tts_duration))
                    pbar.update(1)
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(0.5 * (retry + 1))
                    else:
                        tqdm.write(f"Ostrzeżenie: Błąd przy generowaniu TTS dla segmentu {idx}: {str(e)}")
                        pbar.update(1)
                        break

    if not tts_files:
        return False, "Błąd: Nie wygenerowano żadnych plików TTS", []

    return True, f"Wygenerowano {len(tts_files)} plików TTS", tts_files


def create_tts_audio_track(
    tts_files: List[Tuple[int, str, float]],
    total_duration_ms: int,
    output_path: str
) -> Tuple[bool, str]:
    """
    Combine TTS segments into a single audio track using concat demuxer.
    This avoids amix completely and ensures constant volume.

    Args:
        tts_files: List of (start_ms, file_path, duration_sec) tuples
        total_duration_ms: Total duration of the final track in milliseconds
        output_path: Path to save the combined audio

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not tts_files:
            return False, "Błąd: Brak plików TTS do połączenia"

        temp_dir = Path(output_path).parent
        
        # Sort segments by start time
        sorted_segments = sorted(tts_files, key=lambda x: x[0])
        
        # Build filter_complex that creates silence gaps and concatenates
        filter_parts = []
        input_args = []
        concat_inputs = []
        
        current_time_ms = 0
        
        for idx, (start_ms, file_path, duration_sec) in enumerate(sorted_segments):
            # Add input file
            input_args.extend(['-i', str(file_path)])
            
            # If there's a gap before this segment, add silence
            if start_ms > current_time_ms:
                gap_duration_sec = (start_ms - current_time_ms) / 1000.0
                silence_label = f"silence{idx}"
                filter_parts.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={gap_duration_sec}[{silence_label}]"
                )
                concat_inputs.append(f"[{silence_label}]")
            
            # Add this segment (convert to stereo if needed)
            segment_label = f"seg{idx}"
            filter_parts.append(
                f"[{idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[{segment_label}]"
            )
            concat_inputs.append(f"[{segment_label}]")
            
            # Update current time
            segment_duration_ms = int(duration_sec * 1000)
            current_time_ms = start_ms + segment_duration_ms
        
        # Add final silence to reach total duration
        if current_time_ms < total_duration_ms:
            final_gap_sec = (total_duration_ms - current_time_ms) / 1000.0
            filter_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={final_gap_sec}[final_silence]"
            )
            concat_inputs.append("[final_silence]")
        
        # Concatenate all parts
        concat_input_str = ''.join(concat_inputs)
        filter_parts.append(
            f"{concat_input_str}concat=n={len(concat_inputs)}:v=0:a=1[out]"
        )
        
        filter_complex = ';'.join(filter_parts)
        
        print(f"Łączenie {len(sorted_segments)} segmentów TTS...")
        
        cmd = [
            'ffmpeg', '-y',
            *input_args,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-ar', '44100',
            '-ac', '2',
            '-c:a', 'pcm_s16le',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            return False, f"Błąd ffmpeg przy łączeniu TTS: {result.stderr}"

        if not Path(output_path).exists():
            return False, f"Błąd: Plik TTS nie został utworzony: {output_path}"

        return True, f"Ścieżka TTS utworzona: {output_path}"

    except subprocess.TimeoutExpired:
        return False, "Błąd: Łączenie TTS przerwane (timeout)"
    except Exception as e:
        return False, f"Błąd przy tworzeniu ścieżki TTS: {str(e)}"


def mix_audio_tracks(
    original_audio_path: str,
    tts_audio_path: str,
    output_path: str,
    original_volume: float = 0.2,
    tts_volume: float = 1.0
) -> Tuple[bool, str]:
    """
    Mix audio using completely manual approach without amix filter.
    """
    try:
        temp_dir = Path(output_path).parent
        
        # Step 1: Apply volume to original (create temp file)
        temp_original = temp_dir / "temp_original_vol.wav"
        cmd1 = [
            'ffmpeg', '-y',
            '-i', str(original_audio_path),
            '-af', f'volume={original_volume}',
            '-ar', '44100',
            '-ac', '2',
            '-c:a', 'pcm_s16le',
            str(temp_original)
        ]
        print(f"Skalowanie oryginalnego audio do {original_volume}...")
        subprocess.run(cmd1, capture_output=True, text=True, timeout=300, check=True)
        
        # Step 2: Apply volume to TTS (create temp file)
        temp_tts = temp_dir / "temp_tts_vol.wav"
        cmd2 = [
            'ffmpeg', '-y',
            '-i', str(tts_audio_path),
            '-af', f'volume={tts_volume}',
            '-ar', '44100',
            '-ac', '2',
            '-c:a', 'pcm_s16le',
            str(temp_tts)
        ]
        print(f"Skalowanie TTS audio do {tts_volume}...")
        subprocess.run(cmd2, capture_output=True, text=True, timeout=300, check=True)
        
        # Step 3: Mix using amerge + pan (pure mathematical addition, no AGC)
        cmd3 = [
            'ffmpeg', '-y',
            '-i', str(temp_original),
            '-i', str(temp_tts),
            '-filter_complex',
            '[0:a][1:a]amerge=inputs=2[merged];'
            '[merged]pan=stereo|c0=c0+c2|c1=c1+c3[out]',
            '-map', '[out]',
            '-ar', '44100',
            '-ac', '2',
            '-c:a', 'pcm_s16le',
            str(output_path)
        ]
        
        print(f"Mixowanie ścieżek audio...")
        result = subprocess.run(cmd3, capture_output=True, text=True, timeout=1800)
        
        # Cleanup temp files
        try:
            if temp_original.exists():
                temp_original.unlink()
            if temp_tts.exists():
                temp_tts.unlink()
        except:
            pass

        if result.returncode != 0:
            return False, f"Błąd ffmpeg przy mixowaniu audio: {result.stderr}"

        if not Path(output_path).exists():
            return False, f"Błąd: Plik zmixowanego audio nie został utworzony: {output_path}"

        return True, f"Audio zmixowane: {output_path}"

    except subprocess.CalledProcessError as e:
        return False, f"Błąd przy przetwarzaniu audio: {e}"
    except subprocess.TimeoutExpired:
        return False, "Błąd: Mixowanie audio przerwane (timeout)"
    except Exception as e:
        return False, f"Błąd przy mixowaniu audio: {str(e)}"


def create_dubbed_video(
    original_video_path: str,
    mixed_audio_path: str,
    output_video_path: str
) -> Tuple[bool, str]:
    """
    Create final dubbed video by combining original video with mixed audio.

    Args:
        original_video_path: Path to original video file
        mixed_audio_path: Path to mixed audio (original + TTS)
        output_video_path: Path to save dubbed video

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        cmd = build_ffmpeg_video_merge_cmd(original_video_path, mixed_audio_path, output_video_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            return False, f"Błąd ffmpeg przy tworzeniu wideo z dubbingiem: {result.stderr}"

        if not Path(output_video_path).exists():
            return False, f"Błąd: Wideo z dubbingiem nie zostało utworzone: {output_video_path}"

        return True, f"Wideo z dubbingiem utworzone: {output_video_path}"

    except subprocess.TimeoutExpired:
        return False, "Błąd: Tworzenie wideo przerwane (timeout)"
    except Exception as e:
        return False, f"Błąd przy tworzeniu wideo: {str(e)}"
    
def burn_subtitles_to_video(
    video_path: str,
    srt_path: str,
    output_path: str,
    subtitle_style: str = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=20"
) -> Tuple[bool, str]:
    """
    Burn (hardcode) subtitles into video permanently.
    
    Args:
        video_path: Path to input video file
        srt_path: Path to SRT subtitle file
        output_path: Path to save output video with burned subtitles
        subtitle_style: ASS subtitle style string
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate inputs
        if not Path(video_path).exists():
            return False, f"Błąd: Plik wideo nie istnieje: {video_path}"
        
        if not Path(srt_path).exists():
            return False, f"Błąd: Plik SRT nie istnieje: {srt_path}"

        print(f"Wgrywanie napisów do wideo...")
        print(f"Wideo: {video_path}")
        print(f"Napisy: {srt_path}")

        cmd = build_ffmpeg_subtitle_burn_cmd(video_path, srt_path, output_path, subtitle_style)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return False, f"Błąd ffmpeg przy wgrywaniu napisów: {result.stderr}"
        
        if not Path(output_path).exists():
            return False, f"Błąd: Wideo z napisami nie zostało utworzone: {output_path}"
        
        return True, f"Napisy wgrane do wideo: {output_path}"
    
    except subprocess.TimeoutExpired:
        return False, "Błąd: Wgrywanie napisów przerwane (timeout)"
    except Exception as e:
        return False, f"Błąd przy wgrywaniu napisów: {str(e)}"



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

    # ===== OPCJE PODSTAWOWE =====
    basic_group = parser.add_argument_group('Opcje podstawowe', 'Podstawowa konfiguracja wejścia/wyjścia')
    basic_group.add_argument('url', nargs='?', help='URL YouTube do transkrypcji')
    basic_group.add_argument('-l', '--local', type=str,
                       help='Ścieżka do lokalnego pliku wideo (MP4, MKV, AVI, MOV)')
    basic_group.add_argument('-o', '--output', type=str,
                       help='Nazwa pliku wyjściowego SRT (domyślnie: video_id.srt lub nazwa_pliku.srt)')
    basic_group.add_argument('--download', type=str, metavar='URL',
                   help='Pobierz tylko wideo z YouTube (bez transkrypcji)')
    basic_group.add_argument('--download-audio-only', type=str, metavar='URL',
                   help='Pobierz tylko audio z YouTube (bez transkrypcji)')
    basic_group.add_argument('--video-quality', type=str, default='1080',
                       choices=['720', '1080', '1440', '2160'],
                       help='Jakość wideo przy pobieraniu z YouTube (domyślnie: 1080)')
    basic_group.add_argument('--audio-quality', type=str, default='best',
                       choices=['best', '192', '128', '96'],
                       help='Jakość audio przy pobieraniu (domyślnie: best)')

    # ===== OPCJE TRANSKRYPCJI =====
    transcription_group = parser.add_argument_group('Opcje transkrypcji', 'Konfiguracja modelu i języka transkrypcji')
    transcription_group.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Rozmiar modelu Whisper (domyślnie: base)')
    transcription_group.add_argument('--language', type=str, default=None,
                       help='Język transkrypcji (domyślnie: auto-detekcja)')
    transcription_group.add_argument('--engine', default='whisper',
                   choices=['whisper', 'faster-whisper', 'whisperx'],
                   help='Silnik transkrypcji (domyślnie: whisper)')
    transcription_group.add_argument('-t', '--translate', type=str, choices=['pl-en', 'en-pl'],
                       help='Tłumaczenie (pl-en: polski->angielski, en-pl: angielski->polski)')

    # ===== OPCJE WHISPERX =====
    whisperx_group = parser.add_argument_group('Opcje WhisperX', 'Zaawansowane funkcje dla silnika WhisperX')

    whisperx_group.add_argument('--whisperx-align', action='store_true',
                       help='Włącz word-level alignment (dokładniejsze timestampy)')

    whisperx_group.add_argument('--whisperx-diarize', action='store_true',
                       help='Włącz speaker diarization (rozpoznawanie mówców)')

    whisperx_group.add_argument('--whisperx-min-speakers', type=int, default=None,
                       help='Minimalna liczba mówców (dla diarization)')

    whisperx_group.add_argument('--whisperx-max-speakers', type=int, default=None,
                       help='Maksymalna liczba mówców (dla diarization)')

    whisperx_group.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace token (wymagany dla diarization)')

    # ===== OPCJE DUBBINGU I NAPISÓW =====
    dubbing_group = parser.add_argument_group('Opcje dubbingu i napisów', 'Konfiguracja TTS i wgrywania napisów do wideo')
    dubbing_group.add_argument('--dub', action='store_true',
                       help='Generuj dubbing TTS (wymaga lokalnego pliku lub pobiera z YouTube)')
    dubbing_group.add_argument('--dub-audio-only', action='store_true',
                       help='Generuj tylko ścieżkę audio z dubbingiem (bez wideo, format WAV)')

    # TTS Engine selection
    dubbing_group.add_argument('--tts-engine', type=str, default='edge',
                       choices=['edge', 'coqui'],
                       help='Silnik TTS (domyślnie: edge)')

    # Edge TTS options
    dubbing_group.add_argument('--tts-voice', type=str, default='pl-PL-MarekNeural',
                       choices=[
                       # Polski
                       'pl-PL-MarekNeural',    # Polski męski
                       'pl-PL-ZofiaNeural',    # Polski żeński
                       # Angielski (US)
                       'en-US-GuyNeural',      # Angielski (US) męski
                       'en-US-JennyNeural',    # Angielski (US) żeński
                       # Angielski (UK)
                       'en-GB-RyanNeural',     # Angielski (UK) męski
                       'en-GB-SoniaNeural',    # Angielski (UK) żeński
                       # Angielski (AU)
                       'en-AU-WilliamNeural',  # Angielski (AU) męski
                       'en-AU-NatashaNeural',  # Angielski (AU) żeński
                       # Niemiecki
                       'de-DE-ConradNeural',   # Niemiecki męski
                       'de-DE-KatjaNeural',    # Niemiecki żeński
                       # Francuski
                       'fr-FR-HenriNeural',    # Francuski męski
                       'fr-FR-DeniseNeural',   # Francuski żeński
                       # Hiszpański
                       'es-ES-AlvaroNeural',   # Hiszpański (ES) męski
                       'es-ES-ElviraNeural',   # Hiszpański (ES) żeński
                       # Włoski
                       'it-IT-DiegoNeural',    # Włoski męski
                       'it-IT-ElsaNeural',     # Włoski żeński
                       # Rosyjski
                       'ru-RU-DmitryNeural',   # Rosyjski męski
                       'ru-RU-SvetlanaNeural', # Rosyjski żeński
                       # Japoński
                       'ja-JP-KeitaNeural',    # Japoński męski
                       'ja-JP-NanamiNeural',   # Japoński żeński
                       # Chiński
                       'zh-CN-YunxiNeural',    # Chiński (uproszczony) męski
                       'zh-CN-XiaoxiaoNeural', # Chiński (uproszczony) żeński
                       # Koreański
                       'ko-KR-InJoonNeural',   # Koreański męski
                       'ko-KR-SunHiNeural',    # Koreański żeński
                       # Ukraiński
                       'uk-UA-OstapNeural',    # Ukraiński męski
                       'uk-UA-PolinaNeural',   # Ukraiński żeński
                       # Czeski
                       'cs-CZ-AntoninNeural',  # Czeski męski
                       'cs-CZ-VlastaNeural',   # Czeski żeński
                       ],
                       help='Głos Edge TTS (domyślnie: pl-PL-MarekNeural)')

    # Coqui TTS options
    dubbing_group.add_argument('--coqui-model', type=str, default='tts_models/pl/mai_female/vits',
                       help='Model Coqui TTS (domyślnie: tts_models/pl/mai_female/vits)')
    dubbing_group.add_argument('--coqui-speaker', type=str, default=None,
                       help='Speaker ID dla modeli multi-speaker (opcjonalnie)')

    dubbing_group.add_argument('--tts-volume', type=float, default=1.0,
                       help='Głośność TTS 0.0-2.0 (domyślnie: 1.0)')
    dubbing_group.add_argument('--original-volume', type=float, default=0.2,
                       help='Głośność oryginalnego audio 0.0-1.0 (domyślnie: 0.2)')
    dubbing_group.add_argument('--dub-output', type=str,
                       help='Nazwa pliku wyjściowego z dubbingiem (domyślnie: video_id_dubbed.mp4)')
    dubbing_group.add_argument('--burn-subtitles', action='store_true',
                   help='Wgraj napisy na stałe do wideo (wymaga wideo z YouTube lub lokalnego)')
    dubbing_group.add_argument('--subtitle-style', type=str,
                    default='FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=20',
                    help='Styl napisów ASS (domyślnie: biały tekst, półprzezroczyste ciemne tło)')
    dubbing_group.add_argument('--burn-output', type=str,
                   help='Nazwa pliku wyjściowego z napisami (domyślnie: {video_id}_subtitled.mp4)')

    # ===== OPCJE ZAAWANSOWANE =====
    advanced_group = parser.add_argument_group('Opcje zaawansowane', 'Zaawansowana konfiguracja i opcje developerskie')
    advanced_group.add_argument('--max-segment-duration', type=int, default=10,
                   help='Maksymalna długość segmentu w sekundach (domyślnie: 10)')
    advanced_group.add_argument('--max-segment-words', type=int, default=15,
                   help='Maksymalna liczba słów w segmencie (domyślnie: 15)')
    advanced_group.add_argument('--fill-gaps', action='store_true',
                   help='Wypełnij luki w timestampach dla lepszej synchronizacji dubbingu')
    advanced_group.add_argument('--min-pause', type=int, default=300,
                   help='Minimalna pauza między segmentami w ms (domyślnie: 300)')
    advanced_group.add_argument('--max-gap-fill', type=int, default=2000,
                   help='Maksymalna luka do wypełnienia w ms (domyślnie: 2000)')
    advanced_group.add_argument('--only-download', action='store_true',
                       help='Tylko pobierz audio, nie transkrybuj (developerski)')
    advanced_group.add_argument('--only-chunk', action='store_true',
                       help='Tylko podziel audio, nie transkrybuj (developerski)')
    advanced_group.add_argument('--only-transcribe', action='store_true',
                       help='Tylko transkrybuj chunki, nie generuj SRT (developerski)')
    advanced_group.add_argument('--test-merge', action='store_true',
                       help='Test generowania SRT z hardcoded danymi (developerski)')
    advanced_group.add_argument('--transcription-timeout', type=int, default=1800,
        help='Timeout per chunk in seconds (default: 1800 = 30 min, 0 = no timeout)')
    advanced_group.add_argument('--debug', action='store_true',
        help='Enable debug logging with detailed diagnostics')



    args = parser.parse_args()

    # Configure debug mode (PHASE 3)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        logger.debug(f"Arguments: {vars(args)}")

    # Handle special modes
    if args.download:
        return handle_video_download_mode(args)

    if args.download_audio_only:
        return handle_audio_download_mode(args)

    if args.test_merge:
        return handle_test_merge_mode(args)

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

    # Check TTS dependency if dubbing is requested
    if args.dub or args.dub_audio_only:
        if args.tts_engine == "edge":
            tts_ok, tts_msg = check_edge_tts_dependency()
            if not tts_ok:
                print(tts_msg)
                return 1
        elif args.tts_engine == "coqui":
            tts_ok, tts_msg = check_coqui_tts_dependency()
            if not tts_ok:
                print(tts_msg)
                return 1
        else:
            print(f"Błąd: Nieznany silnik TTS: {args.tts_engine}")
            return 1

    # Create temporary directory for intermediate files
    temp_dir = None
    original_video_path = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="transcribe_")
        print(f"Katalog tymczasowy: {temp_dir}")

        # Process input source: local file or YouTube URL
        success, error_msg, audio_path, original_video_path, input_stem = process_input_source(args, temp_dir)
        if not success:
            print(error_msg)
            return 1

        if args.only_download:
            # For --only-download, copy file to current directory before cleanup
            audio_file = Path(audio_path)
            dest_path = Path.cwd() / audio_file.name
            shutil.copy2(audio_path, dest_path)
            print(f"Plik audio skopiowany do: {dest_path}")
            return 0

        # Run transcription pipeline (handles audio splitting, transcription, translation)
        success, error_msg, all_segments = run_transcription_pipeline(audio_path, args, temp_dir)
        if not success:
            # Special exit codes for --only-chunk and --only-transcribe modes
            if error_msg in ["ONLY_CHUNK_MODE", "ONLY_TRANSCRIBE_MODE"]:
                return 0
            print(error_msg)
            return 1

        # Generate SRT file
        success, error_msg, srt_filename = generate_srt_output(all_segments, input_stem, args, temp_dir)
        if not success:
            print(error_msg)
            return 1

        # Burn subtitles to video if requested
        if args.burn_subtitles:
            success, error_msg = burn_subtitles_to_video_pipeline(
                original_video_path,
                srt_filename,
                input_stem,
                args,
                temp_dir
            )
            if not success:
                print(error_msg)
                return 1

        # TTS Dubbing if requested
        if args.dub or args.dub_audio_only:
            success, error_msg = run_dubbing_pipeline(
                all_segments,
                audio_path,
                original_video_path,
                input_stem,
                args,
                temp_dir
            )
            if not success:
                print(error_msg)
                return 1


        OutputManager.success("Transkrypcja zakończona pomyślnie!")
        # print(f"[OK] Plik SRT zapisany: {srt_filename}")
        # print(f"\nMożesz otworzyć plik SRT w VLC lub innym odtwarzaczu wideo.")

        return 0

    finally:
        # Always cleanup temporary files, even on error
        if temp_dir:
            cleanup_temp_files(temp_dir)


if __name__ == '__main__':
    sys.exit(main())
