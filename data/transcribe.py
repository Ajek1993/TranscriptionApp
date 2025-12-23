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

# ===== WARNING SUPPRESSION (MUST BE FIRST) =====
# Musi być przed jakimikolwiek importami modułów, aby zmienne środowiskowe
# zostały ustawione przed załadowaniem bibliotek (torch, tensorflow, itp.)
import sys
import os

# Wczesne wykrycie flagi --debug
debug_mode = '--debug' in sys.argv

# Ustawienie zmiennych środowiskowych przed importami
if not debug_mode:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONWARNINGS'] = 'ignore'

# Teraz można importować warning_suppressor
from warning_suppressor import suppress_third_party_warnings
suppress_third_party_warnings(debug_mode=debug_mode)

# ===== STANDARDOWE IMPORTY =====
import re
import subprocess
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


# Import from refactored modules
from output_manager import OutputManager
from command_builders import (
    build_ffprobe_audio_info_cmd,
    build_ffprobe_video_info_cmd,
    build_ffprobe_duration_cmd,
    build_ffmpeg_audio_extraction_cmd,
    build_ffmpeg_audio_split_cmd,
    build_ffmpeg_video_merge_cmd,
    build_ffmpeg_subtitle_burn_cmd,
    build_ytdlp_audio_download_cmd,
    build_ytdlp_video_download_cmd
)
from validators import (
    validate_youtube_url,
    validate_video_file,
    check_dependencies,
    check_edge_tts_dependency,
    check_coqui_tts_dependency,
    SUPPORTED_VIDEO_EXTENSIONS,
    TRANSLATOR_AVAILABLE,
    EDGE_TTS_AVAILABLE,
    COQUI_TTS_AVAILABLE
)
from utils import cleanup_temp_files
from segment_processor import split_long_segments, fill_timestamp_gaps, format_srt_timestamp
from srt_writer import write_srt
from device_manager import detect_device, get_gpu_memory_info, WHISPER_MODEL_MEMORY_REQUIREMENTS
from audio_processor import get_audio_duration, get_audio_duration_ms, split_audio
from youtube_processor import download_audio, download_video, extract_audio_from_video
from translation import translate_segments
from transcription_engines import transcribe_chunk
from tts_generator import (
    generate_tts_segments,
    create_tts_audio_track,
    determine_tts_target_language as tts_determine_language,
    XTTS_SUPPORTED_LANGUAGES as TTS_XTTS_LANGUAGES
)
from audio_mixer import mix_audio_tracks, create_dubbed_video, burn_subtitles_to_video


# ===== LOGGING SETUP =====

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# XTTS v2 supported languages


# ===== COMMAND BUILDERS =====

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
    OutputManager.stage_header(5, "Generowanie dubbingu TTS")

    # For full dubbing, we need video file
    if args.dub and not original_video_path:
        return False, "Błąd: Dubbing wymaga pliku wideo"

    # Create TTS directory in temp
    tts_dir = Path(temp_dir) / "tts"
    tts_dir.mkdir(exist_ok=True)

    # Determine target language for TTS
    tts_language = tts_determine_language(
        transcription_language=args.language,
        translation_spec=args.translate
    )

    print(f"Język TTS: {tts_language}", end="")
    if args.translate:
        src, tgt = args.translate.split('-')
        print(f" (tłumaczenie: {src} → {tgt})")
    else:
        print()

    # Generate TTS for each segment
    success, message, tts_files = generate_tts_segments(
        segments,
        str(tts_dir),
        voice=args.tts_voice,
        engine=args.tts_engine,
        coqui_model=args.coqui_model,
        coqui_speaker=args.coqui_speaker,
        speaker_wav=audio_path,
        target_language=tts_language
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


def generate_dual_language_ass_output(
    original_segments: List[Tuple[int, int, str]],
    translated_segments: List[Tuple[int, int, str]],
    input_stem: str,
    args,
    temp_dir: str
) -> Tuple[bool, str, str]:
    """
    Generate dual-language ASS file from original and translated segments.

    Returns:
        Tuple of (success, error_msg, ass_filename)
    """
    from ass_writer import write_dual_language_ass

    OutputManager.stage_header(3, "Generowanie pliku ASS dwujęzycznego")
    print(f"Segmenty oryginalne: {len(original_segments)}")
    print(f"Segmenty przetłumaczone: {len(translated_segments)}")

    # ASS always in temp_dir (for burning)
    ass_filename = str(Path(temp_dir) / f"{input_stem}_dual.ass")

    # Write ASS file
    success, message = write_dual_language_ass(
        original_segments,
        translated_segments,
        ass_filename
    )

    if not success:
        return False, message, ""

    print(message)
    return True, "", ass_filename


# ===== SUBTITLE BURNING PIPELINE =====

def burn_subtitles_to_video_pipeline(original_video_path: str, srt_filename: str, input_stem: str, args, temp_dir: str) -> Tuple[bool, str]:
    """
    Handle complete subtitle burning pipeline.

    Returns:
        Tuple of (success, error_msg)
    """
    OutputManager.stage_header(4, "Wgrywanie napisów do wideo")

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

def _transcribe_all_chunks(chunk_paths: List[str], args, force_device: str = 'auto') -> Tuple[bool, str, List[Tuple[int, int, str]]]:
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
                hf_token=getattr(args, 'hf_token', None),
                force_device=force_device
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


def run_transcription_pipeline(audio_path: str, args, temp_dir: str) -> Tuple[bool, str, List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """
    Run complete transcription pipeline.

    Handles:
    - Audio splitting
    - Chunk transcription
    - Segment splitting (for dubbing)
    - Gap filling
    - Translation (if requested)

    Returns:
        Tuple of (success, error_msg, original_segments, translated_segments)
        - For non-dual-language mode: original_segments will be empty []
        - For dual-language mode: both lists will be populated
    """
    # Stage 2: Split audio into chunks
    success, message, chunk_paths = split_audio(audio_path, output_dir=temp_dir)
    if not success:
        return False, message, [], []

    print(message)

    if args.only_chunk:
        # For --only-chunk, copy chunks to current directory before cleanup
        for chunk_path in chunk_paths:
            chunk_file = Path(chunk_path)
            dest_path = Path.cwd() / chunk_file.name
            shutil.copy2(chunk_path, dest_path)
            print(f"Chunk skopiowany do: {dest_path}")
        return False, "ONLY_CHUNK_MODE", [], []

    # Stage 3: Transcribe all chunks
    success, message, all_segments = _transcribe_all_chunks(chunk_paths, args, force_device=args.device)
    if not success:
        return False, message, [], []

    # Preserve original segments for dual-language mode
    original_segments = all_segments.copy() if args.dual_language else []

    # Translation step (if requested)
    success, message, translated_segments = _translate_segments_if_requested(all_segments, args)
    if not success:
        return False, message, [], []

    if args.only_transcribe:
        print(f"\nŁącznie transkrybowanych segmentów: {len(all_segments)}")
        return False, "ONLY_TRANSCRIBE_MODE", [], []

    # Return both sets for dual-language, or translated/original for normal mode
    if args.dual_language:
        return True, "", original_segments, translated_segments
    else:
        # Backward compatibility: return translated (or original if no translation) in second position
        return True, "", translated_segments, []


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
                    default='FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=20',
                    help='Styl napisów ASS (domyślnie: biały tekst, półprzezroczyste ciemne tło)')
    dubbing_group.add_argument('--burn-output', type=str,
                   help='Nazwa pliku wyjściowego z napisami (domyślnie: {video_id}_subtitled.mp4)')
    dubbing_group.add_argument('--dual-language', action='store_true',
                   help='Wypal napisy dwujęzyczne (żółty oryginał + białe tłumaczenie). '
                        'Wymaga: --translate i --burn-subtitles')

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
    advanced_group.add_argument('--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device selection for transcription engines: auto (default, GPU if available), cuda (force GPU), cpu (force CPU). Note: Coqui TTS always tries GPU regardless of this setting.')



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

    # Validate --dual-language requirements
    if args.dual_language:
        if not args.translate:
            print("Błąd: --dual-language wymaga --translate")
            return 1
        if not args.burn_subtitles:
            print("Błąd: --dual-language wymaga --burn-subtitles")
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
        success, error_msg, original_segments, translated_segments = run_transcription_pipeline(audio_path, args, temp_dir)
        if not success:
            # Special exit codes for --only-chunk and --only-transcribe modes
            if error_msg in ["ONLY_CHUNK_MODE", "ONLY_TRANSCRIBE_MODE"]:
                return 0
            print(error_msg)
            return 1

        # Determine which segments to use for single-language operations (backward compatibility)
        final_segments = translated_segments if translated_segments else original_segments
        subtitle_filename = ""

        # Generate subtitles: ASS for dual-language, SRT otherwise
        if args.dual_language:
            success, error_msg, subtitle_filename = generate_dual_language_ass_output(
                original_segments,
                translated_segments,
                input_stem,
                args,
                temp_dir
            )
        else:
            success, error_msg, subtitle_filename = generate_srt_output(
                final_segments,
                input_stem,
                args,
                temp_dir
            )

        if not success:
            print(error_msg)
            return 1

        # Burn subtitles to video if requested
        if args.burn_subtitles:
            success, error_msg = burn_subtitles_to_video_pipeline(
                original_video_path,
                subtitle_filename,  # Can be SRT or ASS - FFmpeg handles both
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
                final_segments,
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
