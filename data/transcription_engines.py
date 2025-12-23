#!/usr/bin/env python3
"""
Transcription Engines Module

This module provides different transcription engines for audio files:
- OpenAI Whisper (GPU/CPU support)
- Faster-Whisper (CPU only, optimized with CTranslate2)
- WhisperX (GPU/CPU, with alignment and diarization)

All engines return transcription segments in the format:
List[Tuple[int, int, str]] - (start_ms, end_ms, text)
"""

import threading
import time
from pathlib import Path
from typing import Tuple, List
from queue import Queue
from tqdm import tqdm

from .device_manager import detect_device
from .output_manager import OutputManager


def _run_transcription_with_timeout(
    model, audio_path: str, language: str,
    timeout_seconds: int, segment_progress_bar, vad_parameters: dict
) -> Tuple[bool, str, List[Tuple[int, int, str]], object]:
    """Run model.transcribe() with timeout protection using threading."""

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


def transcribe_with_whisper(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int,
    force_device: str = 'auto'
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
    device, device_info = detect_device(force_device=force_device)
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
    timeout_seconds: int,
    force_device: str = 'auto'
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z Faster-Whisper (zawsze CPU).

    Faster-Whisper wymusza użycie CPU ze względu na problemy z CTranslate2.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return False, "Błąd: faster-whisper nie jest zainstalowany. Zainstaluj: pip install faster-whisper", []

    # Log if user tried to force GPU
    if force_device == 'cuda':
        tqdm.write("Warning: Faster-Whisper zawsze używa CPU (CTranslate2 limitation). Ignoruję --device cuda")

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
    hf_token: str = None,
    force_device: str = 'auto'
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
    device, device_info = detect_device(force_device=force_device)
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

    # Wykryj język jeśli nie był podany jawnie
    detected_language = result.get("language", language)
    if detected_language:
        tqdm.write(f"Wykryty język: {detected_language}")

    # Word-level alignment (opcjonalnie)
    if align:
        tqdm.write("Wykonywanie word-level alignment...")

        # Użyj wykrytego języka jeśli nie był podany jawnie
        align_language = language if language else detected_language

        if not align_language:
            tqdm.write("Ostrzeżenie: Nie można wykonać alignment - brak informacji o języku")
        else:
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=align_language,
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
    hf_token: str = None,
    force_device: str = 'auto'
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
        force_device: Device override ('auto', 'cuda', 'cpu')

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
                segment_progress_bar, timeout_seconds,
                force_device=force_device
            )

        elif engine == "faster-whisper":
            return transcribe_with_faster_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                force_device=force_device
            )

        elif engine == "whisperx":
            return transcribe_with_whisperx(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                align=whisperx_align,
                diarize=whisperx_diarize,
                min_speakers=whisperx_min_speakers,
                max_speakers=whisperx_max_speakers,
                hf_token=hf_token,
                force_device=force_device
            )

        else:
            return False, f"Błąd: Nieobsługiwany silnik transkrypcji: {engine}", []

    except ImportError as e:
        return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}", []
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []
