#!/usr/bin/env python3
"""
Transcription Engines Module

This module provides different transcription engines for audio files:
- OpenAI Whisper (GPU/CPU support)
- WhisperX (GPU/CPU, with alignment and diarization)

All engines return transcription segments in the format:
List[Tuple[int, int, str]] - (start_ms, end_ms, text)

Optional speaker_info dict with gender detection:
{"SPEAKER_00": {"gender": "male", "pitch_mean": 120.5, "confidence": 0.85}}
"""

import threading
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from tqdm import tqdm

from .device_manager import detect_device, check_vram_for_model, clear_cuda_cache
from .output_manager import OutputManager
from .audio_processor import get_audio_duration


def _run_with_progress(transcribe_fn, wav_path: str, segment_progress_bar: tqdm,
                       device: str):
    """
    Wrapper który uruchamia transkrypcję z aktualizacją postępu w tle.

    Estymuje postęp na podstawie: elapsed_time / estimated_total_time
    - GPU (cuda): ~0.3-0.5x realtime
    - CPU: ~2-5x realtime (zależnie od modelu)
    """
    # Pobierz długość audio
    success, audio_duration = get_audio_duration(wav_path)
    if not success or audio_duration <= 0:
        audio_duration = 60.0  # fallback

    # Estymuj współczynnik czasu (ile sekund transkrypcji na sekundę audio)
    # Te wartości można dostroić na podstawie rzeczywistych pomiarów
    time_factor = 0.4 if device == "cuda" else 3.0
    estimated_total_time = audio_duration * time_factor

    result = [None]
    error = [None]
    stop_progress = threading.Event()

    def transcription_thread():
        try:
            result[0] = transcribe_fn()
        except Exception as e:
            error[0] = e
        finally:
            stop_progress.set()

    def progress_thread():
        start_time = time.time()
        while not stop_progress.is_set():
            elapsed = time.time() - start_time
            # Oblicz procent (cap at 99% until done)
            progress_pct = min((elapsed / estimated_total_time) * 100, 99.0)

            if segment_progress_bar:
                mins, secs = divmod(int(elapsed), 60)
                segment_progress_bar.set_postfix_str(
                    f"{progress_pct:.0f}% | {mins}:{secs:02d} elapsed"
                )

            stop_progress.wait(0.5)  # Update every 0.5s

    # Uruchom wątki
    t_transcribe = threading.Thread(target=transcription_thread)
    t_progress = threading.Thread(target=progress_thread, daemon=True)

    t_transcribe.start()
    t_progress.start()
    t_transcribe.join()

    stop_progress.set()

    # Ustaw 100% na końcu
    if segment_progress_bar:
        segment_progress_bar.set_postfix_str("100% | done")

    if error[0]:
        raise error[0]

    return result[0]


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

    # Sprawdzenie dostępnej VRAM
    if device == "cuda":
        vram_ok, vram_warning = check_vram_for_model(model_size)
        if not vram_ok:
            tqdm.write(f"OSTRZEŻENIE: {vram_warning}")

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

    # Użyj wrappera z progress
    def do_transcribe():
        return model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda")
        )

    try:
        result = _run_with_progress(
            do_transcribe,
            str(wav_path),
            segment_progress_bar,
            device
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

    tqdm.write(f"Wykryty język: {result['language']}")

    # Zwolnij pamięć GPU po transkrypcji
    clear_cuda_cache()

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
    force_device: str = 'auto',
    detect_gender: bool = False
) -> Union[Tuple[bool, str, List[Tuple[int, int, str]]], Tuple[bool, str, List[Tuple[int, int, str]], Dict[str, dict]]]:
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

    # Sprawdzenie dostępnej VRAM
    if device == "cuda":
        vram_ok, vram_warning = check_vram_for_model(model_size)
        if not vram_ok:
            tqdm.write(f"OSTRZEŻENIE: {vram_warning}")

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

    # Użyj wrappera z progress (obejmuje ładowanie audio i transkrypcję)
    def do_transcribe():
        audio = whisperx.load_audio(str(wav_path))
        return model.transcribe(audio, batch_size=16), audio

    try:
        result_and_audio = _run_with_progress(
            do_transcribe,
            str(wav_path),
            segment_progress_bar,
            device
        )
        result, audio = result_and_audio
    except Exception as e:
        return False, f"Błąd podczas transkrypcji WhisperX: {str(e)}", []

    # Wykryj język jeśli nie był podany jawnie
    detected_language = result.get("language", language)
    if detected_language:
        tqdm.write(f"Wykryty język: {detected_language}")

    # Word-level alignment (opcjonalnie)
    # Without alignment, timestamps come from VAD chunks and drift by seconds,
    # so a failed alignment must be reported rather than silently swallowed.
    align_warning = None
    if align:
        tqdm.write("Wykonywanie word-level alignment...")

        # Użyj wykrytego języka jeśli nie był podany jawnie
        align_language = language if language else detected_language

        if not align_language:
            align_warning = "Alignment POMINIĘTY - brak informacji o języku (czasy będą przybliżone)"
            tqdm.write(f"Ostrzeżenie: {align_warning}")
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
                align_warning = f"Alignment NIE POWIÓDŁ SIĘ: {e} (czasy będą przybliżone)"
                tqdm.write(f"Ostrzeżenie: {align_warning}")

    # Speaker diarization (opcjonalnie)
    # Failures here are non-fatal but must stay visible - a silently skipped
    # diarization looks identical to one that ran and found a single speaker.
    diarization_warning = None
    if diarize:
        if not hf_token:
            diarization_warning = (
                "Diaryzacja POMINIĘTA: brak tokenu HuggingFace "
                "(ustaw HF_TOKEN w .env lub podaj --hf-token)"
            )
            tqdm.write(f"Ostrzeżenie: {diarization_warning}")
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
                diarization_warning = f"Diaryzacja NIE POWIODŁA SIĘ: {e}"
                tqdm.write(f"Ostrzeżenie: {diarization_warning}")

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

    # Zwolnij pamięć GPU po transkrypcji
    clear_cuda_cache()

    # Wykrywanie płci mówców (opcjonalnie)
    speaker_info = {}
    if detect_gender and diarize:
        try:
            from .speaker_analyzer import analyze_speaker_gender
            tqdm.write("Analiza płci mówców...")
            speaker_info = analyze_speaker_gender(str(wav_path), segments)
        except Exception as e:
            tqdm.write(f"Ostrzeżenie: Analiza płci nie powiodła się: {e}")

    message = f"Transkrypcja zakończona: {len(segments)} segmentów"
    for warning in (align_warning, diarization_warning):
        if warning:
            message += f"\nOSTRZEŻENIE: {warning}"

    if detect_gender:
        return True, message, segments, speaker_info
    return True, message, segments


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
    force_device: str = 'auto',
    detect_gender: bool = False
) -> Union[Tuple[bool, str, List[Tuple[int, int, str]]], Tuple[bool, str, List[Tuple[int, int, str]], Dict[str, dict]]]:
    """
    Transkrypcja pliku WAV przy użyciu wybranego silnika.

    Args:
        wav_path: Ścieżka do pliku WAV
        model_size: Rozmiar modelu (tiny, base, small, medium, large)
        language: Kod języka (pl, en, etc.)
        engine: Silnik transkrypcji ('whisper', 'whisperx')
        segment_progress_bar: Progress bar dla segmentów
        timeout_seconds: Timeout dla transkrypcji (0 = bez limitu)
        whisperx_align: Włącz word-level alignment (tylko WhisperX)
        whisperx_diarize: Włącz speaker diarization (tylko WhisperX)
        whisperx_min_speakers: Min liczba mówców (tylko WhisperX)
        whisperx_max_speakers: Max liczba mówców (tylko WhisperX)
        hf_token: HuggingFace token (tylko WhisperX diarization)
        force_device: Device override ('auto', 'cuda', 'cpu')
        detect_gender: Włącz wykrywanie płci mówców (tylko WhisperX z diarization)

    Returns:
        Tuple (success: bool, message: str, segments: List[(start_ms, end_ms, text)])
        With detect_gender=True: adds speaker_info dict as 4th element
    """
    try:
        # Sprawdź czy plik istnieje
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"Błąd: Plik audio nie istnieje: {wav_path}", []

        # Wybór silnika transkrypcji
        if engine == "whisper":
            result = transcribe_with_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                force_device=force_device
            )
            # Whisper doesn't support diarization, so no gender detection
            if detect_gender:
                return result[0], result[1], result[2], {}
            return result

        elif engine == "whisperx":
            return transcribe_with_whisperx(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                align=whisperx_align,
                diarize=whisperx_diarize,
                min_speakers=whisperx_min_speakers,
                max_speakers=whisperx_max_speakers,
                hf_token=hf_token,
                force_device=force_device,
                detect_gender=detect_gender
            )

        else:
            if detect_gender:
                return False, f"Błąd: Nieobsługiwany silnik transkrypcji: {engine}", [], {}
            return False, f"Błąd: Nieobsługiwany silnik transkrypcji: {engine}", []

    except ImportError as e:
        if detect_gender:
            return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}", [], {}
        return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}", []
    except Exception as e:
        if detect_gender:
            return False, f"Błąd podczas transkrypcji: {str(e)}", [], {}
        return False, f"Błąd podczas transkrypcji: {str(e)}", []
