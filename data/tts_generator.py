#!/usr/bin/env python3
"""
TTS Generator Module

This module provides Text-to-Speech generation functionality using:
- Edge TTS (Microsoft Azure voices, fast and free)
- Coqui TTS (Local TTS, supports XTTS v2 with voice cloning)

All TTS functions handle automatic speed adjustment to fit audio segments
and return segment information in the format:
List[Tuple[int, int, str, float]] - (start_ms, end_ms, file_path, duration_sec)
"""

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Tuple, List, NamedTuple
from tqdm import tqdm

from .device_manager import detect_device
from .audio_processor import get_audio_duration

# Na początku pliku, po importach:
TTS_SLOT_FILL_RATIO = 0.99# Wypełnienie 95% czasu segmentu (daje 50ms marginesu dla 1000ms slotu)

# Gap Extension Configuration
MIN_PAUSE_BEFORE_NEXT_MS = 50   # Minimalna cisza przed następnym segmentem (150ms = naturalna pauza w mowie)
MAX_GAP_EXTENSION_RATIO = 0.99    # Użyj max 70% dostępnego gap'u (zostaw 30% jako bufor)

# Conditional imports for TTS engines
try:
    import edge_tts
except ImportError:
    pass

try:
    from TTS.api import TTS
except ImportError:
    pass


# XTTS v2 supported languages
XTTS_SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl",
    "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"
]


class SegmentTiming(NamedTuple):
    """
    Rozszerzona struktura timing info dla segmentów TTS.

    Attributes:
        original_start_ms: Oryginalny timestamp początku (dla napisów SRT/ASS)
        original_end_ms: Oryginalny timestamp końca (dla napisów SRT/ASS)
        adjusted_start_ms: Przesunięty timestamp początku (dla audio TTS)
        adjusted_end_ms: Przesunięty timestamp końca (dla audio TTS)
        tts_file: Ścieżka do pliku TTS
        tts_duration_sec: Rzeczywista długość TTS
        shift_ms: Wartość przesunięcia względem oryginału (dla raportowania)
    """
    original_start_ms: int
    original_end_ms: int
    adjusted_start_ms: int
    adjusted_end_ms: int
    tts_file: str
    tts_duration_sec: float
    shift_ms: int


def determine_tts_target_language(
    transcription_language: str = None,
    translation_spec: str = None
) -> str:
    """
    Determine which language TTS should use.

    Logic:
    1. If --translate specified: use TARGET language (e.g., "pl-en" -> "en")
    2. Else if --language specified: use transcription language
    3. Else: default to "pl"

    Args:
        transcription_language: Value from --language flag (e.g., "pl", "en")
        translation_spec: Value from --translate flag (e.g., "pl-en", "en-pl")

    Returns:
        ISO language code for TTS ("pl", "en", etc.)

    Examples:
        >>> determine_tts_target_language(None, "pl-en")
        "en"
        >>> determine_tts_target_language("pl", None)
        "pl"
        >>> determine_tts_target_language(None, None)
        "pl"
    """
    if translation_spec:
        # Translation takes priority - use TARGET language
        src_lang, tgt_lang = translation_spec.split('-')
        return tgt_lang

    if transcription_language:
        return transcription_language

    return "pl"  # Default fallback


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
    tts_instance: 'TTS' = None,
    language: str = "pl"
) -> Tuple[bool, str, float]:
    """
    Generate TTS audio for a single text segment with Coqui TTS.

    Args:
        text: Text to convert to speech
        output_path: Path to save the MP3 file
        model_name: Coqui TTS model name (default: tts_models/pl/mai_female/vits)
        speaker: Speaker ID for multi-speaker models (optional)
        speaker_wav: Path to reference audio for voice cloning (XTTS v2)
        speed: Speech speed multiplier (default: 1.0, range: 0.5-2.0)
        language: Language code for XTTS models (e.g., "pl", "en")
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
            # Coqui TTS zawsze próbuje GPU (niezależnie od --device dla transkrypcji)
            tts_device, tts_device_info = detect_device(force_device='auto')
            try:
                tts = tts.to(tts_device)
            except Exception as e:
                # Silently ignore if .to() fails - TTS will use default device
                pass
        else:
            tts = tts_instance

        # Generate TTS audio
        kwargs = {"text": text, "file_path": temp_wav}
        if "xtts" in model_name.lower():
            kwargs["language"] = language
            if speaker_wav:  # Priority 1: WAV file for voice cloning
                kwargs["speaker_wav"] = speaker_wav
            elif speaker:  # Priority 2: speaker name
                kwargs["speaker"] = speaker
            else:
                return False, "XTTS v2 wymaga speaker_wav LUB speaker", 0.0
        elif speaker:
            kwargs["speaker"] = speaker

        tts.tts_to_file(**kwargs)

        # Validate speed parameter
        if not isinstance(speed, (int, float)) or speed <= 0 or speed > 2.0:
            print(f"WARNING: Nieprawidłowy speed={speed}, używam 1.0")
            speed = 1.0

        if speed != speed:  # Check for NaN
            print("WARNING: Speed is NaN, używam 1.0")
            speed = 1.0

        # Apply speed adjustment and convert to MP3 if needed
        # Use epsilon tolerance for float comparison to avoid precision issues
        if abs(speed - 1.0) > 0.001:  # 0.1% tolerance
            # Measure duration BEFORE speed adjustment (for verification)
            success_before, duration_before = get_audio_duration(temp_wav)
            if not success_before:
                return False, "Błąd: Nie można odczytać długości temp WAV", 0.0

            # Use ffmpeg to adjust speed and convert to MP3
            cmd = [
                'ffmpeg', '-y', '-i', temp_wav,
                '-filter:a', f'atempo={speed}',
                '-b:a', '192k',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Clean up temporary WAV
            if Path(temp_wav).exists():
                Path(temp_wav).unlink()

            if result.returncode != 0:
                return False, f"Błąd konwersji audio: {result.stderr}", 0.0

            # Get duration AFTER speed adjustment
            success, duration_sec = get_audio_duration(output_path)
            if not success:
                return False, "Błąd: Nie można odczytać długości wygenerowanego TTS", 0.0

            # Verify that atempo filter actually worked
            expected_duration = duration_before / speed
            duration_ratio = duration_sec / expected_duration

            # If ratio is far from 1.0, atempo may have failed
            if abs(duration_ratio - 1.0) > 0.1:  # 10% tolerance
                print(f"WARNING: FFmpeg atempo może nie działać poprawnie!")
                print(f"  Duration przed: {duration_before:.2f}s")
                print(f"  Speed: {speed:.2f}x")
                print(f"  Duration oczekiwany: {expected_duration:.2f}s")
                print(f"  Duration faktyczny: {duration_sec:.2f}s")
                print(f"  Ratio: {duration_ratio:.2f} (powinno być ~1.0)")
                # NOTE: Nie zwracamy błędu - pozwalamy na kontynuację z warning
        else:
            # Just convert WAV to MP3 (no speed adjustment)
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
    speaker_wav: str = None,
    target_language: str = "pl"
) -> Tuple[bool, str, List[Tuple[int, int, str, float]]]:
    """
    Generate TTS for all segments with automatic speed adjustment.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        output_dir: Directory to save TTS files
        voice: Voice name (for Edge TTS)
        engine: TTS engine ('edge' or 'coqui')
        coqui_model: Coqui TTS model name (for Coqui TTS)
        coqui_speaker: Speaker ID for multi-speaker Coqui models (optional)
        speaker_wav: Path to reference audio for voice cloning (XTTS v2)
        target_language: Target language code for TTS (e.g., "pl", "en")

    Returns:
        Tuple of (success: bool, message: str, tts_files: List[(start_ms, end_ms, path, duration_sec)])
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
            # Coqui TTS zawsze próbuje GPU (niezależnie od --device dla transkrypcji)
            tts_device, tts_device_info = detect_device(force_device='auto')
            try:
                coqui_tts_instance = coqui_tts_instance.to(tts_device)
                print(f"Model Coqui TTS załadowany pomyślnie na: {tts_device_info}")
            except Exception as e:
                print(f"Warning: Nie można przenieść Coqui TTS na {tts_device}: {e}")
                print("Coqui TTS użyje domyślnego urządzenia")
        except Exception as e:
            return False, f"Błąd ładowania modelu Coqui TTS: {str(e)}", []

    print(f"Generowanie TTS dla {len(segments)} segmentów (engine: {engine})...")

    # Preprocessing: oblicz dostępne gap'y dla każdego segmentu (Gap Extension)
    segment_gaps = []
    for idx in range(len(segments)):
        start_ms, end_ms, text = segments[idx]
        slot_duration_ms = end_ms - start_ms

        # Oblicz dostępny gap do następnego segmentu
        if idx < len(segments) - 1:
            next_start_ms = segments[idx + 1][0]
            available_gap_ms = max(0, next_start_ms - end_ms)
        else:
            available_gap_ms = 0  # Ostatni segment - brak gap'u

        # Oblicz ile gap'u możemy wykorzystać (zachowaj MIN_PAUSE_BEFORE_NEXT_MS)
        usable_gap_ms = max(0, available_gap_ms - MIN_PAUSE_BEFORE_NEXT_MS)

        # Oblicz rozszerzony slot (wykorzystaj część gap'u)
        extended_slot_ms = slot_duration_ms + int(usable_gap_ms * MAX_GAP_EXTENSION_RATIO)

        segment_gaps.append({
            'slot_duration_ms': slot_duration_ms,
            'available_gap_ms': available_gap_ms,
            'usable_gap_ms': usable_gap_ms,
            'extended_slot_ms': extended_slot_ms
        })

    with tqdm(total=len(segments), desc="Generating TTS", unit="seg") as pbar:
        for idx, (start_ms, end_ms, text) in enumerate(segments):
            # Skip empty text
            # Bezpieczne czyszczenie tekstu
            if text is None:
                cleaned_text = ""
            else:
                cleaned_text = str(text).strip()

            # Skip empty text (po strip)
            if not cleaned_text:
                pbar.update(1)
                continue

            text = cleaned_text

            # Pobierz obliczone gap'y dla tego segmentu
            gap_info = segment_gaps[idx]
            slot_duration_ms = gap_info['slot_duration_ms']
            extended_slot_ms = gap_info['extended_slot_ms']
            slot_duration_sec = slot_duration_ms / 1000.0
            extended_slot_sec = extended_slot_ms / 1000.0

            tts_file = output_path / f"tts_{idx:04d}.mp3"

            # Generate with 1.25x speed by default (proactive overflow prevention)
            speed = 1.25  # For Coqui TTS - domyślne przyspieszenie
            rate = "+25%"  # For Edge TTS - domyślne przyspieszenie
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
                            text, str(tts_file), coqui_model, coqui_speaker, speaker_wav, speed,
                            coqui_tts_instance, language=target_language
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

                    # Check if TTS is too long for the EXTENDED slot (Gap Extension)
                    # Używamy extended_slot_sec który zawiera wykorzystany gap
                    target_duration_sec = extended_slot_sec * TTS_SLOT_FILL_RATIO

                    if tts_duration > target_duration_sec:
                        # TTS za długi nawet z gap extension - przyspiesz
                        speed_multiplier = tts_duration / target_duration_sec
                        speed_multiplier = max(1.0, min(speed_multiplier, 2.0))  # Only speed UP (1.0x-2.0x)

                        # Informacja o gap extension jeśli wykorzystany
                        gap_info_str = ""
                        if extended_slot_ms > slot_duration_ms:
                            gap_info_str = f", +{extended_slot_ms - slot_duration_ms}ms gap"

                        if engine == "edge":
                            rate_percent = int((speed_multiplier - 1.0) * 100)
                            rate_percent = min(rate_percent, 100)  # Cap at +100%
                            rate = f"+{rate_percent}%"
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {extended_slot_sec:.2f}s{gap_info_str}), przyspieszam do {rate}")
                        elif engine == "coqui":
                            speed = speed_multiplier
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {extended_slot_sec:.2f}s{gap_info_str}), przyspieszam do {speed:.2f}x")

                        # Regenerate with adjusted speed
                        if engine == "edge":
                            success, message, tts_duration = asyncio.run(
                                generate_tts_for_segment(text, str(tts_file), voice, rate)
                            )
                        elif engine == "coqui":
                            tqdm.write(f"  DEBUG: Regeneruję TTS z speed={speed:.3f}")
                            success, message, tts_duration = generate_tts_coqui_for_segment(
                                text, str(tts_file), coqui_model, coqui_speaker, speaker_wav, speed,
                                coqui_tts_instance, language=target_language
                            )
                            if success:
                                tqdm.write(f"  DEBUG: Po regeneracji duration={tts_duration:.3f}s (target={target_duration_sec:.3f}s)")

                        if not success:
                            if retry < max_retries - 1:
                                time.sleep(0.5 * (retry + 1))
                                continue
                            else:
                                tqdm.write(f"Ostrzeżenie: Nie udało się wygenerować TTS dla segmentu {idx}: {message}")
                                pbar.update(1)
                                break

                        # Sprawdź czy TTS nadal nie mieści się po przyspieszeniu
                        if tts_duration > target_duration_sec:
                            overflow_ms = int((tts_duration - target_duration_sec) * 1000)
                            text_preview = text[:50] + "..." if len(text) > 50 else text
                            tqdm.write(
                                f"WARNING: Segment {idx}: TTS przekracza slot nawet przy max przyspieszeniu!\n"
                                f"  Tekst: \"{text_preview}\"\n"
                                f"  Slot: {slot_duration_ms}ms | Rozszerzony: {extended_slot_ms}ms | TTS: {int(tts_duration * 1000)}ms (+{overflow_ms}ms overflow)"
                            )

                    tts_files.append((start_ms, end_ms, str(tts_file), tts_duration))
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


def adjust_timestamps_for_overflow(
    tts_files: List[Tuple[int, int, str, float]],
    min_pause_ms: int = 0
) -> List[SegmentTiming]:
    """
    Dynamicznie przesuwa timestampy dla segmentów TTS z overflow.

    Nowa uproszczona logika:
    - Każdy segment zaczyna się w oryginalnym timestampie LUB zaraz po poprzednim (co później)
    - Automatyczny powrót do oryginalnych timestampów gdy jest miejsce
    - Przy overflow: następny segment od razu, bez przerwy (0ms)

    Args:
        tts_files: Lista (original_start_ms, original_end_ms, tts_file, tts_duration_sec)
        min_pause_ms: Nieużywane (zachowane dla kompatybilności API)

    Returns:
        Lista SegmentTiming z oryginalnymi + adjusted timestampami
    """
    if not tts_files:
        return []

    result = []
    previous_end_ms = 0  # Gdzie kończy się poprzedni TTS

    for idx, (orig_start_ms, orig_end_ms, tts_file, tts_duration_sec) in enumerate(tts_files):
        tts_duration_ms = int(tts_duration_sec * 1000)

        # Optymalny start: oryginalny timestamp LUB zaraz po poprzednim (co później)
        adjusted_start_ms = max(orig_start_ms, previous_end_ms)
        adjusted_end_ms = adjusted_start_ms + tts_duration_ms

        # Oblicz shift dla raportowania
        shift_ms = adjusted_start_ms - orig_start_ms

        # Debug info
        if shift_ms > 0:
            print(f"  Segment {idx}: przesunięty o +{shift_ms}ms (poprzedni TTS skończył się w {previous_end_ms}ms)")
        elif previous_end_ms > 0 and adjusted_start_ms == orig_start_ms:
            # Powrót do oryginalnego timestampu
            print(f"  Segment {idx}: powrót do oryginalnego timestampu ({orig_start_ms}ms)")

        timing = SegmentTiming(
            original_start_ms=orig_start_ms,
            original_end_ms=orig_end_ms,
            adjusted_start_ms=adjusted_start_ms,
            adjusted_end_ms=adjusted_end_ms,
            tts_file=tts_file,
            tts_duration_sec=tts_duration_sec,
            shift_ms=shift_ms
        )
        result.append(timing)

        # Aktualizuj gdzie kończy się ten TTS (dla następnego segmentu)
        previous_end_ms = adjusted_end_ms

    # WARNING dla dużych przesunięć na końcu
    final_shift = result[-1].shift_ms if result else 0
    if final_shift > 10000:
        print(f"WARNING: Duże przesunięcie ostatniego segmentu (+{final_shift}ms)!")
        print("  TTS może znacząco wyprzedzić oryginalne napisy.")

    return result


def create_tts_audio_track(
    segment_timings: List[SegmentTiming],
    total_duration_ms: int,
    output_path: str
) -> Tuple[bool, str]:
    """
    Combine TTS segments into a single audio track using concat demuxer file.
    This avoids command line length limits on Windows.
    Uses Dynamic Timestamp Shifting for proper synchronization.

    Args:
        segment_timings: List of SegmentTiming with adjusted timestamps
        total_duration_ms: Total duration of the final track (including shifts)
        output_path: Path to save the combined audio

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not segment_timings:
            return False, "Błąd: Brak plików TTS do połączenia"

        temp_dir = Path(output_path).parent

        # Sort segments by adjusted start time
        sorted_segments = sorted(segment_timings, key=lambda x: x.adjusted_start_ms)

        print(f"Łączenie {len(sorted_segments)} segmentów TTS...")

        # Create list of all audio segments with silence gaps
        all_segments = []
        current_time_ms = 0

        for idx, timing in enumerate(sorted_segments):
            # Extract data from SegmentTiming (use adjusted timestamps)
            start_ms = timing.adjusted_start_ms
            file_path = timing.tts_file
            actual_duration_sec = timing.tts_duration_sec

            # If there's a gap before this segment, add silence
            if start_ms > current_time_ms:
                gap_duration_sec = (start_ms - current_time_ms) / 1000.0
                silence_file = temp_dir / f"sil_{idx}.wav"

                # Generate silence file
                cmd_silence = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi',
                    '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-t', str(gap_duration_sec),
                    '-c:a', 'pcm_s16le',
                    str(silence_file)
                ]
                subprocess.run(cmd_silence, capture_output=True, text=True, timeout=30, check=True)
                all_segments.append(str(silence_file))

            # Convert TTS file to WAV format for concat
            wav_file = temp_dir / f"s_{idx}.wav"
            cmd_convert = [
                'ffmpeg', '-y',
                '-i', str(file_path),
                '-ar', '44100',
                '-ac', '2',
                '-c:a', 'pcm_s16le',
                str(wav_file)
            ]
            subprocess.run(cmd_convert, capture_output=True, text=True, timeout=30, check=True)
            all_segments.append(str(wav_file))

            # Update current time - use actual TTS duration
            actual_duration_ms = int(actual_duration_sec * 1000)
            current_time_ms = start_ms + actual_duration_ms

        # Add final silence to reach total duration
        if current_time_ms < total_duration_ms:
            final_gap_sec = (total_duration_ms - current_time_ms) / 1000.0
            final_silence_file = temp_dir / "sil_f.wav"

            cmd_final_silence = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                '-t', str(final_gap_sec),
                '-c:a', 'pcm_s16le',
                str(final_silence_file)
            ]
            subprocess.run(cmd_final_silence, capture_output=True, text=True, timeout=30, check=True)
            all_segments.append(str(final_silence_file))

        # Create concat file list (short path)
        concat_file = temp_dir / "c.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for seg_file in all_segments:
                # Use forward slashes for ffmpeg compatibility
                seg_file_unix = seg_file.replace('\\', '/')
                f.write(f"file '{seg_file_unix}'\n")

        # Use concat demuxer to combine all files
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c:a', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
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
    except subprocess.CalledProcessError as e:
        return False, f"Błąd przy przetwarzaniu segmentów TTS: {e}"
    except Exception as e:
        return False, f"Błąd przy tworzeniu ścieżki TTS: {str(e)}"
