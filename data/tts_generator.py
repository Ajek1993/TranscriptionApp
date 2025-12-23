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
from typing import Tuple, List
from tqdm import tqdm

from .device_manager import detect_device
from .audio_processor import get_audio_duration

# Na początku pliku, po importach:
TTS_SLOT_FILL_RATIO = 0.999  # Wypełnienie 99.9% czasu segmentu

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

                    # Check if TTS is too long for the slot and needs to be sped up
                    if tts_duration > slot_duration_sec:
                        # Target % of slot duration to leave a small gap
                        target_duration_sec = slot_duration_sec * TTS_SLOT_FILL_RATIO
                        speed_multiplier = tts_duration / target_duration_sec
                        speed_multiplier = max(1.0, min(speed_multiplier, 1.5))  # Only speed UP (1.0x-1.5x)

                        if engine == "edge":
                            rate_percent = int((speed_multiplier - 1.0) * 100)
                            rate_percent = min(rate_percent, 50)  # Cap at +50%
                            rate = f"+{rate_percent}%"
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {slot_duration_sec:.2f}s), przyspieszam do {rate}")
                        elif engine == "coqui":
                            speed = speed_multiplier
                            tqdm.write(f"Segment {idx}: TTS za długi ({tts_duration:.2f}s > {slot_duration_sec:.2f}s), przyspieszam do {speed:.2f}x")

                        # Regenerate with adjusted speed
                        if engine == "edge":
                            success, message, tts_duration = asyncio.run(
                                generate_tts_for_segment(text, str(tts_file), voice, rate)
                            )
                        elif engine == "coqui":
                            success, message, tts_duration = generate_tts_coqui_for_segment(
                                text, str(tts_file), coqui_model, coqui_speaker, speaker_wav, speed,
                                coqui_tts_instance, language=target_language
                            )

                        if not success:
                            if retry < max_retries - 1:
                                time.sleep(0.5 * (retry + 1))
                                continue
                            else:
                                tqdm.write(f"Ostrzeżenie: Nie udało się wygenerować TTS dla segmentu {idx}: {message}")
                                pbar.update(1)
                                break

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


def create_tts_audio_track(
    tts_files: List[Tuple[int, int, str, float]],
    total_duration_ms: int,
    output_path: str
) -> Tuple[bool, str]:
    """
    Combine TTS segments into a single audio track using concat demuxer.
    This avoids amix completely and ensures constant volume.

    Args:
        tts_files: List of (start_ms, end_ms, file_path, actual_duration_sec) tuples
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

        for idx, (start_ms, end_ms, file_path, actual_duration_sec) in enumerate(sorted_segments):
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

            # Update current time - use end_ms from segment to avoid timing drift
            slot_duration_ms = end_ms - start_ms
            current_time_ms = end_ms

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
