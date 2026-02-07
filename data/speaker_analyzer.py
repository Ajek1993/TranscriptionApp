#!/usr/bin/env python3
"""
Speaker Gender Analyzer Module

This module provides voice-based gender detection for speaker diarization.
Uses fundamental frequency (F0/pitch) analysis via librosa.

Gender classification is based on typical pitch ranges:
- Male voice: 85-180 Hz
- Female voice: 165-255 Hz
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .output_manager import OutputManager


# Pitch thresholds for gender classification (Hz)
MALE_F0_MIN = 85
MALE_F0_MAX = 180
FEMALE_F0_MIN = 165
FEMALE_F0_MAX = 255

# Threshold for classification - midpoint between typical ranges
GENDER_THRESHOLD_HZ = 165  # Below = male, above = female


def analyze_speaker_gender(
    audio_path: str,
    segments: List[Tuple[int, int, str]],
    sr: int = 22050
) -> Dict[str, dict]:
    """
    Analyze speaker gender from audio segments with diarization.

    Args:
        audio_path: Path to audio file (WAV)
        segments: List of (start_ms, end_ms, text) with speaker labels in text
        sr: Sample rate for audio loading

    Returns:
        Dict mapping speaker ID to info dict:
        {
            "SPEAKER_00": {"gender": "male", "pitch_mean": 120.5, "confidence": 0.85},
            "SPEAKER_01": {"gender": "female", "pitch_mean": 210.3, "confidence": 0.92}
        }
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        OutputManager.warning(
            "librosa nie jest zainstalowany. Wykrywanie płci niemożliwe. "
            "Zainstaluj: pip install librosa"
        )
        return {}

    # Extract unique speakers from segments
    speakers = _extract_speakers(segments)
    if not speakers:
        OutputManager.info("Brak informacji o mówcach w segmentach - pomijanie analizy płci")
        return {}

    OutputManager.info(f"Analiza płci dla {len(speakers)} mówców...")

    # Load audio file
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        OutputManager.warning(f"Nie można załadować audio do analizy płci: {e}")
        return {}

    # Collect pitch values for each speaker
    speaker_pitches: Dict[str, List[float]] = defaultdict(list)

    for start_ms, end_ms, text in segments:
        speaker = _get_speaker_from_text(text)
        if not speaker:
            continue

        # Convert ms to samples
        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)

        # Ensure valid range
        if start_sample >= len(audio) or end_sample <= start_sample:
            continue

        end_sample = min(end_sample, len(audio))
        segment_audio = audio[start_sample:end_sample]

        # Skip very short segments (< 0.5s)
        if len(segment_audio) < sr * 0.5:
            continue

        # Extract pitch using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment_audio,
                fmin=MALE_F0_MIN,
                fmax=FEMALE_F0_MAX,
                sr=sr
            )

            # Filter valid pitch values
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                speaker_pitches[speaker].extend(valid_f0.tolist())

        except Exception as e:
            # Silent continue - some segments may fail
            continue

    # Classify each speaker based on aggregated pitch
    speaker_info = {}

    for speaker, pitches in speaker_pitches.items():
        if not pitches:
            speaker_info[speaker] = {
                "gender": "unknown",
                "pitch_mean": None,
                "confidence": 0.0
            }
            continue

        pitch_mean = float(np.median(pitches))  # Use median for robustness
        pitch_std = float(np.std(pitches)) if len(pitches) > 1 else 0

        # Classify gender
        if pitch_mean < GENDER_THRESHOLD_HZ:
            gender = "male"
            # Confidence based on how far from threshold
            distance_from_threshold = GENDER_THRESHOLD_HZ - pitch_mean
            max_distance = GENDER_THRESHOLD_HZ - MALE_F0_MIN
        else:
            gender = "female"
            distance_from_threshold = pitch_mean - GENDER_THRESHOLD_HZ
            max_distance = FEMALE_F0_MAX - GENDER_THRESHOLD_HZ

        # Calculate confidence (0.5 to 1.0)
        confidence = 0.5 + 0.5 * min(distance_from_threshold / max_distance, 1.0)

        # Reduce confidence if high variance
        if pitch_std > 30:  # High variance in pitch
            confidence *= 0.8

        speaker_info[speaker] = {
            "gender": gender,
            "pitch_mean": round(pitch_mean, 1),
            "confidence": round(confidence, 2)
        }

        gender_pl = "mężczyzna" if gender == "male" else "kobieta"
        OutputManager.info(
            f"  {speaker}: {gender_pl} (pitch: {pitch_mean:.1f} Hz, pewność: {confidence:.0%})"
        )

    # Add unknown entries for speakers without pitch data
    for speaker in speakers:
        if speaker not in speaker_info:
            speaker_info[speaker] = {
                "gender": "unknown",
                "pitch_mean": None,
                "confidence": 0.0
            }

    return speaker_info


def _extract_speakers(segments: List[Tuple[int, int, str]]) -> List[str]:
    """Extract unique speaker IDs from segment texts."""
    speakers = set()
    for _, _, text in segments:
        speaker = _get_speaker_from_text(text)
        if speaker:
            speakers.add(speaker)
    return sorted(speakers)


def _get_speaker_from_text(text: str) -> Optional[str]:
    """Extract speaker ID from text like '[SPEAKER_00] Hello'."""
    # Match patterns like [SPEAKER_00], [Speaker 0], etc.
    match = re.match(r'\[(SPEAKER[_\s]?\d+|Speaker\s*\d+)\]', text, re.IGNORECASE)
    if match:
        # Normalize speaker ID format
        speaker_id = match.group(1).upper().replace(' ', '_')
        if not speaker_id.startswith('SPEAKER_'):
            speaker_id = 'SPEAKER_' + speaker_id.replace('SPEAKER', '').strip('_')
        return speaker_id
    return None


def format_speaker_info_for_prompt(speaker_info: Dict[str, dict]) -> str:
    """
    Format speaker info for LLM translation prompt.

    Returns string like:
    - SPEAKER_00: mężczyzna (on)
    - SPEAKER_01: kobieta (ona)
    """
    if not speaker_info:
        return ""

    lines = []
    for speaker, info in sorted(speaker_info.items()):
        gender = info.get("gender", "unknown")
        if gender == "male":
            lines.append(f"- {speaker}: mężczyzna (on)")
        elif gender == "female":
            lines.append(f"- {speaker}: kobieta (ona)")
        else:
            lines.append(f"- {speaker}: płeć nieznana (on/ona)")

    return "\n".join(lines)
