#!/usr/bin/env python3
"""
SRT file writer for transcription segments.

This module provides functionality for writing transcription segments
to SRT subtitle files with proper formatting.
"""

from typing import List, Tuple
from .segment_processor import format_srt_timestamp


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
