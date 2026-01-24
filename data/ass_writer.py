#!/usr/bin/env python3
"""
ASS (Advanced SubStation Alpha) file writer for dual-language subtitles.

This module provides functionality for writing dual-language subtitle files
in ASS format with custom styles for original and translated text.
"""

from typing import List, Tuple


def format_ass_timestamp(ms: int) -> str:
    """
    Convert milliseconds to ASS timestamp format (H:MM:SS.CS).

    ASS format uses centiseconds (CS) instead of milliseconds,
    and doesn't use leading zeros for hours.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted timestamp string in ASS format (e.g., "0:01:23.45")
    """
    total_seconds = ms // 1000
    centiseconds = (ms % 1000) // 10  # Convert milliseconds to centiseconds

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def write_dual_language_ass(
    original_segments: List[Tuple[int, int, str]],
    translated_segments: List[Tuple[int, int, str]],
    output_path: str
) -> Tuple[bool, str]:
    """
    Write dual-language ASS subtitle file with two styles.

    Creates an ASS file with:
    - Original text in yellow (higher position, MarginV=50)
    - Translated text in white (lower position, MarginV=20)
    - Both using 14pt Arial font with opaque box background

    Args:
        original_segments: List of (start_ms, end_ms, text) tuples for original language
        translated_segments: List of (start_ms, end_ms, text) tuples for translation
        output_path: Path to the output ASS file

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate segment counts
        if len(original_segments) != len(translated_segments):
            print(f"Ostrzeżenie: Różna liczba segmentów - "
                  f"oryginał: {len(original_segments)}, tłumaczenie: {len(translated_segments)}")
            min_len = min(len(original_segments), len(translated_segments))
            original_segments = original_segments[:min_len]
            translated_segments = translated_segments[:min_len]

        if not original_segments:
            print("Ostrzeżenie: Brak segmentów do zapisania")
            # Create empty ASS file with warning comment
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write("; Pusty plik ASS - brak segmentów transkrypcji\n")
            return True, f"Zapisano pusty plik ASS (brak segmentów): {output_path}"

        with open(output_path, 'w', encoding='utf-8-sig') as f:
            # ===== SCRIPT INFO =====
            f.write("[Script Info]\n")
            f.write("Title: Dual Language Subtitles\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("\n")

            # ===== V4+ STYLES =====
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")

            # Style: Original (yellow, higher position)
            f.write("Style: Original,Arial,12,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,4,0,0,2,10,10,50,1\n")

            # Style: Translation (white, lower position)
            f.write("Style: Translation,Arial,12,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,4,0,0,2,10,10,20,1\n")
            f.write("\n")

            # ===== EVENTS =====
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            # Write dialogue lines
            for (orig_start, orig_end, orig_text), (trans_start, trans_end, trans_text) in zip(
                original_segments, translated_segments
            ):
                # Original language subtitle (yellow, higher)
                f.write(
                    f"Dialogue: 0,{format_ass_timestamp(orig_start)},{format_ass_timestamp(orig_end)},"
                    f"Original,,0,0,0,,{orig_text}\n"
                )

                # Translated subtitle (white, lower)
                f.write(
                    f"Dialogue: 0,{format_ass_timestamp(trans_start)},{format_ass_timestamp(trans_end)},"
                    f"Translation,,0,0,0,,{trans_text}\n"
                )

        return True, f"Zapisano {len(original_segments)} par napisów do pliku ASS: {output_path}"

    except Exception as e:
        return False, f"Błąd przy zapisie pliku ASS: {str(e)}"
