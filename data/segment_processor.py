#!/usr/bin/env python3
"""
Segment processing utilities for transcription segments.

This module provides functions for:
- Splitting long segments into smaller chunks
- Filling gaps in timestamps
- Formatting timestamps for SRT output
"""

import re
from typing import List, Tuple


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
