#!/usr/bin/env python3
"""
Audio Mixer Module

This module provides functionality for mixing audio tracks and creating final video outputs:
- Mix original audio with TTS audio tracks
- Create dubbed videos by combining video with mixed audio
- Burn subtitles into video files

All functions use ffmpeg for audio/video processing.
"""

import subprocess
from pathlib import Path
from typing import Tuple

from command_builders import build_ffmpeg_video_merge_cmd, build_ffmpeg_subtitle_burn_cmd


def mix_audio_tracks(
    original_audio_path: str,
    tts_audio_path: str,
    output_path: str,
    original_volume: float = 0.2,
    tts_volume: float = 1.0
) -> Tuple[bool, str]:
    """
    Mix audio using completely manual approach without amix filter.

    Args:
        original_audio_path: Path to original audio file
        tts_audio_path: Path to TTS audio file
        output_path: Path to save mixed audio
        original_volume: Volume multiplier for original audio (default: 0.2)
        tts_volume: Volume multiplier for TTS audio (default: 1.0)

    Returns:
        Tuple of (success: bool, message: str)
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
    subtitle_style: str = "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=20"
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
