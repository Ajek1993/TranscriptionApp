#!/usr/bin/env python3
"""
YouTube Processor Module
Handles downloading audio/video from YouTube and extracting audio from video files.
"""

import re
import subprocess
from pathlib import Path
from typing import Tuple
import yt_dlp

from .command_builders import (
    build_ytdlp_audio_download_cmd,
    build_ytdlp_video_download_cmd,
    build_ffprobe_audio_info_cmd,
    build_ffprobe_video_info_cmd,
    build_ffmpeg_audio_extraction_cmd
)
from .output_manager import OutputManager
from .audio_processor import (
    list_audio_streams,
    format_audio_stream,
    select_audio_stream
)

def get_video_title(url: str) -> str:
    """Pobiera tytuł wideo z YouTube bez pobierania."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,  # Tłumi wewnętrzne ostrzeżenia yt-dlp
            'ignoreerrors': True,  # Ignoruje błędy cicho
            'extractor_args': {'youtube': {'player_client': ['android_vr,web_safari']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown').strip()
            # Czyszczenie tytułu (usuwanie niedozwolonych znaków)
            title = re.sub(r'[<>:"/\\|?*]', '_', title)[:200]  # max 200 znaków
            return title
    except:
        return None



def get_youtube_audio_languages(url: str) -> list:
    """
    List the audio track languages YouTube offers for a video.

    Videos with AI auto-dubbing expose several audio tracks; knowing about them
    is the difference between transcribing the original speech and transcribing
    a machine dub.

    Returns:
        Sorted list of language codes, empty if unavailable or single-track.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'extractor_args': {'youtube': {'player_client': ['android_vr,web_safari']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            languages = {
                fmt.get('language')
                for fmt in (info.get('formats') or [])
                if fmt.get('acodec') not in (None, 'none') and fmt.get('language')
            }
            return sorted(languages)
    except Exception:
        return []


def download_audio(url: str, output_dir: str = ".", audio_lang: str = None) -> Tuple[bool, str, str]:
    """
    Download audio from YouTube and save as WAV (mono, 16kHz).

    Args:
        url: YouTube URL
        output_dir: Directory to save the audio file
        audio_lang: Force a specific audio track language (None = original)

    Returns:
        Tuple of (success: bool, message: str, audio_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract video ID for naming
    # video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
    # if not video_id_match:
    #     return False, "Błąd: Nie udało się wyodrębnić ID wideo z URL", ""

    # video_id = video_id_match.group(1)
    # audio_file = output_path / f"{video_id}.wav"

    title = get_video_title(url)
    if not title:
        video_id = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
        title = video_id.group(1) if video_id else "audio"
    audio_file = output_path / f"{title}.wav"


    try:
        OutputManager.info(f"Pobieranie audio z YouTube... ({url})")

        available_langs = get_youtube_audio_languages(url)
        if len(available_langs) > 1:
            OutputManager.info(
                f"Film ma {len(available_langs)} ścieżek audio ({', '.join(available_langs)}) - "
                f"pobieram {audio_lang or 'oryginalną'}"
            )

        cmd = build_ytdlp_audio_download_cmd(url, str(audio_file), audio_lang=audio_lang)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd yt-dlp"
            if "private" in error_msg.lower() or "not available" in error_msg.lower():
                return False, "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie.", ""
            else:
                return False, f"Błąd yt-dlp: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        OutputManager.info(f"Audio pobrane: {audio_file}")

        # Verify audio format with ffprobe
        probe_cmd = build_ffprobe_audio_info_cmd(audio_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            channels = int(output_info[0]) if len(output_info) > 0 else 0
            sample_rate = int(output_info[1]) if len(output_info) > 1 else 0
            OutputManager.detail(f"Format audio: {channels} kanał(y), {sample_rate} Hz")

        return True, f"Audio pobrane pomyślnie: {audio_file}", str(audio_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Pobieranie przerwane (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy pobieraniu: {str(e)}", ""


def download_video(
    url: str,
    output_dir: str = ".",
    quality: str = "1080",
    audio_lang: str = None
) -> Tuple[bool, str, str]:
    """
    Download video from YouTube in specified quality.

    Args:
        url: YouTube URL
        output_dir: Directory to save the video file
        quality: Preferred video quality (default: "1080")
        audio_lang: Force a specific audio track language (None = original)

    Returns:
        Tuple of (success: bool, message: str, video_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract video ID for naming
    # video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
    # if not video_id_match:
    #     return False, "Błąd: Nie udało się wyodrębnić ID wideo z URL", ""

    # video_id = video_id_match.group(1)
    # video_file = output_path / f"{video_id}.mp4"

    title = get_video_title(url)
    if not title:
        video_id = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)', url)
        title = video_id.group(1) if video_id else "video"
    video_file = output_path / f"{title}.mp4"


    try:
        print(f"Pobieranie wideo z YouTube w jakości {quality}p... ({url})")

        cmd = build_ytdlp_video_download_cmd(url, str(video_file), quality, audio_lang=audio_lang)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd yt-dlp"
            if "private" in error_msg.lower() or "not available" in error_msg.lower():
                return False, "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie.", ""
            else:
                return False, f"Błąd yt-dlp: {error_msg}", ""

        if not video_file.exists():
            return False, f"Błąd: Plik wideo nie został utworzony: {video_file}", ""

        # Get video info with ffprobe
        probe_cmd = build_ffprobe_video_info_cmd(video_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            if len(output_info) >= 3:
                width = output_info[0]
                height = output_info[1]
                codec = output_info[2]
                print(f"Wideo pobrane: {width}x{height}, codec: {codec}")
            else:
                print(f"Wideo pobrane: {video_file}")
        else:
            print(f"Wideo pobrane: {video_file}")

        return True, f"Wideo pobrane pomyślnie: {video_file}", str(video_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Pobieranie przerwane (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy pobieraniu wideo: {str(e)}", ""


def extract_audio_from_video(
    video_path: str,
    output_dir: str = ".",
    high_quality: bool = False,
    audio_track: str = "auto",
    preferred_language: str = None
) -> Tuple[bool, str, str]:
    """
    Extract audio from a local video file and convert to WAV.

    Args:
        video_path: Path to the local video file
        output_dir: Directory to save the extracted audio
        high_quality: If True, extract in high quality (stereo, 48kHz, 24-bit)
                     for final video output. If False, extract in low quality
                     (mono, 16kHz, 16-bit) for Whisper transcription.
        audio_track: "auto" to pick the track automatically, or a track index
                     (audio-relative, as reported by list_audio_streams)
        preferred_language: ISO 639-1 code used by the automatic pick

    Returns:
        Tuple of (success: bool, message: str, audio_path: str)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on input video filename
    video_file = Path(video_path)
    suffix = "_hq" if high_quality else ""
    audio_file = output_path / f"{video_file.stem}{suffix}.wav"

    try:
        quality_desc = "wysokiej jakości" if high_quality else "dla transkrypcji"
        print(f"Ekstrakcja audio ({quality_desc}) z pliku wideo... ({video_path})")

        # Multi-track files (movie rips with dubs) need an explicit track choice
        streams = list_audio_streams(video_path)
        if len(streams) > 1:
            print(f"Wykryto {len(streams)} ścieżek audio:")
            for stream in streams:
                print(f"  {format_audio_stream(stream)}")

        if str(audio_track).lower() in ("auto", "", "none"):
            track_index, reason = select_audio_stream(streams, preferred_language)
        else:
            track_index = int(audio_track)
            reason = f"Wybrano ścieżkę #{track_index} (wskazana ręcznie)"

        if len(streams) > 1:
            print(reason)

        cmd = build_ffmpeg_audio_extraction_cmd(
            video_path, audio_file,
            high_quality=high_quality,
            audio_stream_index=track_index
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Nieznany błąd ffmpeg"
            return False, f"Błąd ffmpeg: {error_msg}", ""

        if not audio_file.exists():
            return False, f"Błąd: Plik audio nie został utworzony: {audio_file}", ""

        print(f"Audio wyekstrahowane: {audio_file}")

        # Verify audio format with ffprobe
        probe_cmd = build_ffprobe_audio_info_cmd(audio_file)
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode == 0:
            output_info = probe_result.stdout.strip().split('\n')
            channels = int(output_info[0]) if len(output_info) > 0 else 0
            sample_rate = int(output_info[1]) if len(output_info) > 1 else 0
            OutputManager.detail(f"Format audio: {channels} kanał(y), {sample_rate} Hz")

        message = f"Audio wyekstrahowane pomyślnie: {audio_file}"
        if len(streams) > 1:
            available = "; ".join(format_audio_stream(s) for s in streams)
            message += f"\nDostępne ścieżki audio: {available}\n{reason}"

        return True, message, str(audio_file)

    except subprocess.TimeoutExpired:
        return False, "Błąd: Ekstrakcja audio przerwana (timeout). Spróbuj ponownie.", ""
    except Exception as e:
        return False, f"Błąd przy ekstrakcji audio: {str(e)}", ""
