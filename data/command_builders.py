"""
Command Builders Module
Builds command-line arguments for ffmpeg, ffprobe, and yt-dlp.
"""

from pathlib import Path


def build_ffprobe_audio_info_cmd(file_path: str) -> list:
    """Build ffprobe command to get audio info (channels, sample_rate)."""
    return [
        'ffprobe', '-v', 'error', '-show_entries',
        'stream=channels,sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1:noescapes=1',
        str(file_path)
    ]


def build_ffprobe_video_info_cmd(file_path: str) -> list:
    """Build ffprobe command to get video info (width, height, codec)."""
    return [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]


def build_ffprobe_audio_streams_cmd(file_path: str) -> list:
    """Build ffprobe command to list all audio streams with language tags (JSON)."""
    return [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=index,codec_name,channels,bit_rate:stream_tags=language,title',
        '-of', 'json',
        str(file_path)
    ]


def build_ffprobe_duration_cmd(file_path: str) -> list:
    """Build ffprobe command to get file duration."""
    return [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]


def build_ffmpeg_audio_extraction_cmd(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    high_quality: bool = False,
    audio_stream_index: int = 0
) -> list:
    """
    Build ffmpeg command to extract audio as WAV.

    Args:
        input_path: Path to input video/audio file
        output_path: Path to output WAV file
        sample_rate: Sample rate in Hz (default: 16000 for transcription)
        channels: Number of audio channels (default: 1 for mono)
        high_quality: If True, extract in high quality (stereo, 48kHz, 24-bit)
                     for final video output. If False, extract in low quality
                     (mono, 16kHz, 16-bit) for Whisper transcription.
        audio_stream_index: Audio-relative stream index to extract (0 = first
                     audio track). Passed to ffmpeg as `-map 0:a:<index>`.
                     Without an explicit -map, ffmpeg picks whichever track
                     carries the `default` disposition, falling back to the one
                     with the most channels. On movie rips both rules tend to
                     land on a dubbed 5.1 track rather than the original.

    Returns:
        List of command arguments for ffmpeg
    """
    stream_map = ['-map', f'0:a:{int(audio_stream_index)}']

    if high_quality:
        # High quality for final video output - preserve original quality
        return [
            'ffmpeg',
            '-i', str(input_path),
            *stream_map,
            '-vn',  # No video
            '-acodec', 'pcm_s24le',  # PCM 24-bit
            '-ar', '48000',  # 48kHz sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output
            str(output_path)
        ]
    else:
        # Low quality for Whisper transcription - faster and sufficient
        return [
            'ffmpeg',
            '-i', str(input_path),
            *stream_map,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Channels
            '-y',  # Overwrite output
            str(output_path)
        ]


def build_ffmpeg_audio_split_cmd(
    input_path: str,
    output_path: str,
    start_time: int,
    duration: int
) -> list:
    """Build ffmpeg command to split audio into chunks."""
    return [
        'ffmpeg',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        str(output_path)
    ]


def build_ffmpeg_video_merge_cmd(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_bitrate: str = '320k'
) -> list:
    """
    Build ffmpeg command to merge video with audio track.

    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path to output video file
        audio_bitrate: Audio bitrate for AAC encoding (default: 320k for high quality)

    Returns:
        List of command arguments for ffmpeg
    """
    return [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', audio_bitrate,
        '-shortest',
        str(output_path)
    ]


def build_ffmpeg_subtitle_burn_cmd(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    subtitle_style: str,
    audio_bitrate: str = '320k'
) -> list:
    """
    Build ffmpeg command to burn subtitles into video.

    Supports both SRT and ASS subtitle formats:
    - SRT files: Applies force_style parameter for custom styling
    - ASS files: Preserves internal style definitions (no force_style)

    Args:
        video_path: Path to input video file
        subtitle_path: Path to subtitle file (.srt or .ass)
        output_path: Path to output video file
        subtitle_style: ASS style string (only applied to SRT files)
        audio_bitrate: Audio bitrate for AAC encoding (default: 320k)

    Returns:
        List of command arguments for ffmpeg
    """
    # Convert paths to absolute and escape for ffmpeg
    subtitle_path_abs = str(Path(subtitle_path).resolve())
    subtitle_path_filter = subtitle_path_abs.replace('\\', '/').replace(':', '\\:')

    # Detect file type and build appropriate filter
    file_extension = Path(subtitle_path).suffix.lower()

    if file_extension == '.ass':
        # ASS files: Use internal styles (no force_style)
        subtitles_filter = f"subtitles='{subtitle_path_filter}':charenc=UTF-8"
    else:
        # SRT files (or unknown): Apply custom style
        subtitles_filter = f"subtitles='{subtitle_path_filter}':force_style='{subtitle_style}':charenc=UTF-8"

    return [
        'ffmpeg', '-y',
        '-i', str(Path(video_path).resolve()),
        '-vf', subtitles_filter,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', audio_bitrate,
        str(output_path)
    ]


def build_ytdlp_audio_format_selector(audio_lang: str = None) -> str:
    """
    Build a yt-dlp format selector that avoids YouTube's dubbed audio tracks.

    YouTube serves multiple audio tracks (AI auto-dubbing) on a growing number
    of videos. A plain `bestaudio` selector can land on a dub, which then gets
    transcribed instead of the original speech.

    Args:
        audio_lang: ISO code of a specific track to force, or None for original
    """
    if audio_lang and audio_lang != "auto":
        preferred = f'ba[language^={audio_lang}]/'
    else:
        preferred = ''

    return (
        f'{preferred}'
        'ba[format_note*=original][ext=m4a]/'
        'ba[format_note*=original]/'
        'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best'
    )


def build_ytdlp_audio_download_cmd(url: str, output_file: str, audio_lang: str = None) -> list:
    """Build yt-dlp command to download audio only (2026 YouTube compatible)."""
    return [
        'yt-dlp',
        '-f', build_ytdlp_audio_format_selector(audio_lang),
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--restrict-filenames',
        '--no-progress',
        '--extractor-args', 'youtube:player_client=android_vr,web_safari',
        '-o', str(output_file),
        url
    ]


def build_ytdlp_video_download_cmd(
    url: str,
    output_file: str,
    quality: str = "1080",
    audio_lang: str = None
) -> list:
    """Build yt-dlp command to download video (2026 YouTube compatible)."""
    # Keep the audio track consistent with build_ytdlp_audio_download_cmd -
    # otherwise subtitles made from one track get muxed against another.
    audio = build_ytdlp_audio_format_selector(audio_lang)
    format_str = (
        f"bestvideo[height<={quality}][ext=mp4]+({audio})/"
        f"bestvideo[height<={quality}]+({audio})/"
        f"best[height<={quality}]/best"
    )

    return [
        'yt-dlp',
        '-f', format_str,
        '--merge-output-format', 'mp4',
        '--restrict-filenames',
        '--no-progress',
        '--extractor-args', 'youtube:player_client=android_vr,web_safari',
        '-o', str(output_file),
        url
    ]
