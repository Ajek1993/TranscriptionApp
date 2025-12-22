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
    channels: int = 1
) -> list:
    """Build ffmpeg command to extract audio as WAV."""
    return [
        'ffmpeg',
        '-i', str(input_path),
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
    output_path: str
) -> list:
    """Build ffmpeg command to merge video with audio track."""
    return [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        str(output_path)
    ]


def build_ffmpeg_subtitle_burn_cmd(
    video_path: str,
    srt_path: str,
    output_path: str,
    subtitle_style: str
) -> list:
    """Build ffmpeg command to burn subtitles into video."""
    # Convert paths to absolute and escape for ffmpeg
    srt_path_abs = str(Path(srt_path).resolve())
    srt_path_filter = srt_path_abs.replace('\\', '/').replace(':', '\\:')

    # Build subtitles filter with custom style
    subtitles_filter = f"subtitles='{srt_path_filter}':force_style='{subtitle_style}':charenc=UTF-8"

    return [
        'ffmpeg', '-y',
        '-i', str(Path(video_path).resolve()),
        '-vf', subtitles_filter,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'copy',
        str(output_path)
    ]


def build_ytdlp_audio_download_cmd(url: str, output_file: str) -> list:
    """Build yt-dlp command to download audio only."""
    return [
        'yt-dlp',
        '-f', 'bestaudio/best',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--restrict-filenames',
        '-o', str(output_file),
        url
    ]


def build_ytdlp_video_download_cmd(url: str, output_file: str, quality: str = "1080") -> list:
    """Build yt-dlp command to download video."""
    format_str = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"

    return [
        'yt-dlp',
        '-f', format_str,
        '--merge-output-format', 'mp4',
        '--restrict-filenames',
        '-o', str(output_file),
        url
    ]
