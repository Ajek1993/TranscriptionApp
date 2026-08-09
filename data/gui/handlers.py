#!/usr/bin/env python3
"""
GUI Handlers Module
Contains event handlers for Gradio interface.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, Optional, List

# Import required modules from parent data/ directory
from ..validators import validate_youtube_url, validate_video_file, validate_youtube_url_with_message, validate_file_extension, validate_srt_file
from ..youtube_processor import download_audio, extract_audio_from_video
from ..transcription_engines import transcribe_chunk
from ..translation import translate_segments
from ..srt_writer import write_srt
from ..device_manager import clear_cuda_cache
import tempfile
import shutil
import os

# Absolute path to output files directory
FILES_DIR = Path(__file__).parent.parent.parent / "files"
FILES_DIR.mkdir(exist_ok=True)  # Ensure directory exists


def cleanup_gradio_temp():
    """Clean up Gradio temporary upload files."""
    gradio_temp = Path(tempfile.gettempdir()) / "gradio"
    if gradio_temp.exists():
        for item in gradio_temp.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                pass


def _expected_audio_language(language: str, source_lang: str, enable_translation: bool) -> Optional[str]:
    """
    Work out which language the audio is expected to be in.

    Used to pick the right track on multi-track files: the Whisper language
    setting is the strongest signal, the translation source language the next
    best. "auto" on both means we have nothing to go on.
    """
    if language and language != "auto":
        return language
    if enable_translation and source_lang and source_lang != "auto":
        return source_lang
    return None


def handle_transcription(
    source_type: str,
    youtube_url: str,
    local_file: Optional[object],
    model: str,
    language: str,
    engine: str,
    enable_translation: bool,
    source_lang: str,
    target_lang: str,
    device: str,
    timeout: int,
    whisperx_align: bool,
    whisperx_diarize: bool,
    use_llm_translate: bool = False,
    llm_provider: str = None,
    llm_model: str = None,
    llm_base_url: str = None,
    llm_api_key: str = None,
    detect_gender: bool = False,
    audio_track: str = "auto",
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """
    Handler for transcription tab.

    Args:
        source_type: "YouTube URL" or "Plik lokalny"
        youtube_url: YouTube URL string
        local_file: Uploaded file object
        model: Whisper model size
        language: Language code (or "auto")
        engine: Transcription engine ("whisper" or "whisperx")
        enable_translation: Whether to enable translation
        source_lang: Source language for translation
        target_lang: Target language for translation
        device: Device to use ("auto", "cuda", "cpu")
        timeout: Timeout in seconds
        whisperx_align: Enable WhisperX alignment
        whisperx_diarize: Enable WhisperX diarization
        use_llm_translate: Use LLM for translation instead of Google Translator
        llm_provider: LLM provider (openai, ollama, anthropic)
        llm_model: LLM model name
        llm_base_url: LLM API base URL
        llm_api_key: LLM API key
        detect_gender: Detect speaker gender for proper grammatical inflection
        audio_track: Audio track to transcribe ("auto" or a track index)
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message: str, srt_file_path: Optional[str])
    """
    status_messages = []
    temp_files = []

    try:
        # Clean CUDA cache before starting
        clear_cuda_cache()
        # === 1. Walidacja i przygotowanie źródła ===
        progress(0, desc="Walidacja źródła...")
        status_messages.append("=== Walidacja źródła ===")

        audio_file_path = None

        if source_type == "YouTube URL":
            # Validate YouTube URL
            if not youtube_url or not youtube_url.strip():
                return "Błąd: Proszę podać URL YouTube.", None

            if not validate_youtube_url(youtube_url):
                return "Błąd: Nieprawidłowy URL YouTube.", None

            status_messages.append(f"YouTube URL: {youtube_url}")
            status_messages.append("\n=== Pobieranie audio ===")
            progress(0.1, desc="Pobieranie audio z YouTube...")

            # Download audio from YouTube - original track, never an AI dub
            success, message, audio_path = download_audio(
                youtube_url,
                output_dir="files",
                audio_lang=_expected_audio_language(language, source_lang, enable_translation)
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            audio_file_path = audio_path
            temp_files.append(audio_path)

        else:  # Plik lokalny
            if local_file is None:
                return "Błąd: Proszę wybrać plik lokalny.", None

            # Get file path from Gradio file object
            file_path = local_file.name if hasattr(local_file, 'name') else str(local_file)

            # Validate video file
            is_valid, result = validate_video_file(file_path)

            if not is_valid:
                return f"Błąd walidacji pliku: {result}", None

            validated_path = result
            status_messages.append(f"Plik lokalny: {validated_path}")
            status_messages.append("\n=== Ekstrakcja audio ===")
            progress(0.1, desc="Ekstrakcja audio z pliku...")

            # Extract audio from video - explicit track choice, so a dubbed
            # track never wins over the original
            success, message, audio_path = extract_audio_from_video(
                validated_path,
                output_dir="files",
                audio_track=audio_track,
                preferred_language=_expected_audio_language(language, source_lang, enable_translation)
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            audio_file_path = audio_path
            temp_files.append(audio_path)

        # === 2. Transkrypcja ===
        progress(0.2, desc="Transkrypcja audio...")
        status_messages.append("\n=== Transkrypcja ===")

        # Build options summary
        options_parts = [f"{engine}/{model}"]
        if engine == "whisperx" and whisperx_align:
            options_parts.append("align")
        if engine == "whisperx" and whisperx_diarize:
            options_parts.append("diarize")
        if enable_translation:
            trans_dir = f"{source_lang.upper()}→{target_lang.upper()}"
            if use_llm_translate:
                llm_info = f"LLM ({llm_provider or 'auto'})"
                if detect_gender and whisperx_diarize:
                    llm_info += " + gender"
                options_parts.append(f"{trans_dir} via {llm_info}")
            else:
                options_parts.append(f"{trans_dir} via Google")

        status_messages.append(f"Opcje: {' | '.join(options_parts)}")

        # Create a simple progress indicator (tqdm won't work in Gradio, but we keep the interface)
        pbar = None

        # Determine if we need gender detection
        should_detect_gender = detect_gender and whisperx_diarize

        result = transcribe_chunk(
            wav_path=audio_file_path,
            model_size=model,
            language=language if language != "auto" else None,
            engine=engine,
            segment_progress_bar=pbar,
            timeout_seconds=int(timeout),
            whisperx_align=whisperx_align,
            whisperx_diarize=whisperx_diarize,
            hf_token=os.environ.get("HF_TOKEN", "") or None,
            force_device=device,
            detect_gender=should_detect_gender
        )

        # Handle different return formats based on detect_gender
        if should_detect_gender:
            success, message, segments, speaker_info = result
            if speaker_info:
                status_messages.append(f"Wykryto {len(speaker_info)} mówców z informacją o płci")
        else:
            success, message, segments = result
            speaker_info = {}

        status_messages.append(message)

        if not success:
            return "\n".join(status_messages), None

        if not segments:
            return "\n".join(status_messages) + "\nOstrzeżenie: Brak segmentów do zapisania.", None

        # === 3. Tłumaczenie (opcjonalne) ===
        final_segments = segments

        if enable_translation:
            progress(0.7, desc="Tłumaczenie tekstu...")
            status_messages.append("\n=== Tłumaczenie ===")
            status_messages.append(f"Kierunek: {source_lang} -> {target_lang}")

            if use_llm_translate:
                status_messages.append(f"Tryb: LLM ({llm_provider or 'auto'} / {llm_model or 'auto'})")
                if speaker_info:
                    status_messages.append(f"Informacje o mówcach: {len(speaker_info)} osób")

            success, message, translated_segments, detected_lang = translate_segments(
                segments=segments,
                source_lang=source_lang,
                target_lang=target_lang,
                use_llm=use_llm_translate,
                speaker_info=speaker_info if use_llm_translate else None,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key
            )

            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            final_segments = translated_segments

        # === 4. Zapis do pliku SRT ===
        progress(0.9, desc="Zapisywanie pliku SRT...")
        status_messages.append("\n=== Zapis pliku SRT ===")

        # Generate output filename
        audio_stem = Path(audio_file_path).stem
        output_srt = FILES_DIR / f"{audio_stem}.srt"
        output_srt.parent.mkdir(parents=True, exist_ok=True)

        success, message = write_srt(final_segments, str(output_srt))
        status_messages.append(message)

        if not success:
            return "\n".join(status_messages), None

        # === 5. Sukces ===
        progress(1.0, desc="Gotowe!")
        status_messages.append("\n=== ✓ Transkrypcja zakończona pomyślnie ===")
        status_messages.append(f"Plik SRT: {output_srt}")

        return "\n".join(status_messages), str(output_srt)

    except Exception as e:
        error_msg = f"\n\n!!! BŁĄD !!!\n{str(e)}"
        status_messages.append(error_msg)
        return "\n".join(status_messages), None
    finally:
        # Clean up temporary files and directories
        for temp_file in temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path)
                    else:
                        temp_path.unlink()
            except Exception:
                pass
        # Clean up Gradio temp files
        cleanup_gradio_temp()
        # Clear CUDA cache after operation
        clear_cuda_cache()


def handle_dubbing(
    source_type: str,
    youtube_url: str,
    video_file: Optional[object],
    use_srt: bool,
    srt_file: Optional[object],
    enable_tts_dubbing: bool,
    burn_subtitles: bool,
    dubbing_type: str,
    bilingual_subtitles: bool,
    tts_engine: str,
    tts_voice: str,
    tts_volume: float,
    original_volume: float,
    narrator_mode: bool,
    merge_gap: int,
    max_segment_length: int,
    max_words_segment: int,
    fill_gaps: bool,
    min_pause: int,
    max_gap: int,
    video_quality: str,
    # Parametry transkrypcji z zakładki Transkrypcja
    transcribe_model: str,
    transcribe_language: str,
    transcribe_engine: str,
    transcribe_enable_translation: bool,
    transcribe_source_lang: str,
    transcribe_target_lang: str,
    transcribe_device: str,
    transcribe_timeout: int,
    transcribe_whisperx_align: bool,
    transcribe_whisperx_diarize: bool,
    # Parametry LLM z zakładki Transkrypcja
    transcribe_use_llm: bool = False,
    transcribe_llm_provider: str = None,
    transcribe_llm_model: str = None,
    transcribe_llm_base_url: str = None,
    transcribe_llm_api_key: str = None,
    transcribe_detect_gender: bool = False,
    # Korekta SRT przed dubbingiem
    correct_srt: bool = False,
    # Wybór ścieżki audio dla plików wielościeżkowych
    audio_track: str = "auto",
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """
    Handler for dubbing/subtitles tab.

    Args:
        source_type: "YouTube URL" or "Plik lokalny" (video source only)
        youtube_url: YouTube URL string
        video_file: Uploaded video/audio file
        use_srt: Whether to use provided SRT file (False = auto-transcription)
        srt_file: Uploaded SRT file (used only if use_srt is True)
        enable_tts_dubbing: Whether to enable TTS dubbing
        burn_subtitles: Whether to burn subtitles into video
        dubbing_type: "Wideo" or "Tylko audio WAV"
        bilingual_subtitles: Whether to create dual-language subtitles
        tts_engine: TTS engine ("edge" or "coqui")
        tts_voice: Voice/model name
        tts_volume: TTS volume (0.0 - 2.0)
        original_volume: Original audio volume (0.0 - 1.0)
        narrator_mode: Enable narrator mode (merge segments)
        merge_gap: Maximum gap to merge segments (ms)
        max_segment_length: Maximum segment length (s)
        max_words_segment: Maximum words in segment
        fill_gaps: Fill gaps between segments
        min_pause: Minimum pause (ms)
        max_gap: Maximum gap (ms)
        video_quality: Video quality for YouTube downloads
        transcribe_model: Whisper model size for auto-transcription
        transcribe_language: Language code for auto-transcription
        transcribe_engine: Transcription engine for auto-transcription
        transcribe_enable_translation: Whether to enable translation for auto-transcription
        transcribe_source_lang: Source language for translation
        transcribe_target_lang: Target language for translation
        transcribe_device: Device to use for auto-transcription
        transcribe_timeout: Timeout for auto-transcription
        transcribe_whisperx_align: Enable WhisperX alignment for auto-transcription
        transcribe_whisperx_diarize: Enable WhisperX diarization for auto-transcription
        transcribe_use_llm: Use LLM for translation instead of Google Translator
        transcribe_llm_provider: LLM provider (openai, ollama, anthropic)
        transcribe_llm_model: LLM model name
        transcribe_llm_base_url: LLM API base URL
        transcribe_llm_api_key: LLM API key
        transcribe_detect_gender: Detect speaker gender for proper grammatical inflection
        correct_srt: Whether to run LLM correction on SRT before dubbing
        audio_track: Audio track to transcribe ("auto" or a track index)
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message: str, output_file_path: Optional[str])
    """
    from ..youtube_processor import download_video
    from ..srt_reader import parse_srt_file
    from ..tts_generator import generate_tts_segments, create_tts_audio_track, smart_adjust_timestamps
    from ..audio_mixer import mix_audio_tracks, create_dubbed_video, burn_subtitles_to_video
    from ..segment_processor import merge_segments_for_narrator
    from ..ass_writer import write_dual_language_ass
    from ..audio_processor import get_audio_duration_ms
    from ..srt_corrector import correct_srt_with_llm, write_corrections_log

    status_messages = []
    temp_files = []

    try:
        # Clean CUDA cache before starting
        clear_cuda_cache()
        # === 1. Walidacja i przygotowanie źródła ===
        progress(0, desc="Walidacja źródła...")
        status_messages.append("=== Walidacja źródła ===")

        video_path = None
        srt_path = None
        segments = None

        if source_type == "YouTube URL":
            # Validate and download from YouTube
            if not youtube_url or not youtube_url.strip():
                return "Błąd: Proszę podać URL YouTube.", None

            if not validate_youtube_url(youtube_url):
                return "Błąd: Nieprawidłowy URL YouTube.", None

            status_messages.append(f"YouTube URL: {youtube_url}")
            status_messages.append("\n=== Pobieranie wideo ===")
            progress(0.05, desc="Pobieranie wideo z YouTube...")

            # Download video - original audio track, never an AI dub
            success, message, video_path_result = download_video(
                youtube_url,
                output_dir="files",
                quality=video_quality,
                audio_lang=_expected_audio_language(
                    transcribe_language, transcribe_source_lang, transcribe_enable_translation
                )
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            video_path = video_path_result
            temp_files.append(video_path)

        elif source_type == "Plik lokalny":
            # Use local video/audio file
            if video_file is None:
                return "Błąd: Proszę wybrać plik lokalny.", None

            file_path = video_file.name if hasattr(video_file, 'name') else str(video_file)

            is_valid, result = validate_video_file(file_path)
            if not is_valid:
                return f"Błąd walidacji pliku: {result}", None

            video_path = result
            status_messages.append(f"Plik lokalny: {video_path}")

        # Nowa logika SRT - walidacja checkboxa i pliku
        if use_srt:
            # Użytkownik chce użyć pliku SRT
            if srt_file is None:
                return "Błąd: Checkbox SRT zaznaczony, ale nie wybrano pliku.", None
            srt_file_path = srt_file.name if hasattr(srt_file, 'name') else str(srt_file)
            srt_path = srt_file_path
            status_messages.append(f"Użyto pliku SRT: {srt_path}")
        # else: use_srt jest False - nastąpi automatyczna transkrypcja w kolejnym kroku

        # === 2. Przygotowanie napisów SRT ===
        if use_srt and srt_path:
            # Wczytaj istniejący plik SRT
            progress(0.15, desc="Wczytywanie pliku SRT...")
            status_messages.append("\n=== Wczytywanie pliku SRT ===")
            success, message, segments = parse_srt_file(srt_path)
            status_messages.append(message)
            if not success:
                return "\n".join(status_messages), None

            # Korekta SRT przez LLM (opcjonalna)
            if correct_srt:
                progress(0.17, desc="Korekta SRT przez LLM...")
                status_messages.append("\n=== Korekta SRT przez LLM ===")

                success, message, corrected_segments, changes = correct_srt_with_llm(
                    segments=segments,
                    provider=transcribe_llm_provider,
                    model=transcribe_llm_model,
                    base_url=transcribe_llm_base_url,
                    api_key=transcribe_llm_api_key
                )
                status_messages.append(message)

                if success:
                    segments = corrected_segments
                    status_messages.append(f"Liczba poprawek: {len(changes)}")

                    # Zapisz log zmian
                    if changes:
                        srt_filename = Path(srt_path).stem
                        corrections_log_path = FILES_DIR / f"{srt_filename}_corrections.txt"
                        write_corrections_log(
                            changes=changes,
                            source_filename=Path(srt_path).name,
                            output_path=str(corrections_log_path),
                            provider=transcribe_llm_provider,
                            model=transcribe_llm_model
                        )
                        status_messages.append(f"Log zmian: {corrections_log_path}")
                else:
                    status_messages.append("Ostrzeżenie: Korekta nieudana, kontynuuję z oryginalnym SRT")
        else:
            # AUTO-TRANSKRYPCJA
            progress(0.15, desc="Automatyczna transkrypcja...")
            status_messages.append("\n=== Automatyczna transkrypcja ===")

            # Build options summary
            options_parts = [f"{transcribe_engine}/{transcribe_model}"]
            if transcribe_engine == "whisperx" and transcribe_whisperx_align:
                options_parts.append("align")
            if transcribe_engine == "whisperx" and transcribe_whisperx_diarize:
                options_parts.append("diarize")
            if transcribe_enable_translation:
                trans_dir = f"{transcribe_source_lang.upper()}→{transcribe_target_lang.upper()}"
                if transcribe_use_llm:
                    llm_info = f"LLM ({transcribe_llm_provider or 'auto'})"
                    if transcribe_detect_gender and transcribe_whisperx_diarize:
                        llm_info += " + gender"
                    options_parts.append(f"{trans_dir} via {llm_info}")
                else:
                    options_parts.append(f"{trans_dir} via Google")

            status_messages.append(f"Opcje: {' | '.join(options_parts)}")

            # Ekstrakcja audio z wideo
            status_messages.append("\n--- Ekstrakcja audio ---")
            success, message, audio_path = extract_audio_from_video(
                video_path,
                output_dir="files",
                audio_track=audio_track,
                preferred_language=_expected_audio_language(
                    transcribe_language, transcribe_source_lang, transcribe_enable_translation
                )
            )
            status_messages.append(message)
            if not success:
                return "\n".join(status_messages), None
            temp_files.append(audio_path)

            # Transkrypcja
            progress(0.25, desc="Transkrypcja audio...")

            # Determine if we need gender detection
            should_detect_gender = transcribe_detect_gender and transcribe_whisperx_diarize and transcribe_use_llm

            result = transcribe_chunk(
                wav_path=audio_path,
                model_size=transcribe_model,
                language=transcribe_language if transcribe_language != "auto" else None,
                engine=transcribe_engine,
                segment_progress_bar=None,
                timeout_seconds=int(transcribe_timeout),
                whisperx_align=transcribe_whisperx_align,
                whisperx_diarize=transcribe_whisperx_diarize,
                hf_token=os.environ.get("HF_TOKEN", "") or None,
                force_device=transcribe_device,
                detect_gender=should_detect_gender
            )

            # Handle different return formats based on detect_gender
            if should_detect_gender:
                success, message, segments, speaker_info = result
                if speaker_info:
                    status_messages.append(f"Wykryto {len(speaker_info)} mówców z informacją o płci")
            else:
                success, message, segments = result
                speaker_info = {}

            status_messages.append(message)
            if not success or not segments:
                return "\n".join(status_messages), None

            # Tłumaczenie (opcjonalne)
            if transcribe_enable_translation:
                progress(0.35, desc="Tłumaczenie...")
                status_messages.append("\n--- Tłumaczenie ---")

                if transcribe_use_llm:
                    status_messages.append(f"Tryb: LLM ({transcribe_llm_provider or 'auto'} / {transcribe_llm_model or 'auto'})")
                    if speaker_info:
                        status_messages.append(f"Informacje o mówcach: {len(speaker_info)} osób")

                success, message, segments, detected_lang = translate_segments(
                    segments=segments,
                    source_lang=transcribe_source_lang,
                    target_lang=transcribe_target_lang,
                    use_llm=transcribe_use_llm,
                    speaker_info=speaker_info if transcribe_use_llm else None,
                    llm_provider=transcribe_llm_provider,
                    llm_model=transcribe_llm_model,
                    llm_base_url=transcribe_llm_base_url,
                    llm_api_key=transcribe_llm_api_key
                )
                status_messages.append(message)
                if not success:
                    return "\n".join(status_messages), None

            # Zapis SRT do pliku tymczasowego
            status_messages.append("\n--- Zapis SRT ---")
            temp_srt_path = FILES_DIR / f"auto_{Path(video_path).stem}.srt"
            success, message = write_srt(segments, str(temp_srt_path))
            status_messages.append(message)
            if not success:
                return "\n".join(status_messages), None
            srt_path = str(temp_srt_path)
            temp_files.append(srt_path)

        status_messages.append(f"Liczba segmentów: {len(segments)}")

        # === 3. Tryb lektora (łączenie segmentów) ===
        if narrator_mode and enable_tts_dubbing:
            progress(0.2, desc="Łączenie segmentów (tryb lektora)...")
            status_messages.append("\n=== Tryb lektora ===")
            status_messages.append(f"Łączenie segmentów (max gap: {merge_gap}ms)")

            original_count = len(segments)
            segments = merge_segments_for_narrator(
                segments,
                max_gap_to_merge_ms=int(merge_gap)
            )
            status_messages.append(f"Połączono: {original_count} -> {len(segments)} segmentów")

        # === 4. Ekstrakcja audio z wideo (tylko gdy potrzebne dla TTS/miksowania) ===
        original_audio_path = None
        if enable_tts_dubbing:
            progress(0.25, desc="Ekstrakcja audio z wideo...")
            status_messages.append("\n=== Ekstrakcja audio ===")

            # Extract audio for TTS processing (low quality is sufficient)
            # Same track as the transcription, so the voiceover sits on top of
            # the audio the subtitles were actually made from
            success, message, original_audio_path = extract_audio_from_video(
                video_path,
                output_dir="files",
                high_quality=False,
                audio_track=audio_track,
                preferred_language=_expected_audio_language(
                    transcribe_language, transcribe_source_lang, transcribe_enable_translation
                )
            )
            status_messages.append(message)
            if not success:
                return "\n".join(status_messages), None

            temp_files.append(original_audio_path)
        else:
            status_messages.append("\n=== Pominięto ekstrakcję audio (niepotrzebne bez dubbingu) ===")

        # === 5. Generowanie TTS ===
        tts_audio_path = None
        if enable_tts_dubbing:
            progress(0.3, desc="Generowanie TTS...")
            status_messages.append("\n=== Generowanie TTS ===")
            status_messages.append(f"Silnik: {tts_engine}, Głos: {tts_voice}")

            tts_output_dir = FILES_DIR / "tts_segments"
            tts_output_dir.mkdir(parents=True, exist_ok=True)

            # Determine coqui parameters
            coqui_model = tts_voice if tts_engine == "coqui" else None
            edge_voice = tts_voice if tts_engine == "edge" else None

            success, message, tts_segments = generate_tts_segments(
                segments=segments,
                output_dir=str(tts_output_dir),
                voice=edge_voice or "pl-PL-MarekNeural",
                engine=tts_engine,
                coqui_model=coqui_model,
                target_language="pl"
            )

            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            status_messages.append(f"Wygenerowano {len(tts_segments)} plików TTS")

            # === 6. Przygotowanie timestampów i łączenie segmentów TTS ===
            progress(0.5, desc="Przygotowanie timestampów TTS...")
            status_messages.append("\n=== Przygotowanie timestampów TTS ===")

            # Convert TTS segments to SegmentTiming objects with smart adjustments
            segment_timings = smart_adjust_timestamps(tts_segments, recovery_mode="aggressive")

            total_shift_ms = segment_timings[-1].shift_ms if segment_timings else 0
            if total_shift_ms > 0:
                status_messages.append(f"Łączne przesunięcie: +{total_shift_ms}ms")
                shifted_count = sum(1 for t in segment_timings if t.shift_ms > 0)
                status_messages.append(f"Przesunięte segmenty: {shifted_count}/{len(segment_timings)}")
            else:
                status_messages.append("Brak przesunięć - wszystkie TTS mieszczą się w slotach")

            # Get total duration of original audio
            status_messages.append("\n=== Łączenie audio TTS ===")

            success, total_duration_ms = get_audio_duration_ms(original_audio_path)
            if not success:
                return "\n".join(status_messages) + "\nBłąd: Nie można odczytać długości audio", None

            # Extend total duration by shifts
            adjusted_total_duration_ms = total_duration_ms + total_shift_ms

            # Combine TTS segments into single track
            tts_combined_path = FILES_DIR / f"tts_combined_{Path(video_path).stem}.wav"
            success, message, tts_temp_files = create_tts_audio_track(
                segment_timings,
                adjusted_total_duration_ms,
                str(tts_combined_path)
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            tts_audio_path = str(tts_combined_path)
            temp_files.append(tts_audio_path)

            # Add TTS temp files and directory to cleanup list
            if success:
                temp_files.extend(tts_temp_files)
                temp_files.append(tts_output_dir)  # Will clean up entire directory

        # === 7. Miksowanie audio (jeśli dubbing włączony) ===
        final_audio_path = None

        if enable_tts_dubbing and tts_audio_path:
            progress(0.7, desc="Miksowanie audio...")
            status_messages.append("\n=== Miksowanie audio ===")

            # Mix audio using the already extracted original audio
            mixed_audio_path = FILES_DIR / f"mixed_audio_{Path(video_path).stem}.wav"
            success, message = mix_audio_tracks(
                original_audio_path=original_audio_path,
                tts_audio_path=tts_audio_path,
                output_path=str(mixed_audio_path),
                original_volume=original_volume,
                tts_volume=tts_volume
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

            final_audio_path = str(mixed_audio_path)
            temp_files.append(final_audio_path)
        else:
            # No dubbing, just use original audio
            final_audio_path = original_audio_path

        # === 8. Tworzenie wyjścia ===
        output_file = None

        if dubbing_type == "Tylko audio WAV":
            # Output only audio
            progress(0.9, desc="Zapisywanie audio WAV...")
            status_messages.append("\n=== Zapisywanie audio WAV ===")
            if final_audio_path:
                output_file = final_audio_path
                status_messages.append(f"Plik audio: {output_file}")
            else:
                return "\n".join(status_messages) + "\nBłąd: Brak pliku audio do zapisania.", None

        else:  # Wideo
            progress(0.8, desc="Przygotowywanie wideo...")
            status_messages.append("\n=== Przygotowywanie wideo ===")

            if enable_tts_dubbing and tts_audio_path:
                # TTS dubbing is enabled - create dubbed video with mixed audio
                status_messages.append("Tworzenie wideo z dubbingiem...")
                dubbed_video_path = FILES_DIR / f"dubbed_{Path(video_path).stem}.mp4"
                success, message = create_dubbed_video(
                    original_video_path=video_path,
                    mixed_audio_path=final_audio_path,
                    output_video_path=str(dubbed_video_path)
                )
                status_messages.append(message)

                if not success:
                    return "\n".join(status_messages), None

                output_file = str(dubbed_video_path)
                # temp_files.append(output_file)
            else:
                # No TTS dubbing - use original video directly (no re-encoding)
                status_messages.append("Używanie oryginalnego wideo (bez dubbingu)...")
                output_file = video_path

            # === 9. Wypalanie napisów (opcjonalne) ===
            if burn_subtitles:
                progress(0.95, desc="Wypalanie napisów do wideo...")
                status_messages.append("\n=== Wypalanie napisów ===")

                # Determine subtitle file to use
                subtitle_to_burn = srt_path

                # If bilingual subtitles requested, create ASS file
                if bilingual_subtitles and segments:
                    # For bilingual, we need both original and translated segments
                    # For now, use the same segments for both (would need translation in full pipeline)
                    ass_path = FILES_DIR / f"{Path(video_path).stem}_bilingual.ass"
                    success_ass, message_ass = write_dual_language_ass(
                        original_segments=segments,
                        translated_segments=segments,  # TODO: use actual translation
                        output_path=str(ass_path)
                    )
                    status_messages.append(message_ass)

                    if success_ass:
                        subtitle_to_burn = str(ass_path)

                # Burn subtitles
                burned_video_path = FILES_DIR / f"burned_{Path(video_path).stem}.mp4"
                success, message = burn_subtitles_to_video(
                    video_path=output_file,
                    subtitle_path=subtitle_to_burn,
                    output_path=str(burned_video_path)
                )
                status_messages.append(message)

                if not success:
                    return "\n".join(status_messages), None

                output_file = str(burned_video_path)

        # === 10. Sukces ===
        progress(1.0, desc="Gotowe!")
        status_messages.append("\n=== ✓ Dubbing zakończony pomyślnie ===")
        status_messages.append(f"Plik wyjściowy: {output_file}")

        return "\n".join(status_messages), output_file

    except Exception as e:
        error_msg = f"\n\n!!! BŁĄD !!!\n{str(e)}"
        import traceback
        error_msg += f"\n{traceback.format_exc()}"
        status_messages.append(error_msg)
        return "\n".join(status_messages), None
    finally:
        # Clean up temporary files and directories
        for temp_file in temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path)
                    else:
                        temp_path.unlink()
            except Exception:
                pass
        # Clean up Gradio temp files
        cleanup_gradio_temp()
        # Clear CUDA cache after operation
        clear_cuda_cache()


def handle_download(
    youtube_url: str,
    download_type: str,
    video_quality: str,
    audio_quality: str,
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """
    Handler for download tab.

    Args:
        youtube_url: YouTube URL string
        download_type: "Wideo" or "Tylko audio"
        video_quality: Video quality (e.g., "1080", "720")
        audio_quality: Audio quality (e.g., "best", "320")
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message: str, file_path: Optional[str])
    """
    from ..youtube_processor import download_video, download_audio

    status_messages = []

    try:
        # Clean CUDA cache before starting (though download doesn't use GPU, good practice)
        clear_cuda_cache()
        # === 1. Walidacja URL ===
        progress(0, desc="Walidacja URL...")
        status_messages.append("=== Walidacja URL ===")

        if not youtube_url or not youtube_url.strip():
            return "Błąd: Proszę podać URL YouTube.", None

        if not validate_youtube_url(youtube_url):
            return "Błąd: Nieprawidłowy URL YouTube.", None

        status_messages.append(f"YouTube URL: {youtube_url}")

        # === 2. Pobieranie ===
        if download_type == "Wideo":
            progress(0.1, desc=f"Pobieranie wideo ({video_quality}p)...")
            status_messages.append(f"\n=== Pobieranie wideo (jakość: {video_quality}p) ===")

            success, message, file_path = download_video(
                youtube_url,
                output_dir="files",
                quality=video_quality
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

        else:  # Tylko audio
            progress(0.1, desc="Pobieranie audio...")
            status_messages.append(f"\n=== Pobieranie audio (jakość: {audio_quality}) ===")

            success, message, file_path = download_audio(
                youtube_url,
                output_dir="files"
            )
            status_messages.append(message)

            if not success:
                return "\n".join(status_messages), None

        # === 3. Sukces ===
        progress(1.0, desc="Gotowe!")
        status_messages.append("\n=== ✓ Pobieranie zakończone pomyślnie ===")
        status_messages.append(f"Plik: {file_path}")

        return "\n".join(status_messages), file_path

    except Exception as e:
        error_msg = f"\n\n!!! BŁĄD !!!\n{str(e)}"
        import traceback
        error_msg += f"\n{traceback.format_exc()}"
        status_messages.append(error_msg)
        return "\n".join(status_messages), None
    finally:
        # Clear CUDA cache after operation
        clear_cuda_cache()


def handle_srt_correction(
    srt_file,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    progress=gr.Progress()
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Handler for SRT correction tab.

    Args:
        srt_file: Uploaded SRT file object
        llm_provider: LLM provider (openai, ollama, anthropic)
        llm_model: LLM model name
        llm_base_url: LLM API base URL
        llm_api_key: LLM API key
        progress: Gradio progress tracker

    Returns:
        Tuple (status_text, corrected_srt_path, corrections_log_path)
    """
    from ..srt_reader import parse_srt_file
    from ..srt_writer import write_srt
    from ..srt_corrector import correct_srt_with_llm, write_corrections_log
    from ..llm_translator import get_llm_config

    status_messages = []

    try:
        # === 1. Walidacja pliku SRT ===
        progress(0, desc="Walidacja pliku SRT...")
        status_messages.append("=== Walidacja pliku SRT ===")

        if srt_file is None:
            return "Błąd: Proszę wybrać plik SRT.", None, None

        srt_path = srt_file.name if hasattr(srt_file, 'name') else str(srt_file)
        srt_filename = Path(srt_path).name
        status_messages.append(f"Plik: {srt_filename}")

        # === 2. Parsowanie pliku SRT ===
        progress(0.1, desc="Wczytywanie pliku SRT...")
        status_messages.append("\n=== Wczytywanie pliku SRT ===")

        success, message, segments = parse_srt_file(srt_path)
        status_messages.append(message)

        if not success:
            return "\n".join(status_messages), None, None

        status_messages.append(f"Liczba segmentów: {len(segments)}")

        # === 3. Korekta przez LLM ===
        progress(0.2, desc="Korekta tekstu przez LLM...")
        status_messages.append("\n=== Korekta przez LLM ===")

        config = get_llm_config(llm_provider, llm_model, llm_base_url, llm_api_key)
        status_messages.append(f"Provider: {config['provider']} / {config['model']}")

        success, message, corrected_segments, changes = correct_srt_with_llm(
            segments=segments,
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key
        )

        status_messages.append(message)

        if not success:
            return "\n".join(status_messages), None, None

        # === 4. Zapis poprawionego pliku SRT ===
        progress(0.8, desc="Zapisywanie plików...")
        status_messages.append("\n=== Zapisywanie plików ===")

        # Generate output filenames
        srt_stem = Path(srt_path).stem
        corrected_srt_path = FILES_DIR / f"{srt_stem}_corrected.srt"
        corrections_log_path = FILES_DIR / f"{srt_stem}_corrections.txt"

        # Write corrected SRT
        success, message = write_srt(corrected_segments, str(corrected_srt_path))
        status_messages.append(message)

        if not success:
            return "\n".join(status_messages), None, None

        # Write corrections log
        success, message = write_corrections_log(
            changes=changes,
            source_filename=srt_filename,
            output_path=str(corrections_log_path),
            provider=config['provider'],
            model=config['model']
        )
        status_messages.append(message)

        # === 5. Sukces ===
        progress(1.0, desc="Gotowe!")
        status_messages.append("\n=== ✓ Korekta zakończona pomyślnie ===")
        status_messages.append(f"Liczba zmian: {len(changes)}")
        status_messages.append(f"Poprawiony SRT: {corrected_srt_path}")
        status_messages.append(f"Log zmian: {corrections_log_path}")

        return "\n".join(status_messages), str(corrected_srt_path), str(corrections_log_path)

    except Exception as e:
        error_msg = f"\n\n!!! BŁĄD !!!\n{str(e)}"
        import traceback
        error_msg += f"\n{traceback.format_exc()}"
        status_messages.append(error_msg)
        return "\n".join(status_messages), None, None
    finally:
        # Clean up Gradio temp files
        cleanup_gradio_temp()
