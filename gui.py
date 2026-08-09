#!/usr/bin/env python3
"""
GUI Application - Main Entry Point
Gradio interface for the Transcription App
"""

# ===== WARNING SUPPRESSION =====
# Musi być przed innymi importami, aby skutecznie tłumić ostrzeżenia
import os
from data.warning_suppressor import suppress_third_party_warnings
suppress_third_party_warnings(debug_mode=False)

from dotenv import load_dotenv
load_dotenv()

import atexit
import gradio as gr
from data.gui import config
from data.gui import handlers
from data.validators import validate_youtube_url_with_message, validate_file_extension, validate_srt_file
from data.audio_processor import list_audio_streams, format_audio_stream


# === Validation helpers for real-time feedback ===

def refresh_audio_tracks(file_obj) -> gr.Dropdown:
    """
    Populate the audio-track picker from an uploaded file.

    Only shown for multi-track files - movie rips often bundle the original
    audio with dubbed/voiceover versions, and transcribing the wrong one
    produces subtitles that neither match the dialogue nor its timing.
    """
    hidden = gr.update(choices=[("Automatycznie", "auto")], value="auto", visible=False)

    if file_obj is None:
        return hidden

    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    streams = list_audio_streams(file_path)

    if len(streams) <= 1:
        return hidden

    choices = [("Automatycznie (oryginał / zgodna z językiem źródłowym)", "auto")]
    choices += [(format_audio_stream(s), str(s["track"])) for s in streams]

    return gr.update(
        choices=choices,
        value="auto",
        visible=True,
        label=f"Ścieżka audio — plik ma {len(streams)} ścieżek (uwaga na dubbing/lektora)"
    )

def validate_youtube_url_realtime(url: str) -> gr.HTML:
    """Validate YouTube URL in real-time and return status HTML."""
    if not url or not url.strip():
        return gr.HTML(value="")
    is_valid, message = validate_youtube_url_with_message(url)
    color = "#22c55e" if is_valid else "#ef4444"  # green or red
    return gr.HTML(value=f'<div style="color: {color}; font-size: 0.9em;">{message}</div>')


def validate_file_upload(file_obj) -> gr.HTML:
    """Validate uploaded file extension and return status HTML."""
    if file_obj is None:
        return gr.HTML(value="")
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    is_valid, message = validate_file_extension(file_path, {'.mp4', '.mkv', '.avi', '.mov', '.mp3', '.wav'})
    color = "#22c55e" if is_valid else "#ef4444"
    return gr.HTML(value=f'<div style="color: {color}; font-size: 0.9em;">{message}</div>')


def validate_srt_upload(file_obj) -> gr.HTML:
    """Validate uploaded SRT file and return status HTML."""
    if file_obj is None:
        return gr.HTML(value="")
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    is_valid, message = validate_srt_file(file_path)
    color = "#22c55e" if is_valid else "#ef4444"
    return gr.HTML(value=f'<div style="color: {color}; font-size: 0.9em;">{message}</div>')


def create_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Aplikacja Transkrypcyjna") as app:
        gr.Markdown("# Aplikacja Transkrypcyjna")

        # Placeholder for tabs (will be implemented in later stages)
        with gr.Tabs():
            with gr.Tab("Transkrypcja"):
                # 2.1 Komponenty źródła
                source_type = gr.Radio(
                    choices=["YouTube URL", "Plik lokalny"],
                    value="YouTube URL",
                    label="Źródło"
                )

                youtube_url = gr.Textbox(
                    label="URL YouTube",
                    placeholder="https://www.youtube.com/watch?v=...",
                    visible=True
                )

                youtube_url_status = gr.HTML(value="")

                local_file = gr.File(
                    label="Plik lokalny (mp4, mkv, avi, mov)",
                    file_types=[".mp4", ".mkv", ".avi", ".mov"],
                    visible=False
                )

                local_file_status = gr.HTML(value="")

                audio_track = gr.Dropdown(
                    choices=[("Automatycznie", "auto")],
                    value="auto",
                    label="Ścieżka audio",
                    visible=False
                )

                # 2.2 Podstawowe ustawienia
                gr.Markdown("### Ustawienia transkrypcji")

                with gr.Row():
                    model = gr.Dropdown(
                        choices=config.MODELS,
                        value=config.DEFAULT_MODEL,
                        label="Model Whisper"
                    )

                    language = gr.Dropdown(
                        choices=[(v, k) for k, v in config.LANGUAGES.items()],
                        value=config.DEFAULT_LANGUAGE,
                        label="Język"
                    )

                    engine = gr.Dropdown(
                        choices=config.ENGINES,
                        value=config.DEFAULT_ENGINE,
                        label="Silnik"
                    )

                # 2.3 Sekcja tłumaczenia
                gr.Markdown("### Tłumaczenie")

                enable_translation = gr.Checkbox(
                    label="Włącz tłumaczenie",
                    value=False
                )

                with gr.Row(visible=False) as translation_row:
                    source_lang = gr.Dropdown(
                        choices=[(v, k) for k, v in config.LANGUAGES.items()],
                        value="auto",
                        label="Język źródłowy"
                    )

                    # Język docelowy bez opcji "auto"
                    target_languages = {k: v for k, v in config.LANGUAGES.items() if k != "auto"}
                    target_lang = gr.Dropdown(
                        choices=[(v, k) for k, v in target_languages.items()],
                        value="pl",
                        label="Język docelowy"
                    )

                # LLM Translation Options
                with gr.Accordion("Tłumaczenie LLM (kontekstowe)", open=False, visible=False) as llm_accordion:
                    gr.Markdown("*Tłumaczenie przez LLM z uwzględnieniem kontekstu i płci mówców*")

                    use_llm_translate = gr.Checkbox(
                        label="Użyj LLM zamiast Google Translator",
                        value=False
                    )

                    detect_gender = gr.Checkbox(
                        label="Wykryj płeć mówców (wymaga diaryzacji WhisperX)",
                        value=False
                    )

                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            choices=config.LLM_PROVIDERS,
                            value=config.DEFAULT_LLM_PROVIDER,
                            label="Provider LLM"
                        )

                        llm_model = gr.Textbox(
                            label="Model LLM",
                            value=config.DEFAULT_LLM_MODEL,
                            placeholder="np. gpt-4o-mini, llama3.1, GLM-4.7"
                        )

                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL (opcjonalnie)",
                            placeholder="np. https://api.z.ai/api/anthropic"
                        )

                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="lub ustaw LLM_API_KEY w środowisku"
                        )

                # 2.4 Zaawansowane ustawienia
                with gr.Accordion("Zaawansowane ustawienia", open=False):
                    with gr.Row():
                        device = gr.Dropdown(
                            choices=config.DEVICES,
                            value="auto",
                            label="Urządzenie"
                        )

                        timeout = gr.Number(
                            value=config.DEFAULT_TIMEOUT,
                            label="Timeout (sekundy)",
                            precision=0
                        )

                    whisperx_align = gr.Checkbox(
                        label="WhisperX: Word-level alignment",
                        value=False
                    )

                    whisperx_diarize = gr.Checkbox(
                        label="WhisperX: Speaker diarization",
                        value=False
                    )

                # Przycisk transkrypcji
                transcribe_btn = gr.Button("Transkrybuj", variant="primary")

                # Output
                transcription_status = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

                transcription_output = gr.File(
                    label="Pobierz plik SRT"
                )

                # Logika ukrywania/pokazywania pól
                def toggle_source(choice):
                    if choice == "YouTube URL":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                def toggle_translation(enabled):
                    return gr.update(visible=enabled), gr.update(visible=enabled)

                source_type.change(
                    fn=toggle_source,
                    inputs=[source_type],
                    outputs=[youtube_url, local_file]
                )

                # Real-time validation with debounce
                youtube_url.change(
                    fn=validate_youtube_url_realtime,
                    inputs=[youtube_url],
                    outputs=[youtube_url_status],
                    show_progress="hidden"
                )

                local_file.change(
                    fn=validate_file_upload,
                    inputs=[local_file],
                    outputs=[local_file_status],
                    show_progress="hidden"
                )

                local_file.change(
                    fn=refresh_audio_tracks,
                    inputs=[local_file],
                    outputs=[audio_track],
                    show_progress="hidden"
                )

                enable_translation.change(
                    fn=toggle_translation,
                    inputs=[enable_translation],
                    outputs=[translation_row, llm_accordion]
                )

                # Handler dla przycisku transkrypcji
                transcribe_btn.click(
                    fn=handlers.handle_transcription,
                    inputs=[
                        source_type,
                        youtube_url,
                        local_file,
                        model,
                        language,
                        engine,
                        enable_translation,
                        source_lang,
                        target_lang,
                        device,
                        timeout,
                        whisperx_align,
                        whisperx_diarize,
                        use_llm_translate,
                        llm_provider,
                        llm_model,
                        llm_base_url,
                        llm_api_key,
                        detect_gender,
                        audio_track
                    ],
                    outputs=[transcription_status, transcription_output]
                )

            with gr.Tab("Dubbing / Napisy"):
                # 3.1 Komponenty źródła
                dubbing_source_type = gr.Radio(
                    choices=["YouTube URL", "Plik lokalny"],
                    value="YouTube URL",
                    label="Źródło wideo"
                )

                dubbing_youtube_url = gr.Textbox(
                    label="URL YouTube",
                    placeholder="https://www.youtube.com/watch?v=...",
                    visible=True
                )

                dubbing_youtube_url_status = gr.HTML(value="")

                dubbing_video_file = gr.File(
                    label="Plik wideo/audio (mp4, mkv, avi, mov, mp3, wav)",
                    file_types=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav"],
                    visible=False
                )

                dubbing_video_file_status = gr.HTML(value="")

                dubbing_audio_track = gr.Dropdown(
                    choices=[("Automatycznie", "auto")],
                    value="auto",
                    label="Ścieżka audio",
                    visible=False
                )

                dubbing_use_srt = gr.Checkbox(
                    label="Użyj pliku SRT (gdy odznaczone - auto-transkrypcja)",
                    value=False
                )

                dubbing_srt_file = gr.File(
                    label="Plik SRT",
                    file_types=[".srt"],
                    visible=False
                )

                dubbing_srt_file_status = gr.HTML(value="")

                dubbing_correct_srt = gr.Checkbox(
                    label="Korekta SRT przez LLM (przed dubbingiem)",
                    value=False,
                    visible=False
                )

                # Info o opcjach transkrypcji (gdy nie używamy SRT)
                transcription_options_info = gr.HTML(
                    value='<div style="padding: 10px; background: #1a1a2e; border-radius: 5px; margin: 10px 0;">'
                          '<b>Opcje auto-transkrypcji:</b> WhisperX / base | Tłumaczenie: wyłączone</div>',
                    visible=True
                )

                # 3.2 Podstawowe opcje
                gr.Markdown("### Opcje dubbingu")

                with gr.Row():
                    enable_tts_dubbing = gr.Checkbox(
                        label="Dubbing TTS",
                        value=True
                    )

                    burn_subtitles = gr.Checkbox(
                        label="Wpal napisy do wideo",
                        value=False
                    )

                with gr.Row():
                    dubbing_type = gr.Radio(
                        choices=["Wideo", "Tylko audio WAV"],
                        value="Wideo",
                        label="Typ dubbingu"
                    )

                    bilingual_subtitles = gr.Checkbox(
                        label="Napisy dwujęzyczne",
                        value=False
                    )

                # 3.3 Ustawienia TTS
                with gr.Group(visible=True) as tts_settings_group:
                    gr.Markdown("### Ustawienia TTS")

                    with gr.Row():
                        tts_engine = gr.Dropdown(
                            choices=config.TTS_ENGINES,
                            value=config.DEFAULT_TTS_ENGINE,
                            label="Silnik TTS"
                        )

                        # Dropdown dla głosów - będzie dynamicznie aktualizowany
                        tts_voice = gr.Dropdown(
                            choices=[(v, k) for k, v in config.VOICES_EDGE.items()],
                            value=config.DEFAULT_VOICE_EDGE,
                            label="Głos"
                        )

                    with gr.Row():
                        tts_volume = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=config.DEFAULT_VOLUME_TTS,
                            step=0.1,
                            label="Głośność TTS"
                        )

                        original_volume = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=config.DEFAULT_VOLUME_ORIGINAL,
                            step=0.05,
                            label="Głośność oryginału"
                        )

                # 3.4 Tryb lektora
                gr.Markdown("### Tryb lektora")

                narrator_mode = gr.Checkbox(
                    label="Tryb lektora (łącz segmenty)",
                    value=False
                )

                merge_gap = gr.Number(
                    value=config.DEFAULT_MERGE_GAP,
                    label="Merge gap (ms)",
                    precision=0,
                    visible=False
                )

                # 3.5 Zaawansowane ustawienia
                with gr.Accordion("Zaawansowane ustawienia", open=False):
                    with gr.Row():
                        max_segment_length = gr.Number(
                            value=config.DEFAULT_MAX_SEGMENT_LENGTH,
                            label="Maks. długość segmentu (s)",
                            precision=0
                        )

                        max_words_segment = gr.Number(
                            value=config.DEFAULT_MAX_WORDS,
                            label="Maks. słów w segmencie",
                            precision=0
                        )

                    with gr.Row():
                        fill_gaps = gr.Checkbox(
                            label="Wypełnij luki",
                            value=False
                        )

                        min_pause = gr.Number(
                            value=config.DEFAULT_MIN_PAUSE,
                            label="Min. pauza (ms)",
                            precision=0
                        )

                    with gr.Row():
                        max_gap = gr.Number(
                            value=config.DEFAULT_MAX_GAP,
                            label="Maks. luka (ms)",
                            precision=0
                        )

                        video_quality = gr.Dropdown(
                            choices=config.VIDEO_QUALITIES,
                            value=config.DEFAULT_VIDEO_QUALITY,
                            label="Jakość wideo"
                        )

                # Przycisk dubbingu
                dubbing_btn = gr.Button("Generuj dubbing", variant="primary")

                # Output
                dubbing_status = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

                dubbing_output = gr.File(
                    label="Pobierz plik wyjściowy"
                )

                # Logika ukrywania/pokazywania pól
                def toggle_dubbing_source(choice):
                    if choice == "YouTube URL":
                        return (
                            gr.update(visible=True),   # youtube_url
                            gr.update(visible=False)   # video_file
                        )
                    else:  # Plik lokalny
                        return (
                            gr.update(visible=False),  # youtube_url
                            gr.update(visible=True)    # video_file
                        )

                def toggle_dubbing_srt(checkbox_value):
                    # Pokazuje/ukrywa dubbing_srt_file i dubbing_correct_srt
                    return (
                        gr.update(visible=checkbox_value),  # dubbing_srt_file
                        gr.update(visible=checkbox_value)   # dubbing_correct_srt
                    )

                def toggle_tts_settings(enabled):
                    return gr.update(visible=enabled)

                def toggle_narrator_merge_gap(enabled):
                    return gr.update(visible=enabled)

                def update_voice_dropdown(engine):
                    """Update voice dropdown based on selected TTS engine"""
                    if engine == "edge":
                        choices = [(v, k) for k, v in config.VOICES_EDGE.items()]
                        default = config.DEFAULT_VOICE_EDGE
                    else:  # coqui
                        choices = [(v, k) for k, v in config.COQUI_MODELS.items()]
                        default = config.DEFAULT_COQUI_MODEL
                    return gr.update(choices=choices, value=default)

                def update_transcription_options_info(
                    use_srt, eng, mod, trans_enabled, use_llm, provider, llm_mod, gender_detect
                ):
                    """Update transcription options info panel for dubbing tab."""
                    if use_srt:
                        return gr.HTML(visible=False)

                    parts = [f"<b>Opcje auto-transkrypcji:</b> {eng} / {mod}"]

                    if trans_enabled:
                        if use_llm:
                            llm_info = f"LLM ({provider}"
                            if llm_mod:
                                llm_info += f" / {llm_mod}"
                            llm_info += ")"
                            if gender_detect:
                                llm_info += " + wykrywanie płci"
                            parts.append(f"Tłumaczenie: {llm_info}")
                        else:
                            parts.append("Tłumaczenie: Google Translator")
                    else:
                        parts.append("Tłumaczenie: wyłączone")

                    html = '<div style="padding: 10px; background: #1a1a2e; border-radius: 5px; margin: 10px 0;">' + \
                           " | ".join(parts) + '</div>'
                    return gr.HTML(value=html, visible=True)

                # Event handlers
                dubbing_source_type.change(
                    fn=toggle_dubbing_source,
                    inputs=[dubbing_source_type],
                    outputs=[dubbing_youtube_url, dubbing_video_file]
                )

                dubbing_use_srt.change(
                    fn=toggle_dubbing_srt,
                    inputs=[dubbing_use_srt],
                    outputs=[dubbing_srt_file, dubbing_correct_srt]
                )

                # Real-time validation with debounce
                dubbing_youtube_url.change(
                    fn=validate_youtube_url_realtime,
                    inputs=[dubbing_youtube_url],
                    outputs=[dubbing_youtube_url_status],
                    show_progress="hidden"
                )

                dubbing_video_file.change(
                    fn=validate_file_upload,
                    inputs=[dubbing_video_file],
                    outputs=[dubbing_video_file_status],
                    show_progress="hidden"
                )

                dubbing_video_file.change(
                    fn=refresh_audio_tracks,
                    inputs=[dubbing_video_file],
                    outputs=[dubbing_audio_track],
                    show_progress="hidden"
                )

                dubbing_srt_file.change(
                    fn=validate_srt_upload,
                    inputs=[dubbing_srt_file],
                    outputs=[dubbing_srt_file_status],
                    show_progress="hidden"
                )

                enable_tts_dubbing.change(
                    fn=toggle_tts_settings,
                    inputs=[enable_tts_dubbing],
                    outputs=[tts_settings_group]
                )

                narrator_mode.change(
                    fn=toggle_narrator_merge_gap,
                    inputs=[narrator_mode],
                    outputs=[merge_gap]
                )

                tts_engine.change(
                    fn=update_voice_dropdown,
                    inputs=[tts_engine],
                    outputs=[tts_voice]
                )

                # Update transcription options info when relevant settings change
                transcription_info_inputs = [
                    dubbing_use_srt, engine, model, enable_translation,
                    use_llm_translate, llm_provider, llm_model, detect_gender
                ]
                for component in transcription_info_inputs:
                    component.change(
                        fn=update_transcription_options_info,
                        inputs=transcription_info_inputs,
                        outputs=[transcription_options_info],
                        show_progress="hidden"
                    )

                # Handler dla przycisku dubbingu
                dubbing_btn.click(
                    fn=handlers.handle_dubbing,
                    inputs=[
                        dubbing_source_type,
                        dubbing_youtube_url,
                        dubbing_video_file,
                        dubbing_use_srt,
                        dubbing_srt_file,
                        enable_tts_dubbing,
                        burn_subtitles,
                        dubbing_type,
                        bilingual_subtitles,
                        tts_engine,
                        tts_voice,
                        tts_volume,
                        original_volume,
                        narrator_mode,
                        merge_gap,
                        max_segment_length,
                        max_words_segment,
                        fill_gaps,
                        min_pause,
                        max_gap,
                        video_quality,
                        # Parametry transkrypcji z zakładki Transkrypcja
                        model,
                        language,
                        engine,
                        enable_translation,
                        source_lang,
                        target_lang,
                        device,
                        timeout,
                        whisperx_align,
                        whisperx_diarize,
                        # Parametry LLM z zakładki Transkrypcja
                        use_llm_translate,
                        llm_provider,
                        llm_model,
                        llm_base_url,
                        llm_api_key,
                        detect_gender,
                        # Korekta SRT
                        dubbing_correct_srt,
                        # Wybór ścieżki audio (pliki wielościeżkowe)
                        dubbing_audio_track
                    ],
                    outputs=[dubbing_status, dubbing_output]
                )

            with gr.Tab("Korekta SRT"):
                # Zakładka standalone korekty SRT przez LLM
                gr.Markdown("### Korekta literówek i błędów w plikach SRT")
                gr.Markdown("*Poprawia błędy OCR, literówki i błędy gramatyczne przy użyciu LLM*")

                # Upload pliku SRT
                correction_srt_file = gr.File(
                    label="Plik SRT do korekty",
                    file_types=[".srt"]
                )

                correction_srt_status = gr.HTML(value="")

                # Ustawienia LLM
                gr.Markdown("### Ustawienia LLM")

                with gr.Row():
                    correction_llm_provider = gr.Dropdown(
                        choices=config.LLM_PROVIDERS,
                        value=config.DEFAULT_LLM_PROVIDER,
                        label="Provider LLM"
                    )

                    correction_llm_model = gr.Textbox(
                        label="Model LLM",
                        value=config.DEFAULT_LLM_MODEL,
                        placeholder="np. gpt-4o-mini, llama3.1"
                    )

                with gr.Row():
                    correction_llm_base_url = gr.Textbox(
                        label="Base URL (opcjonalnie)",
                        placeholder="np. https://api.z.ai/api/anthropic"
                    )

                    correction_llm_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="lub ustaw LLM_API_KEY w środowisku"
                    )

                # Przycisk korekty
                correction_btn = gr.Button("Popraw napisy", variant="primary")

                # Output
                correction_status = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

                with gr.Row():
                    correction_output_srt = gr.File(
                        label="Pobierz poprawiony SRT"
                    )

                    correction_output_log = gr.File(
                        label="Pobierz log zmian"
                    )

                # Walidacja pliku SRT
                correction_srt_file.change(
                    fn=validate_srt_upload,
                    inputs=[correction_srt_file],
                    outputs=[correction_srt_status],
                    show_progress="hidden"
                )

                # Handler dla przycisku korekty
                correction_btn.click(
                    fn=handlers.handle_srt_correction,
                    inputs=[
                        correction_srt_file,
                        correction_llm_provider,
                        correction_llm_model,
                        correction_llm_base_url,
                        correction_llm_api_key
                    ],
                    outputs=[correction_status, correction_output_srt, correction_output_log]
                )

            with gr.Tab("Pobieranie"):
                # 4.1 Komponenty
                gr.Markdown("### Pobieranie z YouTube")

                download_youtube_url = gr.Textbox(
                    label="URL YouTube",
                    placeholder="https://www.youtube.com/watch?v=...",
                )

                download_youtube_url_status = gr.HTML(value="")

                download_type = gr.Radio(
                    choices=["Wideo", "Tylko audio"],
                    value="Wideo",
                    label="Co pobrać"
                )

                # Dropdowns for quality - dynamically visible based on download type
                download_video_quality = gr.Dropdown(
                    choices=config.VIDEO_QUALITIES,
                    value=config.DEFAULT_VIDEO_QUALITY,
                    label="Jakość wideo",
                    visible=True
                )

                download_audio_quality = gr.Dropdown(
                    choices=config.AUDIO_QUALITIES,
                    value="best",
                    label="Jakość audio",
                    visible=False
                )

                # Download button
                download_btn = gr.Button("Pobierz", variant="primary")

                # Output
                download_status = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

                download_output = gr.File(
                    label="Pobierz plik"
                )

                # Logic for showing/hiding quality dropdowns
                def toggle_download_quality(choice):
                    if choice == "Wideo":
                        return (
                            gr.update(visible=True),   # video_quality
                            gr.update(visible=False)   # audio_quality
                        )
                    else:  # Tylko audio
                        return (
                            gr.update(visible=False),
                            gr.update(visible=True)
                        )

                # Event handlers
                download_type.change(
                    fn=toggle_download_quality,
                    inputs=[download_type],
                    outputs=[download_video_quality, download_audio_quality]
                )

                # Real-time validation with debounce
                download_youtube_url.change(
                    fn=validate_youtube_url_realtime,
                    inputs=[download_youtube_url],
                    outputs=[download_youtube_url_status],
                    show_progress="hidden"
                )

                # Handler for download button
                download_btn.click(
                    fn=handlers.handle_download,
                    inputs=[
                        download_youtube_url,
                        download_type,
                        download_video_quality,
                        download_audio_quality
                    ],
                    outputs=[download_status, download_output]
                )

    return app


if __name__ == "__main__":
    atexit.register(handlers.cleanup_gradio_temp)
    print("Uruchamianie GUI...")
    print(f"Zaladowane modele: {', '.join(config.MODELS)}")
    print(f"Dostepne silniki: {', '.join(config.ENGINES)}")
    print(f"Dostepne jezyki: {len(config.LANGUAGES)}")

    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
