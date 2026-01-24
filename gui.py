#!/usr/bin/env python3
"""
GUI Application - Main Entry Point
Gradio interface for the Transcription App
"""

import gradio as gr
from data.gui import config
from data.gui import handlers
from data.validators import validate_youtube_url_with_message, validate_file_extension, validate_srt_file


# === Validation helpers for real-time feedback ===

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

                    hf_token = gr.Textbox(
                        label="HuggingFace Token (wymagany dla diarization)",
                        type="password",
                        placeholder="hf_..."
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
                    return gr.update(visible=enabled)

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

                enable_translation.change(
                    fn=toggle_translation,
                    inputs=[enable_translation],
                    outputs=[translation_row]
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
                        hf_token
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
                    return gr.update(visible=checkbox_value)  # pokazuje/ukrywa dubbing_srt_file

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

                # Event handlers
                dubbing_source_type.change(
                    fn=toggle_dubbing_source,
                    inputs=[dubbing_source_type],
                    outputs=[dubbing_youtube_url, dubbing_video_file]
                )

                dubbing_use_srt.change(
                    fn=toggle_dubbing_srt,
                    inputs=[dubbing_use_srt],
                    outputs=[dubbing_srt_file]
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
                        hf_token
                    ],
                    outputs=[dubbing_status, dubbing_output]
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
