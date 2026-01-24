#!/usr/bin/env python3
"""
GUI Application - Main Entry Point
Gradio interface for the Transcription App
"""

import gradio as gr
from data.gui import config

def create_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Aplikacja Transkrypcyjna") as app:
        gr.Markdown("# Aplikacja Transkrypcyjna")
        gr.Markdown("### Etap 1: Struktura bazowa - zaimplementowana ✓")

        # Placeholder for tabs (will be implemented in later stages)
        with gr.Tabs():
            with gr.Tab("Transkrypcja"):
                gr.Markdown("Zakladka transkrypcji - do implementacji w Etapie 2")

            with gr.Tab("Dubbing / Napisy"):
                gr.Markdown("Zakladka dubbingu/napisow - do implementacji w Etapie 3")

            with gr.Tab("Pobieranie"):
                gr.Markdown("Zakladka pobierania - do implementacji w Etapie 4")

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
