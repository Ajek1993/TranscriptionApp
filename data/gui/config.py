#!/usr/bin/env python3
"""
GUI Configuration Module
Contains all configuration constants for the Gradio interface.
"""

# Lista modeli Whisper
MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3"
]

# Silniki transkrypcji
ENGINES = [
    "whisper",
    "whisperx"
]

# Silniki TTS
TTS_ENGINES = [
    "edge",
    "coqui"
]

# Slownik jezykow (ISO 639-1)
LANGUAGES = {
    "auto": "Automatyczny",
    "pl": "Polski",
    "en": "Angielski",
    "es": "Hiszpanski",
    "fr": "Francuski",
    "de": "Niemiecki",
    "it": "Wloski",
    "pt": "Portugalski",
    "ru": "Rosyjski",
    "tr": "Turecki",
    "nl": "Holenderski",
    "cs": "Czeski",
    "ar": "Arabski",
    "zh-cn": "Chinski (uproszczony)",
    "ja": "Japonski",
    "hu": "Wegierski",
    "ko": "Koreanski"
}

# Glosy Edge TTS (Microsoft Azure)
VOICES_EDGE = {
    # Polski
    "pl-PL-MarekNeural": "Polski - Marek (mezczyzna)",
    "pl-PL-ZofiaNeural": "Polski - Zofia (kobieta)",

    # Angielski
    "en-US-GuyNeural": "Angielski (US) - Guy (mezczyzna)",
    "en-US-JennyNeural": "Angielski (US) - Jenny (kobieta)",
    "en-GB-RyanNeural": "Angielski (UK) - Ryan (mezczyzna)",
    "en-GB-SoniaNeural": "Angielski (UK) - Sonia (kobieta)",

    # Niemiecki
    "de-DE-ConradNeural": "Niemiecki - Conrad (mezczyzna)",
    "de-DE-KatjaNeural": "Niemiecki - Katja (kobieta)",

    # Francuski
    "fr-FR-HenriNeural": "Francuski - Henri (mezczyzna)",
    "fr-FR-DeniseNeural": "Francuski - Denise (kobieta)",

    # Hiszpanski
    "es-ES-AlvaroNeural": "Hiszpanski - Alvaro (mezczyzna)",
    "es-ES-ElviraNeural": "Hiszpanski - Elvira (kobieta)",

    # Wloski
    "it-IT-DiegoNeural": "Wloski - Diego (mezczyzna)",
    "it-IT-ElsaNeural": "Wloski - Elsa (kobieta)"
}

# Modele Coqui TTS
COQUI_MODELS = {
    "tts_models/pl/mai_female/vits": "Polski - Mai (kobieta)",
    "tts_models/multilingual/multi-dataset/xtts_v2": "XTTS v2 (wielojezyczny, wymaga reference audio)"
}

# Urzadzenia
DEVICES = [
    "auto",
    "cuda",
    "cpu"
]

# Jakosci wideo
VIDEO_QUALITIES = [
    "2160",  # 4K
    "1440",  # 2K
    "1080",  # Full HD
    "720",   # HD
    "480",   # SD
    "360"    # Low
]

# Jakosci audio
AUDIO_QUALITIES = [
    "best",
    "320",
    "256",
    "192",
    "128"
]

# Wartosci domyslne
DEFAULT_VOLUME_TTS = 1.0  # Glosnosc TTS (0.0 - 2.0)
DEFAULT_VOLUME_ORIGINAL = 0.3  # Glosnosc oryginalu (0.0 - 1.0)
DEFAULT_TIMEOUT = 1800  # Timeout w sekundach (30 minut)
DEFAULT_MERGE_GAP = 1000  # Merge gap w ms (tryb lektora)
DEFAULT_MAX_SEGMENT_LENGTH = 10  # Maksymalna dlugosc segmentu w sekundach
DEFAULT_MAX_WORDS = 20  # Maksymalna liczba slow w segmencie
DEFAULT_MIN_PAUSE = 150  # Minimalna pauza w ms
DEFAULT_MAX_GAP = 3000  # Maksymalna luka w ms
DEFAULT_VIDEO_QUALITY = "1080"  # Domyslna jakosc wideo
DEFAULT_MODEL = "base"  # Domyslny model Whisper
DEFAULT_ENGINE = "whisper"  # Domyslny silnik transkrypcji
DEFAULT_TTS_ENGINE = "edge"  # Domyslny silnik TTS
DEFAULT_LANGUAGE = "pl"  # Domyslny jezyk
DEFAULT_VOICE_EDGE = "pl-PL-MarekNeural"  # Domyslny glos Edge TTS
DEFAULT_COQUI_MODEL = "tts_models/pl/mai_female/vits"  # Domyslny model Coqui
