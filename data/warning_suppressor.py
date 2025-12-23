#!/usr/bin/env python3
"""
Warning Suppressor Module

Tłumienie ostrzeżeń z bibliotek zewnętrznych (yt-dlp, whisper, TTS, torch, tensorflow).
Moduł ten centralnie zarządza konfiguracją tłumienia ostrzeżeń, aby zapewnić
czyste wyjście dla użytkownika końcowego.

Ostrzeżenia są domyślnie tłumione, ale mogą być ponownie włączone w trybie --debug.
"""

import os
import warnings
import logging


def suppress_third_party_warnings(debug_mode: bool = False):
    """
    Tłumi ostrzeżenia z bibliotek zewnętrznych, chyba że włączony tryb debug.

    Ta funkcja konfiguruje:
    - Python warnings filter (DeprecationWarning, FutureWarning, UserWarning)
    - Zmienne środowiskowe dla TensorFlow, HuggingFace, NumPy
    - Loggery dla bibliotek: yt-dlp, whisper, faster-whisper, whisperx,
      torch, tensorflow, transformers, TTS, numpy, scipy, numba, pyannote

    Args:
        debug_mode: Jeśli True, ostrzeżenia NIE będą tłumione (dla debugowania)

    Przykład:
        >>> import sys
        >>> debug = '--debug' in sys.argv
        >>> suppress_third_party_warnings(debug_mode=debug)
    """

    if debug_mode:
        # W trybie debug pozwól na wszystkie ostrzeżenia
        return

    # ===== PYTHON WARNINGS MODULE =====
    # Tłumi wbudowane ostrzeżenia Pythona
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # ===== ZMIENNE ŚRODOWISKOWE =====
    # Uwaga: Te zmienne są ustawiane tutaj, ale idealne jest ustawienie ich
    # PRZED importem tego modułu (w transcribe.py)

    # TensorFlow: Tłumi INFO i WARNING, pokazuje tylko ERROR
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

    # HuggingFace Tokenizers: Wyłącz ostrzeżenia o równoległości
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # NumPy/ogólne ostrzeżenia Pythona
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # ===== LOGGERY BIBLIOTEK ZEWNĘTRZNYCH =====
    # Ustawienie poziomu ERROR dla wszystkich loggerów bibliotek zewnętrznych

    # yt-dlp (YouTube downloader)
    logging.getLogger('yt_dlp').setLevel(logging.ERROR)

    # Whisper engines (transkrypcja audio)
    logging.getLogger('whisper').setLevel(logging.ERROR)
    logging.getLogger('faster_whisper').setLevel(logging.ERROR)
    logging.getLogger('whisperx').setLevel(logging.ERROR)

    # PyTorch (deep learning framework)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('torch.nn').setLevel(logging.ERROR)
    logging.getLogger('torch.cuda').setLevel(logging.ERROR)

    # TensorFlow (deep learning framework)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorboard').setLevel(logging.ERROR)

    # Transformers (HuggingFace)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

    # Coqui TTS (text-to-speech)
    logging.getLogger('TTS').setLevel(logging.ERROR)
    logging.getLogger('trainer').setLevel(logging.ERROR)

    # NumPy/SciPy (scientific computing)
    logging.getLogger('numpy').setLevel(logging.ERROR)
    logging.getLogger('scipy').setLevel(logging.ERROR)

    # Numba (JIT compiler, używany przez WhisperX)
    logging.getLogger('numba').setLevel(logging.ERROR)

    # PyAnnote (speaker diarization dla WhisperX)
    logging.getLogger('pyannote').setLevel(logging.ERROR)
    logging.getLogger('pyannote.audio').setLevel(logging.ERROR)

    # ABSL (TensorFlow dependency)
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        # ABSL może nie być zainstalowany, nie ma problemu
        pass
