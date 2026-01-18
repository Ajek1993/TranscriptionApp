#!/usr/bin/env python3
"""
Translation Module
Handles translation of text segments using Google Translator.
"""

from typing import List, Tuple
from tqdm import tqdm

# Import TRANSLATOR_AVAILABLE flag from validators
from .validators import TRANSLATOR_AVAILABLE

# Conditional import for GoogleTranslator
try:
    from deep_translator import GoogleTranslator
except ImportError:
    pass


def translate_segments(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    batch_size: int = 50
) -> Tuple[bool, str, List[Tuple[int, int, str]], str]:
    """
    Translate text content of segments while preserving timestamps.

    Supports automatic language detection when source_lang='auto'.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        source_lang: Source language code ('pl', 'en', etc.) or 'auto' for auto-detection
        target_lang: Target language code ('pl', 'en', etc.)
        batch_size: Number of segments to translate in one batch

    Returns:
        Tuple of (success: bool, message: str, translated_segments: List, detected_lang: str)
        - detected_lang: The detected source language (or original source_lang if not 'auto')
    """
    if not TRANSLATOR_AVAILABLE:
        return False, "Błąd: deep-translator nie jest zainstalowany. Zainstaluj: pip install deep-translator", [], ""

    if not segments:
        return True, "Brak segmentów do tłumaczenia", [], source_lang

    detected_lang = source_lang

    try:
        # Auto-detect language if source_lang is 'auto'
        if source_lang == 'auto':
            from .srt_reader import detect_language_from_segments
            detected_lang = detect_language_from_segments(segments)
            print(f"Wykryty język źródłowy: {detected_lang}")

            # Check if detected language equals target - skip translation
            if detected_lang == target_lang:
                print(f"Wykryty język ({detected_lang}) jest taki sam jak docelowy ({target_lang}) - pomijam tłumaczenie")
                return True, f"Tłumaczenie pominięte (wykryty język = docelowy: {detected_lang})", segments, detected_lang

            source_lang = detected_lang

        # Użyj bezpośrednio skrótów języków - GoogleTranslator je obsługuje
        translator = GoogleTranslator(source=source_lang, target=target_lang)

        translated_segments = []

        print(f"Tłumaczenie {len(segments)} segmentów ({source_lang} -> {target_lang})...")

        # Process in batches for efficiency
        with tqdm(total=len(segments), desc="Translating", unit="seg") as pbar:
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]

                # Extract texts for batch translation
                texts = [text for _, _, text in batch]

                # Translate texts
                try:
                    translated_texts = []
                    for text in texts:
                        try:
                            translated_text = translator.translate(text)
                            translated_texts.append(translated_text)
                        except Exception as e:
                            # Keep original text on translation error
                            tqdm.write(f"Ostrzeżenie: Nie udało się przetłumaczyć tekstu: {text[:50]}... ({str(e)})")
                            translated_texts.append(text)
                except Exception as e:
                    return False, f"Błąd podczas tłumaczenia: {str(e)}", [], detected_lang

                # Rebuild segments with translated text
                for j, (start_ms, end_ms, _) in enumerate(batch):
                    translated_text = translated_texts[j] if j < len(translated_texts) else batch[j][2]
                    translated_segments.append((start_ms, end_ms, translated_text))

                pbar.update(len(batch))

        return True, f"Przetłumaczono {len(translated_segments)} segmentów", translated_segments, detected_lang

    except Exception as e:
        return False, f"Błąd podczas tłumaczenia: {str(e)}", [], detected_lang
