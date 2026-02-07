#!/usr/bin/env python3
"""
Translation Module
Handles translation of text segments using Google Translator or LLM.

Supports:
- Google Translator (default) - fast, free, segment-by-segment
- LLM translation - context-aware, gender-aware grammatical inflection
"""

from typing import List, Tuple, Dict, Optional
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
    batch_size: int = 50,
    use_llm: bool = False,
    speaker_info: Dict[str, dict] = None,
    llm_provider: str = None,
    llm_model: str = None,
    llm_base_url: str = None,
    llm_api_key: str = None
) -> Tuple[bool, str, List[Tuple[int, int, str]], str]:
    """
    Translate text content of segments while preserving timestamps.

    Supports:
    - Google Translator (default) - fast, segment-by-segment
    - LLM translation (use_llm=True) - context-aware, gender-aware

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        source_lang: Source language code ('pl', 'en', etc.) or 'auto' for auto-detection
        target_lang: Target language code ('pl', 'en', etc.)
        batch_size: Number of segments to translate in one batch (Google Translator only)
        use_llm: Use LLM for translation instead of Google Translator
        speaker_info: Speaker gender info from speaker_analyzer (for LLM)
        llm_provider: LLM provider (openai, ollama, anthropic)
        llm_model: LLM model name
        llm_base_url: LLM API base URL
        llm_api_key: LLM API key

    Returns:
        Tuple of (success: bool, message: str, translated_segments: List, detected_lang: str)
        - detected_lang: The detected source language (or original source_lang if not 'auto')
    """
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

        # Try LLM translation if requested
        if use_llm:
            return _translate_with_llm(
                segments, source_lang, target_lang,
                speaker_info, llm_provider, llm_model, llm_base_url, llm_api_key,
                detected_lang
            )

        # Fall back to Google Translator
        if not TRANSLATOR_AVAILABLE:
            return False, "Błąd: deep-translator nie jest zainstalowany. Zainstaluj: pip install deep-translator", [], ""

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


def _translate_with_llm(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    speaker_info: Dict[str, dict],
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    detected_lang: str
) -> Tuple[bool, str, List[Tuple[int, int, str]], str]:
    """
    Internal function to handle LLM translation with fallback.
    """
    try:
        from .llm_translator import translate_with_llm, check_llm_availability

        # Check if LLM is available
        available, msg = check_llm_availability(
            llm_provider, llm_model, llm_base_url, llm_api_key
        )

        if not available:
            print(f"LLM niedostępny: {msg}")
            print("Fallback na Google Translator...")
            # Return None to signal fallback should happen
            return _fallback_to_google(
                segments, source_lang, target_lang, detected_lang
            )

        # Attempt LLM translation
        success, message, translated = translate_with_llm(
            segments, source_lang, target_lang,
            speaker_info=speaker_info,
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key
        )

        if success:
            return True, message, translated, detected_lang

        # LLM failed, try fallback
        print(f"LLM translation failed: {message}")
        print("Fallback na Google Translator...")
        return _fallback_to_google(
            segments, source_lang, target_lang, detected_lang
        )

    except ImportError:
        print("Moduł llm_translator niedostępny, używam Google Translator...")
        return _fallback_to_google(
            segments, source_lang, target_lang, detected_lang
        )
    except Exception as e:
        print(f"Błąd LLM: {e}")
        print("Fallback na Google Translator...")
        return _fallback_to_google(
            segments, source_lang, target_lang, detected_lang
        )


def _fallback_to_google(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    detected_lang: str
) -> Tuple[bool, str, List[Tuple[int, int, str]], str]:
    """
    Fallback translation using Google Translator.
    """
    if not TRANSLATOR_AVAILABLE:
        return False, "Błąd: deep-translator nie jest zainstalowany i LLM niedostępny", [], detected_lang

    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_segments = []

        print(f"Tłumaczenie {len(segments)} segmentów ({source_lang} -> {target_lang})...")

        with tqdm(total=len(segments), desc="Translating", unit="seg") as pbar:
            for start_ms, end_ms, text in segments:
                try:
                    translated_text = translator.translate(text)
                    translated_segments.append((start_ms, end_ms, translated_text))
                except Exception as e:
                    tqdm.write(f"Ostrzeżenie: Nie udało się przetłumaczyć: {text[:50]}... ({e})")
                    translated_segments.append((start_ms, end_ms, text))
                pbar.update(1)

        return True, f"Przetłumaczono {len(translated_segments)} segmentów (Google)", translated_segments, detected_lang

    except Exception as e:
        return False, f"Błąd podczas tłumaczenia: {str(e)}", [], detected_lang
