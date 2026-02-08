#!/usr/bin/env python3
"""
SRT Corrector Module

This module provides functionality for correcting typos and grammatical errors
in SRT subtitle files using Large Language Models.

Key features:
- LLM-powered correction of OCR errors and typos
- Preservation of segment numbers and timestamps
- Generation of correction logs for review
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .output_manager import OutputManager
from .llm_translator import (
    get_llm_config,
    _call_openai_api,
    _call_anthropic_api,
    ENV_LLM_API_KEY
)


# Default batch size for segment processing
DEFAULT_BATCH_SIZE = 50


def filter_credits_segments(
    segments: List[Tuple[int, int, str]]
) -> Tuple[List[Tuple[int, int, str]], int]:
    """
    Filtruje segmenty z credits/metadata i czyści pozostałości z dialogów.

    Usuwa:
    - Segmenty składające się tylko z wielkich liter (nazwiska aktorów)
    - Segmenty z typowymi frazami credits (Created by, Written by, etc.)
    - Krótkie segmenty typu "and", "&" (pozostałości credits)
    - Napisy końcowe (end credits) z rolami i nazwiskami

    Czyści:
    - Prefiksy z nazwiskami na początku segmentów dialogowych
    - Sufiksy z nazwiskami na końcu segmentów dialogowych
    - Łączniki credits ("and", "&") z początku/końca tekstu

    Args:
        segments: Lista krotek (start_ms, end_ms, text)

    Returns:
        Tuple (filtered_segments, removed_count)
    """
    # Typowe frazy credits (case-insensitive)
    credits_phrases = [
        "series created by",
        "created by",
        "written by",
        "directed by",
        "inspired by",
        "based on",
        "executive producer",
        "produced by",
        "music by",
    ]

    # Krótkie słowa credits do usunięcia jako całe segmenty
    credits_short_words = {"and", "&"}

    filtered = []
    removed_count = 0

    for start_ms, end_ms, text in segments:
        text_stripped = text.strip()

        # Krok 1: Sprawdź czy to krótki śmieć (samo "and", "&")
        if text_stripped.lower() in credits_short_words:
            removed_count += 1
            continue

        # Krok 2: Sprawdź czy to credits phrase
        text_lower = text_stripped.lower()
        is_credits_phrase = any(phrase in text_lower for phrase in credits_phrases)

        if is_credits_phrase:
            removed_count += 1
            continue

        # Krok 3: Sprawdź czy to same wielkie litery (nazwiska aktorów)
        # Dopuszczamy spacje, myślniki i apostrofy w nazwiskach
        text_no_spaces = re.sub(r"[\s\-'.]", "", text_stripped)
        is_all_uppercase = (
            text_no_spaces.isupper()
            and len(text_no_spaces) > 0
            and len(text_stripped.split()) <= 5  # max 5 słów (imię + nazwisko x2)
        )

        if is_all_uppercase:
            removed_count += 1
            continue

        # Krok 4: Sprawdź czy to napisy końcowe (role + nazwiska)
        if _is_end_credits(text_stripped):
            removed_count += 1
            continue

        # Krok 5: Wyczyść prefiks (nazwiska na początku)
        cleaned = _clean_uppercase_prefix(text_stripped)

        # Krok 6: Wyczyść sufiks (nazwiska na końcu)
        cleaned = _clean_uppercase_suffix(cleaned)

        # Krok 7: Usuń "and" z początku/końca
        cleaned = _clean_credits_connectors(cleaned)

        # Krok 8: Sprawdź czy coś zostało
        if cleaned:
            filtered.append((start_ms, end_ms, cleaned))
        else:
            removed_count += 1

    return filtered, removed_count


def _is_end_credits(text: str) -> bool:
    """
    Wykrywa napisy końcowe (end credits) z rolami i nazwiskami.

    Przykłady do usunięcia:
        "Stunt Coordinator Pete Ford Stunt Performer Emma Britton"
        "Assistant to Ms Parfitt Jackie Cast Location Manager Olivia O'Brien"
        "Sylvie Baldwin Olivia Young Tracey Baldwin"
        "Location Assis..." (ucięte)
        "Productio..." (ucięte)

    Returns:
        True jeśli tekst to prawdopodobnie napisy końcowe
    """
    if not text:
        return False

    text_lower = text.lower()

    # Typowe role filmowe i ich początki (dla ucięć)
    film_roles = [
        "coordinator", "manager", "producer", "director", "assistant",
        "performer", "supervisor", "editor", "designer", "consultant",
        "officer", "head of", "stunt", "location", "production",
        "casting", "wardrobe", "makeup", "camera", "sound", "gaffer",
        "grip", "electrician", "driver", "caterer", "accountant",
        "secretary", "runner", "trainee", "double", "stand-in"
    ]

    # Początki ról (dla ucięć typu "Productio...", "Assis...")
    film_role_prefixes = [
        "coordin", "manag", "produc", "direct", "assist", "perform",
        "supervi", "edit", "design", "consult", "offic", "locat",
        "cast", "wardr", "makeu", "camer", "electri", "account",
        "secreta", "train"
    ]

    # Sprawdź czy zawiera role filmowe
    has_film_role = any(role in text_lower for role in film_roles)

    # Sprawdź czy to ucięta rola
    is_truncated = text.rstrip().endswith("...")
    text_no_dots = text_lower.rstrip(".")

    has_truncated_role = False
    if is_truncated:
        # Sprawdź czy ucięte słowo to początek roli
        for prefix in film_role_prefixes:
            if text_no_dots.endswith(prefix) or prefix in text_no_dots:
                has_truncated_role = True
                break

    # Policz słowa zaczynające się wielką literą (ignoruj pojedyncze litery/inicjały)
    words = text.split()

    # Krótkie teksty - sprawdź tylko role
    if len(words) < 3:
        if has_film_role or has_truncated_role:
            return True
        return False

    # Dla wzorca nazwisk liczymy tylko "prawdziwe" słowa (min 2 znaki)
    real_words = [w for w in words if len(re.sub(r'[^\w]', '', w)) >= 2]
    if len(real_words) < 3:
        return False

    capitalized_words = sum(1 for w in real_words if w and w[0].isupper())
    capitalized_ratio = capitalized_words / len(real_words)

    # Sprawdź czy brak typowej interpunkcji dialogowej
    has_dialog_punctuation = bool(re.search(r'[?!]|,\s+\w', text))

    # Wykrywanie wzorca "Imię Nazwisko Imię Nazwisko..."
    # Większość słów z wielkiej litery, brak interpunkcji dialogowej
    is_names_pattern = (
        capitalized_ratio >= 0.7
        and not has_dialog_punctuation
        and len(words) >= 4
    )

    # Decyzja: credits jeśli ma role filmowe LUB wzorzec nazwisk LUB ucięta rola
    if has_film_role:
        return True

    if has_truncated_role:
        return True

    if is_names_pattern:
        return True

    # Ucięte słowa z większością wielkich liter mogą być credits
    if is_truncated and capitalized_ratio >= 0.5 and len(words) >= 2:
        return True

    return False


def _clean_uppercase_suffix(text: str) -> str:
    """
    Usuwa sufiks z wielkich liter (nazwiska) z końca tekstu dialogowego.

    Przykład:
        "tekst dialogu A R FITT" -> "tekst dialogu"
        "Nie każde rozstanie jest bolesneA R FITT" -> "Nie każde rozstanie jest bolesne"
        "Normalny tekst" -> "Normalny tekst"

    Returns:
        Oczyszczony tekst
    """
    if not text:
        return ""

    words = text.split()
    if len(words) < 3:
        return text

    # Szukaj od końca - gdzie kończą się słowa all-caps
    dialog_end_idx = len(words)

    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        # Usuń interpunkcję do analizy
        word_clean = re.sub(r'[^\w]', '', word)

        if not word_clean:
            continue

        # Jeśli słowo nie jest all-caps - to koniec dialogu
        if not word_clean.isupper():
            dialog_end_idx = i + 1
            break
    else:
        # Wszystkie słowa to caps - prawdopodobnie cały tekst to credits
        return ""

    # Sprawdź ile słów all-caps jest na końcu
    caps_suffix_len = len(words) - dialog_end_idx

    if caps_suffix_len < 2:
        # Mniej niż 2 słowa all-caps - prawdopodobnie nie nazwisko
        return text

    # Usuń sufiks z nazwiskami
    dialog_words = words[:dialog_end_idx]

    # Usuń trailing interpunkcję z ostatniego słowa dialogu jeśli wygląda na błąd OCR
    # np. "bolesneA" - gdzie "A" to początek nazwiska połączony z dialogiem
    if dialog_words:
        last_word = dialog_words[-1]
        # Sprawdź czy ostatnie słowo kończy się wielką literą poprzedzoną małą
        # Wzorzec: "słowoX" lub "słowoXY"
        match = re.match(r'^(.+[a-z])([A-Z]+)$', last_word)
        if match:
            dialog_words[-1] = match.group(1)

    return " ".join(dialog_words).strip()


def _clean_credits_connectors(text: str) -> str:
    """
    Usuwa łączniki credits ("and", "&") z początku i końca tekstu.

    Przykłady:
        "and Nie każde rozstanie" -> "Nie każde rozstanie"
        "tekst dialogu and" -> "tekst dialogu"
        "& Przykład" -> "Przykład"

    Returns:
        Oczyszczony tekst
    """
    if not text:
        return ""

    # Usuń z początku
    text = re.sub(r'^(?:and|AND|&)\s+', '', text)

    # Usuń z końca
    text = re.sub(r'\s+(?:and|AND|&)$', '', text)

    return text.strip()


def _clean_uppercase_prefix(text: str) -> str:
    """
    Usuwa prefiks z wielkich liter (nazwiska) z początku tekstu dialogowego.

    Przykład:
        "JENNY AGUTTER LINDA BASSETT Nie każde rozstanie..." -> "Nie każde rozstanie..."
        "LAURA MKto nie pamięta" -> "MKto nie pamięta" (jedno nazwisko + błąd OCR)
        "Normalny tekst dialogu" -> "Normalny tekst dialogu"
        "SAME WIELKIE LITERY" -> "" (pusty - do usunięcia)
        "A to zaczyna się od jednej litery" -> "A to zaczyna się od jednej litery"

    Returns:
        Oczyszczony tekst lub pusty string jeśli był to sam credits
    """
    if not text:
        return ""

    # Znajdź pozycję gdzie zaczynają się małe litery (dialog)
    # Szukamy pierwszego słowa zaczynającego się od małej litery
    # lub słowa z mieszanymi literami (nie all-caps)

    words = text.split()
    dialog_start_idx = None

    for i, word in enumerate(words):
        # Usuń interpunkcję do analizy
        word_clean = re.sub(r'[^\w]', '', word)

        if not word_clean:
            continue

        # Jeśli słowo nie jest CAŁE wielkie - to początek dialogu
        if not word_clean.isupper():
            dialog_start_idx = i
            break

    if dialog_start_idx is None:
        # Cały tekst to wielkie litery - prawdopodobnie credits
        return ""

    if dialog_start_idx == 0:
        # Tekst nie zaczyna się od wielkiego prefixu - zwróć bez zmian
        return text

    # Sprawdź czy prefix to prawdopodobnie nazwisko
    # Minimum 2 słowa all-caps, LUB 1 słowo all-caps które wygląda jak nazwisko
    # (nie jest krótkim słowem typu OK, A, I)
    if dialog_start_idx == 1:
        first_word_clean = re.sub(r'[^\w]', '', words[0])
        # Krótkie słowa (1-2 znaki) lub typowe skróty - nie usuwaj
        if len(first_word_clean) <= 2 or first_word_clean in {"OK", "TV", "CD", "DVD", "DJ"}:
            return text
        # Dłuższe słowo all-caps - prawdopodobnie nazwisko (np. LAURA)
        # Usuń je

    # Zwróć tekst od miejsca gdzie zaczyna się dialog
    dialog_text = " ".join(words[dialog_start_idx:])
    return dialog_text.strip()


def correct_srt_with_llm(
    segments: List[Tuple[int, int, str]],
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[bool, str, List[Tuple[int, int, str]], List[dict]]:
    """
    Correct segments using LLM for typo and grammar fixes.

    Processes segments in batches to avoid token limits.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        provider: LLM provider (openai, ollama, anthropic)
        model: Model name
        base_url: API base URL
        api_key: API key
        batch_size: Number of segments per API call (default: 50)

    Returns:
        Tuple (success, message, corrected_segments, changes_log)

        changes_log: List of dicts:
        [
            {"segment": 1, "original": "bytość", "corrected": "byłość"},
            {"segment": 5, "original": "Dópa", "corrected": "Dupa"},
        ]
    """
    config = get_llm_config(provider, model, base_url, api_key)

    OutputManager.info(
        f"Korekta SRT przez LLM: {config['provider']} / {config['model']}"
    )

    # Validate API key for non-local providers
    if config["provider"] != "ollama" and not config["api_key"]:
        return False, f"Brak klucza API. Ustaw zmienną {ENV_LLM_API_KEY}", [], []

    # Filtruj credits przed korektą - zawsze włączone
    segments, removed = filter_credits_segments(segments)
    if removed > 0:
        OutputManager.info(f"Usunięto {removed} segmentów z credits/metadata")

    if not segments:
        return True, "Wszystkie segmenty były credits/metadata - brak do korekty", [], []

    # Process in batches
    all_corrected = []
    all_changes = []
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(segments))
        batch_segments = segments[start_idx:end_idx]

        OutputManager.info(
            f"  Batch {batch_idx + 1}/{total_batches}: segmenty {start_idx + 1}-{end_idx}"
        )

        # Build correction prompt for this batch
        prompt = _build_correction_prompt(batch_segments, segment_offset=start_idx)

        # Call LLM API
        try:
            if config["api_format"] == "anthropic":
                response = _call_anthropic_api(prompt, config)
            else:
                response = _call_openai_api(prompt, config)
        except Exception as e:
            return False, f"Błąd API LLM (batch {batch_idx + 1}): {str(e)}", [], []

        # Parse response back to segments
        try:
            corrected_batch = _parse_correction_response(
                response, batch_segments, segment_offset=start_idx
            )
            all_corrected.extend(corrected_batch)

            # Compare and generate changes log for this batch
            batch_changes = _compare_segments(
                batch_segments, corrected_batch, segment_offset=start_idx
            )
            all_changes.extend(batch_changes)

        except Exception as e:
            OutputManager.warning(f"Błąd parsowania odpowiedzi LLM (batch {batch_idx + 1}): {e}")
            return False, f"Błąd parsowania odpowiedzi LLM: {str(e)}", [], []

    # Verify total count
    if len(all_corrected) != len(segments):
        OutputManager.warning(
            f"Liczba poprawionych segmentów ({len(all_corrected)}) "
            f"różni się od oryginału ({len(segments)})"
        )

    changes_count = len(all_changes)
    msg = f"Poprawiono {len(all_corrected)} segmentów przez LLM ({total_batches} batchy). Zmian: {changes_count}"
    return True, msg, all_corrected, all_changes


def _build_correction_prompt(
    segments: List[Tuple[int, int, str]],
    segment_offset: int = 0
) -> str:
    """Build correction prompt for LLM."""

    # Build segment list with markers (using global indices)
    segment_lines = []
    for i, (start_ms, end_ms, text) in enumerate(segments):
        global_idx = segment_offset + i
        segment_lines.append(f"[SEG_{global_idx:04d}] {text}")

    segments_text = "\n".join(segment_lines)

    prompt = f"""Popraw literówki i błędy gramatyczne w poniższych napisach.

ZASADY:
- Zachowaj numery segmentów [SEG_XXXX] dokładnie jak w oryginale
- Poprawiaj tylko oczywiste błędy (literówki, błędy ortograficzne/gramatyczne)
- NIE zmieniaj stylu, interpunkcji ani znaczenia
- Częste błędy OCR: "t" zamiast "ł", "rn" zamiast "m", "l" zamiast "I"
- Błędy OCR często wstawiają wielkie litery w środek słów: "TByoTtO" -> "To było to", "MKto" -> "Kto"
- Napraw słowa z przypadkowymi wielkimi literami w środku (np. "bolesneA" -> "bolesne")
- Segmenty zostały już oczyszczone z credits/nazwisk - poprawiaj tylko tekst dialogów
- Każdy segment na JEDNEJ linii: [SEG_XXXX] tekst
- Zwróć TYLKO poprawione segmenty (bez komentarzy)

NAPISY:
{segments_text}

POPRAWIONE:"""

    return prompt


def _parse_correction_response(
    response: str,
    original_segments: List[Tuple[int, int, str]],
    segment_offset: int = 0
) -> List[Tuple[int, int, str]]:
    """
    Parse LLM response back into segment tuples.

    Expected format:
    [SEG_0000] Poprawiony tekst.
    [SEG_0001] Inny tekst.

    Obsługuje też multiline - gdy tekst segmentu jest rozłożony na wiele linii,
    zbiera linie aż do następnego markera [SEG_XXXX].

    Args:
        response: LLM response text
        original_segments: Original segments for this batch
        segment_offset: Global offset for segment indices
    """
    # Create mapping of global segment indices to (local_idx, timing, original_text)
    segment_timing = {}
    for local_idx, (start, end, orig_text) in enumerate(original_segments):
        global_idx = segment_offset + local_idx
        segment_timing[global_idx] = (start, end, orig_text)

    # Parse response - obsługa multiline
    lines = response.strip().split("\n")
    parsed_segments = {}  # seg_idx -> text

    current_seg_idx = None
    current_text_lines = []

    for line in lines:
        # Match segment marker
        match = re.match(r'\[SEG_(\d+)\]\s*(.*)', line)
        if match:
            # Zapisz poprzedni segment (jeśli był)
            if current_seg_idx is not None and current_seg_idx in segment_timing:
                parsed_segments[current_seg_idx] = " ".join(current_text_lines).strip()

            # Nowy segment
            current_seg_idx = int(match.group(1))
            text_after_marker = match.group(2).strip()
            current_text_lines = [text_after_marker] if text_after_marker else []
        elif current_seg_idx is not None:
            # Kontynuacja tekstu (multiline)
            stripped = line.strip()
            if stripped:
                current_text_lines.append(stripped)

    # Zapisz ostatni segment
    if current_seg_idx is not None and current_seg_idx in segment_timing:
        parsed_segments[current_seg_idx] = " ".join(current_text_lines).strip()

    # Buduj listę wynikową - zachowaj kolejność oryginalnych segmentów
    # Użyj fallback na oryginał dla brakujących
    corrected_segments = []
    missing_count = 0

    for local_idx, (start, end, orig_text) in enumerate(original_segments):
        global_idx = segment_offset + local_idx

        if global_idx in parsed_segments and parsed_segments[global_idx]:
            corrected_segments.append((start, end, parsed_segments[global_idx]))
        else:
            # Fallback na oryginał
            corrected_segments.append((start, end, orig_text))
            missing_count += 1

    if missing_count > 0:
        OutputManager.warning(
            f"Fallback na oryginał dla {missing_count}/{len(original_segments)} segmentów"
        )

    return corrected_segments


def _compare_segments(
    original_segments: List[Tuple[int, int, str]],
    corrected_segments: List[Tuple[int, int, str]],
    segment_offset: int = 0
) -> List[dict]:
    """
    Compare original and corrected segments to generate changes log.

    Args:
        original_segments: Original segments
        corrected_segments: Corrected segments
        segment_offset: Global offset for segment indices

    Returns:
        List of change dictionaries with keys: segment, original, corrected
    """
    changes = []

    for i, (orig, corr) in enumerate(zip(original_segments, corrected_segments)):
        orig_text = orig[2]
        corr_text = corr[2]

        if orig_text != corr_text:
            changes.append({
                "segment": segment_offset + i + 1,  # 1-based segment number
                "original": orig_text,
                "corrected": corr_text
            })

    return changes


def write_corrections_log(
    changes: List[dict],
    source_filename: str,
    output_path: str,
    provider: str = None,
    model: str = None
) -> Tuple[bool, str]:
    """
    Write corrections log to a text file.

    Args:
        changes: List of change dictionaries from _compare_segments
        source_filename: Name of the source SRT file
        output_path: Path for the output log file
        provider: LLM provider used
        model: LLM model used

    Returns:
        Tuple (success, message)
    """
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        provider_info = f"{provider or 'auto'} / {model or 'auto'}"

        lines = [
            "=== Raport korekty SRT ===",
            f"Plik źródłowy: {source_filename}",
            f"Data: {now}",
            f"Provider: {provider_info}",
            f"Liczba zmian: {len(changes)}",
            "",
            "--- Zmiany ---",
            ""
        ]

        if not changes:
            lines.append("Brak zmian - tekst nie wymagał korekty.")
        else:
            for change in changes:
                lines.append(f"Segment {change['segment']}:")
                lines.append(f"  Oryginał: {change['original']}")
                lines.append(f"  Korekta:  {change['corrected']}")
                lines.append("")

        content = "\n".join(lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True, f"Zapisano log zmian: {output_path}"

    except Exception as e:
        return False, f"Błąd zapisu logu zmian: {str(e)}"


def check_correction_availability(
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None
) -> Tuple[bool, str]:
    """
    Check if LLM correction is available with current configuration.

    Returns:
        Tuple (available, message)
    """
    config = get_llm_config(provider, model, base_url, api_key)

    # Check API key for non-local providers
    if config["provider"] != "ollama" and not config["api_key"]:
        return False, f"Brak klucza API ({ENV_LLM_API_KEY})"

    # Check if required library is installed
    if config["api_format"] == "anthropic":
        try:
            import anthropic
        except ImportError:
            return False, "Biblioteka 'anthropic' nie jest zainstalowana"
    else:
        try:
            import openai
        except ImportError:
            return False, "Biblioteka 'openai' nie jest zainstalowana"

    return True, f"LLM dostępny: {config['provider']} / {config['model']}"
