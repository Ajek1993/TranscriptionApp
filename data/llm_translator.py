#!/usr/bin/env python3
"""
LLM Translator Module

This module provides context-aware translation using Large Language Models.
Supports OpenAI-compatible and Anthropic-compatible APIs.

Key features:
- Speaker gender-aware translation for proper grammatical inflection
- Full context translation (entire transcript at once)
- Support for multiple LLM providers (OpenAI, Ollama, Anthropic, api.z.ai)
"""

import os
import re
from typing import Dict, List, Tuple, Optional

from .output_manager import OutputManager
from .speaker_analyzer import format_speaker_info_for_prompt


# Environment variable names for configuration
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_API_FORMAT = "LLM_API_FORMAT"  # "openai" or "anthropic"


# Language name mapping for prompts
LANGUAGE_NAMES = {
    "pl": "polski",
    "en": "angielski",
    "de": "niemiecki",
    "fr": "francuski",
    "es": "hiszpański",
    "it": "włoski",
    "pt": "portugalski",
    "ru": "rosyjski",
    "uk": "ukraiński",
    "ja": "japoński",
    "ko": "koreański",
    "zh": "chiński",
}


def get_llm_config(
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None
) -> dict:
    """
    Get LLM configuration from parameters or environment variables.

    Priority: function parameters > environment variables > defaults

    Returns:
        Dict with keys: provider, model, base_url, api_key, api_format
    """
    # API key
    resolved_api_key = api_key or os.environ.get(ENV_LLM_API_KEY, "")

    # Base URL
    resolved_base_url = base_url or os.environ.get(ENV_LLM_BASE_URL, "")

    # Model
    resolved_model = model or os.environ.get(ENV_LLM_MODEL, "")

    # API format (openai or anthropic)
    env_format = os.environ.get(ENV_LLM_API_FORMAT, "").lower()

    # Determine provider and API format
    if provider:
        resolved_provider = provider.lower()
    elif resolved_base_url:
        # Guess provider from base URL
        if "anthropic" in resolved_base_url.lower() or "z.ai" in resolved_base_url.lower():
            resolved_provider = "anthropic"
        elif "ollama" in resolved_base_url.lower() or "localhost:11434" in resolved_base_url:
            resolved_provider = "ollama"
        else:
            resolved_provider = "openai"
    else:
        resolved_provider = env_format or "openai"

    # Set API format based on provider
    if resolved_provider in ("anthropic", "claude"):
        api_format = "anthropic"
    else:
        api_format = "openai"

    # Override with explicit env format if set
    if env_format in ("openai", "anthropic"):
        api_format = env_format

    # Set default model if not specified
    if not resolved_model:
        if resolved_provider == "ollama":
            resolved_model = "llama3.1"
        elif resolved_provider == "anthropic":
            resolved_model = "claude-sonnet-4-20250514"
        else:
            resolved_model = "gpt-4o-mini"

    # Set default base URL if not specified
    if not resolved_base_url:
        if resolved_provider == "ollama":
            resolved_base_url = "http://localhost:11434/v1"
        elif resolved_provider == "anthropic":
            resolved_base_url = "https://api.anthropic.com"
        else:
            resolved_base_url = "https://api.openai.com/v1"

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
        "api_format": api_format
    }


# Default batch size for segment processing
DEFAULT_BATCH_SIZE = 50  # Segments per API call


def translate_with_llm(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    speaker_info: Dict[str, dict] = None,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Translate segments using LLM with speaker gender context.

    Processes segments in batches to avoid token limits.

    Args:
        segments: List of (start_ms, end_ms, text) tuples
        source_lang: Source language code (e.g., "en")
        target_lang: Target language code (e.g., "pl")
        speaker_info: Dict from speaker_analyzer with gender info
        provider: LLM provider (openai, ollama, anthropic)
        model: Model name
        base_url: API base URL
        api_key: API key
        batch_size: Number of segments per API call (default: 50)

    Returns:
        Tuple (success, message, translated_segments)
    """
    config = get_llm_config(provider, model, base_url, api_key)

    OutputManager.info(
        f"Tłumaczenie LLM: {config['provider']} / {config['model']}"
    )

    # Validate API key for non-local providers
    if config["provider"] != "ollama" and not config["api_key"]:
        return False, f"Brak klucza API. Ustaw zmienną {ENV_LLM_API_KEY}", []

    # Process in batches
    all_translated = []
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(segments))
        batch_segments = segments[start_idx:end_idx]

        OutputManager.info(
            f"  Batch {batch_idx + 1}/{total_batches}: segmenty {start_idx + 1}-{end_idx}"
        )

        # Build translation prompt for this batch
        prompt = _build_translation_prompt(
            batch_segments, source_lang, target_lang, speaker_info,
            segment_offset=start_idx
        )

        # Call LLM API
        try:
            if config["api_format"] == "anthropic":
                response = _call_anthropic_api(prompt, config)
            else:
                response = _call_openai_api(prompt, config)
        except Exception as e:
            return False, f"Błąd API LLM (batch {batch_idx + 1}): {str(e)}", []

        # Parse response back to segments
        try:
            translated_batch = _parse_llm_response(
                response, batch_segments, segment_offset=start_idx
            )
            all_translated.extend(translated_batch)
        except Exception as e:
            OutputManager.warning(f"Błąd parsowania odpowiedzi LLM (batch {batch_idx + 1}): {e}")
            return False, f"Błąd parsowania odpowiedzi LLM: {str(e)}", []

    # Verify total count
    if len(all_translated) != len(segments):
        OutputManager.warning(
            f"Liczba przetłumaczonych segmentów ({len(all_translated)}) "
            f"różni się od oryginału ({len(segments)})"
        )

    return True, f"Przetłumaczono {len(all_translated)} segmentów przez LLM ({total_batches} batchy)", all_translated


def _build_translation_prompt(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    speaker_info: Dict[str, dict] = None,
    segment_offset: int = 0
) -> str:
    """Build translation prompt with speaker context."""

    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    # Build speaker section if available
    speaker_section = ""
    if speaker_info:
        speaker_lines = format_speaker_info_for_prompt(speaker_info)
        if speaker_lines:
            speaker_section = f"""
MÓWCY:
{speaker_lines}

"""

    # Build segment list with markers (using global indices)
    segment_lines = []
    for i, (start_ms, end_ms, text) in enumerate(segments):
        global_idx = segment_offset + i
        segment_lines.append(f"[SEG_{global_idx:04d}] {text}")

    segments_text = "\n".join(segment_lines)

    prompt = f"""Przetłumacz poniższą transkrypcję z języka {source_name} na {target_name}.
{speaker_section}ZASADY:
- Użyj poprawnej odmiany gramatycznej dla każdego mówcy (na podstawie jego płci)
- Zachowaj numery segmentów [SEG_XXXX] dokładnie jak w oryginale
- Zachowaj oznaczenia mówców [SPEAKER_XX] jeśli występują
- Nie dodawaj żadnych komentarzy ani wyjaśnień
- Zwróć TYLKO przetłumaczone segmenty

TRANSKRYPCJA:
{segments_text}

TŁUMACZENIE:"""

    return prompt


def _call_openai_api(prompt: str, config: dict) -> str:
    """Call OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai nie jest zainstalowany. Zainstaluj: pip install openai"
        )

    client = OpenAI(
        api_key=config["api_key"] or "dummy",  # Ollama doesn't need key
        base_url=config["base_url"]
    )

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent translations
        max_tokens=8192
    )

    return response.choices[0].message.content


def _call_anthropic_api(prompt: str, config: dict) -> str:
    """Call Anthropic-compatible API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "anthropic nie jest zainstalowany. Zainstaluj: pip install anthropic"
        )

    # Handle custom base URL (e.g., api.z.ai)
    base_url = config["base_url"]
    if base_url and not base_url.endswith("/v1"):
        # Anthropic SDK expects base URL without /v1
        if "/v1" in base_url:
            base_url = base_url.replace("/v1", "")

    client = Anthropic(
        api_key=config["api_key"],
        base_url=base_url if base_url != "https://api.anthropic.com" else None
    )

    response = client.messages.create(
        model=config["model"],
        max_tokens=8192,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text


def _parse_llm_response(
    response: str,
    original_segments: List[Tuple[int, int, str]],
    segment_offset: int = 0
) -> List[Tuple[int, int, str]]:
    """
    Parse LLM response back into segment tuples.

    Expected format:
    [SEG_0000] [SPEAKER_00] Przetłumaczony tekst.
    [SEG_0001] [SPEAKER_01] Inny tekst.

    Args:
        response: LLM response text
        original_segments: Original segments for this batch
        segment_offset: Global offset for segment indices
    """
    translated_segments = []

    # Create mapping of global segment indices to (local_idx, timing)
    segment_timing = {}
    for local_idx, (start, end, _) in enumerate(original_segments):
        global_idx = segment_offset + local_idx
        segment_timing[global_idx] = (start, end)

    # Parse response line by line
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match segment marker
        match = re.match(r'\[SEG_(\d+)\]\s*(.*)', line)
        if match:
            seg_idx = int(match.group(1))
            text = match.group(2).strip()

            if seg_idx in segment_timing:
                start_ms, end_ms = segment_timing[seg_idx]
                translated_segments.append((start_ms, end_ms, text))

    # Fallback: if parsing failed, try to match by line count
    if len(translated_segments) == 0:
        # Filter out empty lines for comparison
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        if len(non_empty_lines) == len(original_segments):
            OutputManager.warning("Fallback: dopasowanie segmentów po liczbie linii")
            for i, line in enumerate(non_empty_lines):
                # Remove any segment markers that might be present
                text = re.sub(r'^\[SEG_\d+\]\s*', '', line)
                start_ms, end_ms, _ = original_segments[i]
                translated_segments.append((start_ms, end_ms, text))

    return translated_segments


def check_llm_availability(
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None
) -> Tuple[bool, str]:
    """
    Check if LLM translation is available with current configuration.

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
