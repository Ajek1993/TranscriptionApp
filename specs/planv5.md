# Plan v5 - Refaktoryzacja silników transkrypcji i optymalizacja Docker

## 1. Przegląd zmian

### 1.1. Cele główne
1. **Zmiana domyślnego silnika transkrypcji** na `whisper` (OpenAI Whisper z GPU/CUDA)
2. **Faster-Whisper jako silnik CPU** - wymuszenie użycia CPU
3. **Implementacja WhisperX** jako trzeci silnik transkrypcyjny
4. **Refaktoryzacja kodu** - wydzielenie silników do osobnych funkcji
5. **Optymalizacja Dockerfile** - zmiana bazowego obrazu NVIDIA CUDA
6. **Plan przyszłościowy** - Coqui TTS, Piper TTS

### 1.2. Zmienione domyślne wartości
- `--engine`: **whisper** (było: `faster-whisper`)
- Faster-Whisper: automatyczne wymuszenie CPU
- Whisper: automatyczne użycie GPU/CUDA jeśli dostępne

---

## 2. KROK 1: Zmiana domyślnych silników transkrypcji

### 2.1. Zmiana domyślnego silnika na `whisper`

**Plik:** `data/transcribe.py`
**Lokalizacja:** Linia ~2136

**Zmiana:**
```python
# PRZED (linia 2136-2138)
transcription_group.add_argument('--engine', default='faster-whisper',
               choices=['faster-whisper', 'whisper'],
               help='Silnik transkrypcji (domyślnie: faster-whisper)')

# PO
transcription_group.add_argument('--engine', default='whisper',
               choices=['whisper', 'faster-whisper', 'whisperx'],
               help='Silnik transkrypcji (domyślnie: whisper)')
```

**Uzasadnienie:**
- Whisper (OpenAI) lepiej wykorzystuje GPU/CUDA
- Faster-Whisper będzie używany dla CPU
- WhisperX jako trzecia opcja (zaawansowane funkcje)

### 2.2. Faster-Whisper: wymuszenie CPU

**Plik:** `data/transcribe.py`
**Lokalizacja:** W refaktoryzowanej funkcji `transcribe_with_faster_whisper()` (nowa funkcja)

**Implementacja:**
```python
def transcribe_with_faster_whisper(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z Faster-Whisper (zawsze CPU).

    Faster-Whisper wymusza użycie CPU ze względu na problemy z CTranslate2.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return False, "Błąd: faster-whisper nie jest zainstalowany. Zainstaluj: pip install faster-whisper", []

    # WYMUSZENIE CPU dla Faster-Whisper
    device = "cpu"
    device_info = "CPU (faster-whisper - wymuszony CPU)"
    tqdm.write(f"Używane urządzenie: {device_info}")
    tqdm.write("  INFO: Faster-Whisper używa CPU ze względu na kompatybilność CTranslate2")

    # Inicjalizacja modelu (zawsze CPU)
    tqdm.write(f"Ładowanie modelu {model_size}...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        return False, f"Błąd podczas ładowania modelu faster-whisper: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja: {Path(wav_path).name}...")

    if timeout_seconds > 0:
        tqdm.write(f"Timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} min)")
    else:
        tqdm.write("Timeout: disabled")

    vad_params = dict(
        threshold=0.5, min_speech_duration_ms=250, max_speech_duration_s=15,
        min_silence_duration_ms=500, speech_pad_ms=400
    )

    # Transcribe with timeout
    success, error_msg, segments, info = _run_transcription_with_timeout(
        model, str(wav_path), language, timeout_seconds, segment_progress_bar, vad_params
    )

    if not success:
        return False, error_msg, []

    if info:
        tqdm.write(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments
```

### 2.3. Whisper: automatyczne GPU/CUDA

**Plik:** `data/transcribe.py`
**Lokalizacja:** Nowa funkcja `transcribe_with_whisper()`

**Implementacja:**
```python
def transcribe_with_whisper(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z OpenAI Whisper (automatyczne GPU/CUDA).

    Whisper automatycznie wykorzystuje GPU jeśli dostępne.
    """
    try:
        import whisper
        import torch
    except ImportError:
        return False, "Błąd: whisper nie jest zainstalowany. Zainstaluj: pip install openai-whisper", []

    # Detekcja urządzenia (GPU preferred)
    device, device_info = detect_device()
    tqdm.write(f"Używane urządzenie: {device_info}")

    # Ładowanie modelu
    tqdm.write(f"Ładowanie modelu OpenAI Whisper {model_size}...")
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        if device == "cuda":
            tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU. Błąd: {e}")
            device = "cpu"
            model = whisper.load_model(model_size, device="cpu")
        else:
            return False, f"Błąd podczas ładowania modelu whisper: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja: {Path(wav_path).name}...")

    # OpenAI Whisper nie używa timeout wrapper (prostsza implementacja)
    try:
        result = model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda")
        )
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []

    # Parsowanie segmentów
    segments = []
    for segment in result["segments"]:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        text = segment["text"].strip()
        segments.append((start_ms, end_ms, text))

        if segment_progress_bar:
            segment_progress_bar.set_postfix_str(f"{len(segments)} segments")

    tqdm.write(f"Wykryty język: {result['language']}")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments
```

---

## 3. KROK 2: Implementacja WhisperX

### 3.1. Dodanie zależności

**Plik:** `requirements.txt`
**Dodać po linii 14:**

```txt
# WhisperX - advanced transcription with word-level alignment
whisperx>=3.1.1
```

### 3.2. Funkcja transkrypcji WhisperX

**Plik:** `data/transcribe.py`
**Lokalizacja:** Nowa funkcja

**Implementacja:**
```python
def transcribe_with_whisperx(
    wav_path: str,
    model_size: str,
    language: str,
    segment_progress_bar: tqdm,
    timeout_seconds: int,
    align: bool = False,
    diarize: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    hf_token: str = None
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja z WhisperX (GPU/CPU, alignment, diarization).

    WhisperX oferuje:
    - Lepszą dokładność timestampów
    - Word-level alignment
    - Speaker diarization (rozpoznawanie mówców)
    """
    try:
        import whisperx
        import torch
    except ImportError:
        return False, "Błąd: whisperx nie jest zainstalowany. Zainstaluj: pip install whisperx", []

    # Detekcja urządzenia
    device, device_info = detect_device()
    tqdm.write(f"Używane urządzenie: {device_info}")

    # Compute type
    compute_type = "float16" if device == "cuda" else "int8"

    # Ładowanie modelu
    tqdm.write(f"Ładowanie modelu WhisperX {model_size}...")
    try:
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language
        )
    except Exception as e:
        if device == "cuda":
            tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU. Błąd: {e}")
            device = "cpu"
            compute_type = "int8"
            model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
        else:
            return False, f"Błąd podczas ładowania modelu WhisperX: {str(e)}", []

    # Transkrypcja
    OutputManager.stage_header(1, "Transkrypcja")
    tqdm.write(f"\nTranskrypcja WhisperX: {Path(wav_path).name}...")

    # Załaduj audio
    try:
        audio = whisperx.load_audio(str(wav_path))
    except Exception as e:
        return False, f"Błąd podczas ładowania audio: {str(e)}", []

    # Transkrypcja
    try:
        result = model.transcribe(audio, batch_size=16)
    except Exception as e:
        return False, f"Błąd podczas transkrypcji WhisperX: {str(e)}", []

    # Word-level alignment (opcjonalnie)
    if align:
        tqdm.write("Wykonywanie word-level alignment...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
        except Exception as e:
            tqdm.write(f"Ostrzeżenie: Alignment nie powiódł się: {e}")

    # Speaker diarization (opcjonalnie)
    if diarize:
        if not hf_token:
            tqdm.write("Ostrzeżenie: Speaker diarization wymaga HuggingFace token (--hf-token)")
        else:
            tqdm.write("Wykonywanie speaker diarization...")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                tqdm.write(f"Ostrzeżenie: Diarization nie powiódł się: {e}")

    # Konwersja segmentów do formatu (start_ms, end_ms, text)
    segments = []
    for segment in result.get("segments", []):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        text = segment["text"].strip()

        # Dodaj speaker info jeśli dostępne
        if "speaker" in segment:
            text = f"[{segment['speaker']}] {text}"

        segments.append((start_ms, end_ms, text))

        if segment_progress_bar:
            segment_progress_bar.set_postfix_str(f"{len(segments)} segments")

    tqdm.write(f"Wykryty język: {result.get('language', language)}")

    return True, f"Transkrypcja zakończona: {len(segments)} segmentów", segments
```

### 3.3. Dodanie argumentów CLI dla WhisperX

**Plik:** `data/transcribe.py`
**Lokalizacja:** Po linii ~2138 (grupa transcription_group)

```python
# WhisperX advanced options
whisperx_group = parser.add_argument_group('Opcje WhisperX', 'Zaawansowane funkcje dla silnika WhisperX')

whisperx_group.add_argument('--whisperx-align', action='store_true',
                   help='Włącz word-level alignment (dokładniejsze timestampy)')

whisperx_group.add_argument('--whisperx-diarize', action='store_true',
                   help='Włącz speaker diarization (rozpoznawanie mówców)')

whisperx_group.add_argument('--whisperx-min-speakers', type=int, default=None,
                   help='Minimalna liczba mówców (dla diarization)')

whisperx_group.add_argument('--whisperx-max-speakers', type=int, default=None,
                   help='Maksymalna liczba mówców (dla diarization)')

whisperx_group.add_argument('--hf-token', type=str, default=None,
                   help='HuggingFace token (wymagany dla diarization)')
```

---

## 4. KROK 3: Refaktoryzacja funkcji transcribe_chunk()

### 4.1. Cel refaktoryzacji

Wydzielić logikę silników transkrypcji do osobnych funkcji:
- `transcribe_with_whisper()` - OpenAI Whisper (GPU)
- `transcribe_with_faster_whisper()` - Faster-Whisper (CPU)
- `transcribe_with_whisperx()` - WhisperX (GPU/CPU + advanced features)

### 4.2. Nowa implementacja transcribe_chunk()

**Plik:** `data/transcribe.py`
**Lokalizacja:** Linia ~1260 (zastąpienie obecnej funkcji)

```python
def transcribe_chunk(
    wav_path: str,
    model_size: str = "base",
    language: str = "pl",
    engine: str = "whisper",
    segment_progress_bar: tqdm = None,
    timeout_seconds: int = 1800,
    # WhisperX options
    whisperx_align: bool = False,
    whisperx_diarize: bool = False,
    whisperx_min_speakers: int = None,
    whisperx_max_speakers: int = None,
    hf_token: str = None
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """
    Transkrypcja pliku WAV przy użyciu wybranego silnika.

    Args:
        wav_path: Ścieżka do pliku WAV
        model_size: Rozmiar modelu (tiny, base, small, medium, large)
        language: Kod języka (pl, en, etc.)
        engine: Silnik transkrypcji ('whisper', 'faster-whisper', 'whisperx')
        segment_progress_bar: Progress bar dla segmentów
        timeout_seconds: Timeout dla transkrypcji (0 = bez limitu)
        whisperx_align: Włącz word-level alignment (tylko WhisperX)
        whisperx_diarize: Włącz speaker diarization (tylko WhisperX)
        whisperx_min_speakers: Min liczba mówców (tylko WhisperX)
        whisperx_max_speakers: Max liczba mówców (tylko WhisperX)
        hf_token: HuggingFace token (tylko WhisperX diarization)

    Returns:
        Tuple (success: bool, message: str, segments: List[(start_ms, end_ms, text)])
    """
    try:
        # Sprawdź czy plik istnieje
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"Błąd: Plik audio nie istnieje: {wav_path}", []

        # Wybór silnika transkrypcji
        if engine == "whisper":
            return transcribe_with_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds
            )

        elif engine == "faster-whisper":
            return transcribe_with_faster_whisper(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds
            )

        elif engine == "whisperx":
            return transcribe_with_whisperx(
                wav_path, model_size, language,
                segment_progress_bar, timeout_seconds,
                align=whisperx_align,
                diarize=whisperx_diarize,
                min_speakers=whisperx_min_speakers,
                max_speakers=whisperx_max_speakers,
                hf_token=hf_token
            )

        else:
            return False, f"Błąd: Nieobsługiwany silnik transkrypcji: {engine}", []

    except ImportError as e:
        return False, f"Błąd: Brak wymaganej biblioteki: {str(e)}", []
    except Exception as e:
        return False, f"Błąd podczas transkrypcji: {str(e)}", []
```

### 4.3. Przekazywanie parametrów WhisperX

**Plik:** `data/transcribe.py`
**Lokalizacja:** ~Linia 474 (w `_transcribe_all_chunks()`)

```python
success, message, segments = transcribe_chunk(
    chunk_path,
    model_size=args.model,
    language=args.language,
    engine=args.engine,
    segment_progress_bar=segment_pbar,
    timeout_seconds=args.transcription_timeout,
    # WhisperX parameters
    whisperx_align=args.whisperx_align if hasattr(args, 'whisperx_align') else False,
    whisperx_diarize=args.whisperx_diarize if hasattr(args, 'whisperx_diarize') else False,
    whisperx_min_speakers=args.whisperx_min_speakers if hasattr(args, 'whisperx_min_speakers') else None,
    whisperx_max_speakers=args.whisperx_max_speakers if hasattr(args, 'whisperx_max_speakers') else None,
    hf_token=args.hf_token if hasattr(args, 'hf_token') else None
)
```

---

## 5. KROK 4: Optymalizacja Dockerfile

### 5.1. Analiza obecnego obrazu

**Obecny obraz:** `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Rozmiar: ~8-10 GB (devel variant)
- Zawiera: CUDA 12.8, cuDNN 9, narzędzia developerskie

**Problem:**
- CTranslate2 nie jest już wymagane (faster-whisper używa CPU)
- Obraz devel jest zbyt duży (zawiera niepotrzebne narzędzia kompilacji)

### 5.2. Nowy obraz bazowy

**Propozycja 1 (Rekomendowana):** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Rozmiar: ~3-4 GB (runtime variant, 50% mniejszy)
- Zawiera: CUDA 12.4, cuDNN 9, runtime libraries (bez dev tools)
- Kompatybilność: PyTorch cu124, OpenAI Whisper, WhisperX

**Propozycja 2 (Alternatywna):** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Rozmiar: ~2.5-3 GB (jeszcze mniejszy)
- Zawiera: CUDA 12.1, cuDNN 8
- Kompatybilność: PyTorch cu121, OpenAI Whisper
- **UWAGA:** Może wymagać dostosowania wersji PyTorch

### 5.3. Zmieniony Dockerfile

**Plik:** `Dockerfile`

```dockerfile
# ============================================
# DOCKERFILE - Transcription Tool with GPU Support
# ============================================

# ZMIANA: Runtime zamiast devel, CUDA 12.4 zamiast 12.8
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 transcriber \
    && mkdir -p /data /models \
    && chown -R transcriber:transcriber /app /data /models

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ZMIANA: Weryfikacja bibliotek (bez ctranslate2)
RUN python -c "import whisper; print('OpenAI Whisper:', whisper.__version__)" && \
    python -c "import faster_whisper; print('Faster-Whisper:', faster_whisper.__version__)" && \
    python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Copy application code
COPY data/transcribe.py .

# Set environment variables for model cache
ENV HF_HOME=/models
ENV TORCH_HOME=/models
ENV XDG_CACHE_HOME=/models

# Volume mounts
VOLUME ["/data", "/models"]

# Switch to non-root user
USER transcriber

# Default entrypoint
ENTRYPOINT ["python", "transcribe.py"]
CMD ["--help"]
```

### 5.4. Zaktualizowany requirements.txt

**Plik:** `requirements.txt`

```txt
# =============================================================================
# REQUIREMENTS FOR DOCKER (CUDA 12.4 + cuDNN 9 Runtime)
# =============================================================================
# Ten plik jest zoptymalizowany dla obrazu Docker nvidia/cuda:12.4.1-cudnn-runtime
# Dla instalacji natywnej na Windows użyj: requirements-windows.txt
# =============================================================================

# Core dependencies
yt-dlp>=2024.12.01
faster-whisper>=1.1.0
tqdm>=4.67.0
deep-translator>=1.11.4
edge-tts>=7.0.0
openai-whisper>=20240930

# WhisperX - advanced transcription with word-level alignment
whisperx>=3.1.1

# REMOVED: ctranslate2==4.6.0 (nie jest już wymagane, faster-whisper używa CPU)

# =============================================================================
# PyTorch is installed separately in Dockerfile:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# =============================================================================

# =============================================================================
# Kompatybilność z CUDA 12.4 + cuDNN 9 Runtime:
# - faster-whisper >= 1.1.0 (używa CPU, brak CTranslate2)
# - openai-whisper >= 20240930 (PyTorch cu124, GPU)
# - WhisperX >= 3.1.1 (PyTorch cu124, GPU/CPU)
# =============================================================================
```

### 5.5. Porównanie rozmiarów obrazów

| Obraz | Rozmiar bazowy | Rozmiar finalny | Oszczędność |
|-------|---------------|-----------------|-------------|
| **Obecny** (12.8.1-cudnn-devel) | ~8-10 GB | ~12-14 GB | - |
| **Nowy** (12.4.1-cudnn-runtime) | ~3-4 GB | ~6-8 GB | **~50%** |

---

## 6. KROK 5: Plan przyszłościowy - Coqui TTS, Piper TTS

### 6.1. Cel

Dodanie alternatywnych silników TTS (Text-to-Speech) do aplikacji jako rozszerzenie obecnej funkcjonalności Edge TTS.

### 6.2. Silniki TTS do zaimplementowania

#### 6.2.1. Coqui TTS
- **Typ:** Lokalny, wysokiej jakości TTS
- **Wymagania:** TTS>=0.22.0, PyTorch CUDA 12.x
- **Zalety:**
  - Bardzo dobra jakość głosu
  - Wiele modeli wielojęzycznych
  - Lokalne przetwarzanie (offline)
- **Wady:**
  - Wolniejszy niż Edge TTS
  - Wymaga GPU dla najlepszej wydajności
  - Większe zużycie pamięci (modele 100-500 MB)

**Modele polskie:**
- `tts_models/pl/mai_female/vits` - polski głos żeński (najlepsza jakość)
- `tts_models/multilingual/multi-dataset/xtts_v2` - XTTS v2 (multi-language, wymaga GPU)
- `tts_models/multilingual/multi-dataset/your_tts` - Your TTS (szybszy)

#### 6.2.2. Piper TTS
- **Typ:** Lokalny, szybki TTS
- **Wymagania:** piper-tts>=1.2.0 lub binarne Piper
- **Zalety:**
  - Bardzo szybki
  - Małe modele (20-100 MB)
  - Działa offline
  - Niskie zużycie RAM
- **Wady:**
  - Średnia jakość głosu (gorsza niż Coqui/Edge)
  - Mniej modeli dostępnych

**Modele polskie:**
- `pl_PL-darkman-medium` - polski głos męski (domyślny)
- `pl_PL-mls_6892-low` - niższa jakość, szybszy
- `pl_PL-mls_6892-medium` - średnia jakość

### 6.3. Architektura implementacji

**Dodatkowa flaga CLI:**
```python
# Grupa TTS
tts_group.add_argument('--tts-engine', default='edge',
                   choices=['edge', 'coqui', 'piper'],
                   help='Silnik TTS (domyślnie: edge)')

# Coqui-specific
tts_group.add_argument('--coqui-model', default='tts_models/pl/mai_female/vits',
                   help='Model Coqui TTS')
tts_group.add_argument('--coqui-speaker', help='ID mówcy (multi-speaker models)')

# Piper-specific
tts_group.add_argument('--piper-model', default='pl_PL-darkman-medium',
                   help='Model Piper TTS')
tts_group.add_argument('--piper-speaker', type=int, default=0,
                   help='ID mówcy (multi-speaker models)')
```

**Funkcje do zaimplementowania:**
1. `generate_tts_coqui()` - generowanie TTS z Coqui
2. `generate_tts_piper()` - generowanie TTS z Piper
3. `ensure_piper_model()` - automatyczne pobieranie modeli Piper
4. `list_coqui_models()` - lista dostępnych modeli Coqui

### 6.4. Zależności

**requirements.txt (przyszłość):**
```txt
# TTS engines (opcjonalne)
TTS>=0.22.0  # Coqui TTS
piper-tts>=1.2.0  # Piper TTS
```

**Docker - osobne obrazy (opcjonalnie):**
```
Dockerfile.coqui    # Obraz z Coqui TTS
Dockerfile.piper    # Obraz z Piper TTS
```

### 6.5. Porównanie silników TTS

| Silnik | Jakość | Szybkość | Offline | Języki PL | GPU | Rozmiar modelu |
|--------|--------|----------|---------|-----------|-----|----------------|
| **Edge TTS** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | ❌ | 2 głosy | ❌ | 0 MB (cloud) |
| **Coqui TTS** | ⭐⭐⭐⭐⭐ | ⚡⚡ | ✅ | 40+ | Opcjonalnie | 100-500 MB |
| **Piper TTS** | ⭐⭐⭐ | ⚡⚡⚡⚡ | ✅ | 30+ | ❌ | 20-100 MB |

### 6.6. Harmonogram implementacji

**Faza 1 (Priorytet niski):**
- Implementacja Coqui TTS
- Testy z modelami polskimi
- Dokumentacja użytkowania

**Faza 2 (Priorytet niski):**
- Implementacja Piper TTS
- Automatyczne pobieranie modeli
- Porównanie jakości/wydajności

**Faza 3 (Opcjonalna):**
- Dodatkowe modele języków obcych
- Fine-tuning modeli dla lepszej jakości
- Docker images z preinstalowanymi modelami

---

## 7. Podsumowanie zmian w plikach

### 7.1. Pliki do modyfikacji

| Plik | Typ zmiany | Opis |
|------|-----------|------|
| `data/transcribe.py` | **Modyfikacja** | Refaktoryzacja, nowe funkcje, zmiana default engine |
| `requirements.txt` | **Modyfikacja** | Dodanie whisperx, usunięcie ctranslate2 |
| `Dockerfile` | **Modyfikacja** | Zmiana bazowego obrazu, weryfikacja bibliotek |
| `specs/planv5.md` | **Nowy** | Ten dokument |

### 7.2. Szczegółowy changelog

#### data/transcribe.py
**Nowe funkcje:**
- `transcribe_with_whisper()` - linia ~1400 (nowa)
- `transcribe_with_faster_whisper()` - linia ~1500 (nowa)
- `transcribe_with_whisperx()` - linia ~1600 (nowa)

**Modyfikacje:**
- `transcribe_chunk()` - linia ~1260 (refaktoryzacja, uproszczenie)
- `_transcribe_all_chunks()` - linia ~474 (dodanie parametrów WhisperX)
- Argumenty CLI - linia ~2136 (zmiana default, dodanie WhisperX group)

#### requirements.txt
**Dodane:**
- `whisperx>=3.1.1` (linia 16)

**Usunięte:**
- `ctranslate2==4.6.0` (linia 17)

#### Dockerfile
**Zmiany:**
- Linia 5: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (było: 12.8.1-cudnn-devel)
- Linia 46-48: Zmieniona weryfikacja bibliotek (bez ctranslate2)

---

## 8. Testy i weryfikacja

### 8.1. Testy podstawowe

```bash
# Test 1: Domyślny silnik (whisper z GPU)
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=yFeZGU3YA20" --model base

# Test 2: Faster-Whisper (CPU enforced)
docker-compose run --rm transcribe "URL" --engine faster-whisper --model base

# Test 3: WhisperX (podstawowy)
docker-compose run --rm transcribe "URL" --engine whisperx --model base

# Test 4: WhisperX z alignment
docker-compose run --rm transcribe "URL" --engine whisperx --model base --whisperx-align

# Test 5: WhisperX z diarization
docker-compose run --rm transcribe "URL" --engine whisperx --model base \
  --whisperx-diarize --hf-token YOUR_TOKEN --whisperx-min-speakers 2 --whisperx-max-speakers 4
```

### 8.2. Testy rozmiarów obrazów

```bash
# Rebuild z nowym obrazem
docker-compose build --no-cache

# Sprawdź rozmiar
docker images transcription-tool:latest

# Oczekiwany wynik: ~6-8 GB (było: ~12-14 GB)
```

### 8.3. Testy wydajnościowe

**Benchmark:** 10-minutowy film YouTube

| Silnik | Urządzenie | Model | Czas | Jakość timestampów |
|--------|------------|-------|------|-------------------|
| whisper | GPU | base | ~2 min | ⭐⭐⭐ |
| faster-whisper | CPU | base | ~5 min | ⭐⭐⭐ |
| whisperx | GPU | base | ~3 min | ⭐⭐⭐⭐⭐ |
| whisperx (align) | GPU | base | ~4 min | ⭐⭐⭐⭐⭐ |

### 8.4. Kryteria sukcesu

✅ **Implementacja uznana za udaną gdy:**
1. Domyślny silnik to `whisper` z automatycznym GPU
2. Faster-Whisper wymusza CPU (komunikat w logach)
3. WhisperX działa z wszystkimi funkcjami (align, diarize)
4. Obraz Docker jest mniejszy o ~50%
5. Wszystkie istniejące funkcje działają bez zmian
6. Testy podstawowe przechodzą bez błędów

---

## 9. Migracja dla użytkowników

### 9.1. Breaking changes

**UWAGA:** Domyślny silnik zmieniony z `faster-whisper` na `whisper`

**Dotychczasowe użycie:**
```bash
# Przed: domyślnie faster-whisper (CPU)
docker-compose run --rm transcribe "URL"
```

**Nowe użycie (po zmianach):**
```bash
# Po: domyślnie whisper (GPU)
docker-compose run --rm transcribe "URL"

# Aby używać faster-whisper (CPU jak poprzednio):
docker-compose run --rm transcribe "URL" --engine faster-whisper
```

### 9.2. Zalecenia migracyjne

**Dla użytkowników bez GPU:**
```bash
# Dodaj alias do .bashrc lub .zshrc
alias transcribe-cpu='docker-compose run --rm transcribe --engine faster-whisper'

# Użycie
transcribe-cpu "URL"
```

**Dla użytkowników z GPU:**
```bash
# Domyślne wywołanie teraz używa GPU automatycznie
docker-compose run --rm transcribe "URL"
```

---

## 10. Dokumentacja do zaktualizowania

### 10.1. README.md

**Sekcje do dodania/modyfikacji:**

#### Nowa sekcja: "Silniki transkrypcji"

```markdown
## Silniki transkrypcji

Aplikacja obsługuje trzy silniki transkrypcji:

### 1. OpenAI Whisper (domyślny)
- **Użycie:** `--engine whisper`
- **Urządzenie:** Automatycznie GPU/CUDA jeśli dostępne
- **Zalety:** Szybki, dobra jakość, najprostszy w użyciu
- **Wady:** Podstawowe timestampy
```

```bash
# Użycie (domyślnie)
docker-compose run --rm transcribe "URL" --model base
```

### 2. Faster-Whisper (CPU)
- **Użycie:** `--engine faster-whisper`
- **Urządzenie:** Wymuszony CPU
- **Zalety:** Działa bez GPU
- **Wady:** Wolniejszy, wymaga więcej czasu
```

```bash
# Użycie
docker-compose run --rm transcribe "URL" --engine faster-whisper --model base
```

### 3. WhisperX (zaawansowany)
- **Użycie:** `--engine whisperx`
- **Urządzenie:** Automatycznie GPU/CUDA jeśli dostępne
- **Zalety:**
  - Najlepsze timestampy (word-level alignment)
  - Speaker diarization (rozpoznawanie mówców)
  - Wysoką dokładność
- **Wady:** Wolniejszy niż Whisper
```

```bash
# Podstawowe użycie
docker-compose run --rm transcribe "URL" --engine whisperx --model base

# Z word-level alignment
docker-compose run --rm transcribe "URL" --engine whisperx --model base --whisperx-align

# Z speaker diarization
docker-compose run --rm transcribe "URL" --engine whisperx --model base \
  --whisperx-diarize --hf-token YOUR_HF_TOKEN --whisperx-min-speakers 2
```

**Porównanie silników:**

| Silnik | Szybkość | Jakość timestampów | GPU | Diarization |
|--------|----------|-------------------|-----|-------------|
| whisper | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ Auto | ❌ |
| faster-whisper | ⚡⚡ | ⭐⭐⭐ | ❌ CPU only | ❌ |
| whisperx | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ Auto | ✅ |
```

### 10.2. docs/README_FUNCTIONS.md

**Dodać dokumentację nowych funkcji:**

```markdown
### transcribe_with_whisper() → Tuple[bool, str, List[Tuple[int, int, str]]]
Transkrypcja z OpenAI Whisper (GPU/CUDA).

### transcribe_with_faster_whisper() → Tuple[bool, str, List[Tuple[int, int, str]]]
Transkrypcja z Faster-Whisper (wymuszony CPU).

### transcribe_with_whisperx() → Tuple[bool, str, List[Tuple[int, int, str]]]
Transkrypcja z WhisperX (zaawansowane funkcje: alignment, diarization).
```

---

## 11. Kolejność implementacji (Recommended)

### Faza 1: Refaktoryzacja (KROK 1-3)
**Czas:** ~4-6 godzin
1. Wydzielenie funkcji `transcribe_with_whisper()`
2. Wydzielenie funkcji `transcribe_with_faster_whisper()` z wymuszeniem CPU
3. Refaktoryzacja `transcribe_chunk()` jako dispatcher
4. Zmiana domyślnego silnika na `whisper`
5. Testy podstawowe

**Checkpoint 1:** ✅ Refaktoryzacja działa, domyślny silnik zmieniony

### Faza 2: WhisperX (KROK 2)
**Czas:** ~6-8 godzin
1. Dodanie zależności `whisperx>=3.1.1`
2. Implementacja `transcribe_with_whisperx()`
3. Dodanie argumentów CLI (alignment, diarization)
4. Testy WhisperX (podstawowe, align, diarize)

**Checkpoint 2:** ✅ WhisperX działa ze wszystkimi funkcjami

### Faza 3: Optymalizacja Docker (KROK 4)
**Czas:** ~2-3 godziny
1. Zmiana obrazu bazowego na runtime
2. Aktualizacja requirements.txt (usunięcie ctranslate2)
3. Zmiana weryfikacji bibliotek
4. Rebuild i testy

**Checkpoint 3:** ✅ Obraz Docker mniejszy o ~50%, wszystko działa

### Faza 4: Dokumentacja i testy
**Czas:** ~2-3 godziny
1. Aktualizacja README.md
2. Aktualizacja dokumentacji funkcji
3. Testy regresyjne
4. Benchmark wydajności

**Checkpoint 4:** ✅ Pełna dokumentacja, wszystkie testy przechodzą

---

## 12. FAQ i Troubleshooting

### Q1: Dlaczego zmieniono domyślny silnik?
**A:** Whisper (OpenAI) lepiej wykorzystuje GPU i jest szybszy. Faster-Whisper ma problemy z CTranslate2 i GPU, więc został przesunięty do trybu CPU.

### Q2: Co jeśli nie mam GPU?
**A:** Użyj `--engine faster-whisper` lub ustaw `CUDA_VISIBLE_DEVICES=""`.

### Q3: WhisperX wymaga HuggingFace token?
**A:** Tylko dla speaker diarization. Podstawowa transkrypcja i alignment działają bez tokenu.

### Q4: Jak pobrać HuggingFace token?
**A:**
1. Zarejestruj się na https://huggingface.co
2. Settings → Access Tokens → New token
3. Użyj: `--hf-token YOUR_TOKEN`

### Q5: Dlaczego obraz Docker jest mniejszy?
**A:** Zmieniono z `devel` (zawiera kompilatory, dev headers) na `runtime` (tylko biblioteki runtime). Oszczędność ~50%.

### Q6: Czy CTranslate2 jest nadal potrzebne?
**A:** Nie. Faster-Whisper teraz używa CPU, więc CTranslate2 (który wymagał GPU) nie jest potrzebne.

---

## 13. Checklisty implementacyjne

### ✅ KROK 1: Zmiana domyślnych silników
- [ ] Zmienić default engine na `whisper` (linia 2136)
- [ ] Dodać `whisperx` do choices (linia 2137)
- [ ] Stworzyć funkcję `transcribe_with_whisper()`
- [ ] Stworzyć funkcję `transcribe_with_faster_whisper()` z wymuszeniem CPU
- [ ] Test: domyślne wywołanie używa whisper
- [ ] Test: faster-whisper wymusza CPU

### ✅ KROK 2: WhisperX
- [ ] Dodać `whisperx>=3.1.1` do requirements.txt
- [ ] Stworzyć funkcję `transcribe_with_whisperx()`
- [ ] Dodać argumenty CLI (whisperx_group)
- [ ] Test: podstawowa transkrypcja WhisperX
- [ ] Test: alignment
- [ ] Test: diarization (z tokenem)

### ✅ KROK 3: Refaktoryzacja transcribe_chunk()
- [ ] Przepisać transcribe_chunk() jako dispatcher
- [ ] Dodać parametry WhisperX do signature
- [ ] Zaktualizować wywołanie w _transcribe_all_chunks()
- [ ] Test: wszystkie silniki działają przez transcribe_chunk()
- [ ] Test: backward compatibility

### ✅ KROK 4: Optymalizacja Docker
- [ ] Zmienić obraz bazowy na runtime (linia 5)
- [ ] Usunąć ctranslate2 z requirements.txt
- [ ] Zaktualizować weryfikację bibliotek (linia 46-48)
- [ ] Rebuild: `docker-compose build --no-cache`
- [ ] Test: sprawdzić rozmiar obrazu
- [ ] Test: wszystkie funkcje działają

### ✅ KROK 5: Dokumentacja
- [ ] Zaktualizować README.md (sekcja "Silniki transkrypcji")
- [ ] Dodać dokumentację funkcji (README_FUNCTIONS.md)
- [ ] Dodać sekcję migracji
- [ ] Testy regresyjne

---

## 14. Kontakt i pytania

Jeśli masz pytania dotyczące implementacji tego planu:
1. Sprawdź FAQ (sekcja 12)
2. Przejrzyj testy (sekcja 8)
3. Sprawdź dokumentację (sekcja 10)

**Status planu:** ✅ Gotowy do implementacji
**Data utworzenia:** 2025-12-20
**Wersja:** 5.0
**Autor:** Claude Sonnet 4.5 + Arkadiusz
