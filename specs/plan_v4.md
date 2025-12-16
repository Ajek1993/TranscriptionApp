# Plan v4 - Nowe funkcje transkrypcji i TTS

## Przegląd

Dodanie nowych funkcji do projektu transkrypcji:
1. Pobieranie tylko audio z YouTube (bez transkrypcji)
2. WhisperX jako nowy silnik transkrypcji
3. Coqui TTS jako alternatywny silnik TTS
4. Piper TTS jako kolejny silnik TTS

## Krok 1: Pobieranie tylko audio (--download-audio-only)

### Cel
Umożliwienie pobierania audio z YouTube bez wykonywania transkrypcji, podobnie jak istniejąca flaga `--download` dla wideo.

### Implementacja

#### 1.1. Dodanie nowej flagi CLI
- Lokalizacja: `transcribe.py` - sekcja argument parsera
- Dodać w grupie "Podstawowe opcje":
  ```python
  parser.add_argument('--download-audio-only', action='store_true',
                      help='Pobierz tylko audio z YouTube (bez transkrypcji)')
  ```

#### 1.2. Logika pobierania audio
- Lokalizacja: `transcribe.py` - funkcja główna `main()`
- Po walidacji URL YouTube:
  - Sprawdzić flagę `args.download_audio_only`
  - Jeśli ustawiona, wywołać `download_audio_from_youtube()`
  - Zapisać audio jako `{VIDEO_ID}.wav` w bieżącym katalogu
  - Wyświetlić komunikat o sukcesie z lokalizacją pliku
  - Zakończyć program (bez transkrypcji)

#### 1.3. Parametry jakości audio
- Opcjonalnie dodać flagę `--audio-quality` z wartościami:
  - `best` (domyślnie) - najlepsza dostępna jakość
  - `192` - 192 kbps
  - `128` - 128 kbps
  - `96` - 96 kbps

#### 1.4. Aktualizacja dokumentacji
- README.md - dodać sekcję "Pobieranie audio z YouTube"
- Przykłady użycia:
  ```bash
  # Pobierz audio w najlepszej jakości
  python transcribe.py --download-audio-only "https://youtube.com/watch?v=VIDEO_ID"

  # Pobierz z niższą jakością (mniejszy plik)
  python transcribe.py --download-audio-only "URL" --audio-quality 128
  ```

### Zależności
- Brak nowych zależności (wykorzystuje istniejące yt-dlp i ffmpeg)

### Testy
- Test pobierania krótkiego wideo z YouTube
- Test z różnymi jakościami audio
- Weryfikacja formatu wyjściowego (WAV, 16kHz mono)

---

## Krok 2: WhisperX jako nowy engine transkrypcji

### Cel
Dodanie WhisperX jako trzeciego silnika transkrypcji, oferującego lepszą dokładność timestampów i alignment słów.

### Implementacja

#### 2.1. Instalacja zależności
- Dodać do `requirements.txt`:
  ```
  whisperx>=3.1.1
  ```
- Opcjonalne zależności dla alignment:
  ```
  phonemizer>=3.2.1
  ```

#### 2.2. Dodanie opcji CLI
- Rozszerzyć `--engine` o nową wartość:
  ```python
  parser.add_argument('--engine', choices=['faster-whisper', 'whisper', 'whisperx'],
                      default='faster-whisper',
                      help='Silnik transkrypcji (faster-whisper/whisper/whisperx)')
  ```

#### 2.3. Funkcja transkrypcji WhisperX
- Lokalizacja: nowa funkcja `transcribe_with_whisperx()`
- Parametry:
  - `audio_path`: ścieżka do pliku audio
  - `model_size`: rozmiar modelu (tiny/base/small/medium/large)
  - `language`: język transkrypcji
  - `device`: 'cuda' lub 'cpu'
  - `compute_type`: 'float16' dla GPU, 'int8' dla CPU

#### 2.4. Workflow WhisperX
```python
def transcribe_with_whisperx(audio_path, model_size='base', language='pl', device='cpu', compute_type='int8'):
    """
    Transkrypcja z WhisperX

    Etapy:
    1. Ładowanie modelu WhisperX
    2. Transkrypcja audio
    3. Word-level alignment (opcjonalnie)
    4. Diarization - identyfikacja mówców (opcjonalnie)
    5. Zwrócenie segmentów z precyzyjnymi timestampami
    """
    import whisperx

    # Ładowanie modelu
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

    # Transkrypcja
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Word-level alignment (jeśli dostępny model dla języka)
    if args.whisperx_align:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Konwersja do formatu kompatybilnego z resztą kodu
    segments = convert_whisperx_segments(result)

    return segments
```

#### 2.5. Dodatkowe flagi dla WhisperX
- `--whisperx-align`: Włącz word-level alignment
- `--whisperx-diarize`: Włącz speaker diarization (wymaga HuggingFace token)
- `--whisperx-min-speakers`: Minimalna liczba mówców (dla diarization)
- `--whisperx-max-speakers`: Maksymalna liczba mówców (dla diarization)

#### 2.6. Integracja w głównym flow
- Lokalizacja: `main()` - etap 3 (transkrypcja)
- Warunek:
  ```python
  if args.engine == 'whisperx':
      segments = transcribe_with_whisperx(chunk_path, args.model, args.language, device, compute_type)
  elif args.engine == 'faster-whisper':
      # istniejący kod
  elif args.engine == 'whisper':
      # istniejący kod
  ```

#### 2.7. Aktualizacja dokumentacji
- README.md - rozszerzyć sekcję "Silniki transkrypcji"
- Dodać informacje o WhisperX:
  - Zalety: najlepsza dokładność timestampów, word-level alignment, speaker diarization
  - Wady: wolniejszy niż faster-whisper, większe zużycie pamięci
  - Kiedy używać: gdy potrzebna jest najwyższa precyzja timestampów lub identyfikacja mówców

### Zależności
- `whisperx>=3.1.1`
- `phonemizer>=3.2.1` (opcjonalnie, dla alignment)
- `pyannote.audio` (opcjonalnie, dla diarization)

### Testy
- Test transkrypcji krótkiego audio
- Test z word-level alignment
- Test na różnych językach (pl, en)
- Porównanie dokładności timestampów z faster-whisper

---

## Krok 3: Coqui TTS jako alternatywny engine TTS (--tts-engine coqui)

### Cel
Dodanie Coqui TTS jako wysokiej jakości alternatywy dla Microsoft Edge TTS, z możliwością lokalnego generowania mowy.

### Implementacja

#### 3.1. Instalacja zależności
- Dodać do `requirements.txt`:
  ```
  TTS>=0.22.0
  ```

#### 3.2. Dodanie flagi --tts-engine
- Lokalizacja: argument parser - grupa "Dubbing i TTS"
  ```python
  parser.add_argument('--tts-engine', choices=['edge', 'coqui', 'piper'],
                      default='edge',
                      help='Silnik TTS (edge/coqui/piper)')
  ```

#### 3.3. Funkcja generowania TTS z Coqui
- Lokalizacja: nowa funkcja `generate_tts_coqui()`
```python
def generate_tts_coqui(text, output_path, voice_model='tts_models/pl/mai_female/vits',
                       speaker=None, language='pl', speed=1.0):
    """
    Generowanie TTS z Coqui TTS

    Args:
        text: tekst do syntezy
        output_path: ścieżka wyjściowa (WAV)
        voice_model: model Coqui TTS
        speaker: ID mówcy (dla modeli multi-speaker)
        language: język (dla modeli multi-language)
        speed: prędkość mowy (1.0 = normalna)
    """
    from TTS.api import TTS

    # Inicjalizacja TTS (cache model)
    if not hasattr(generate_tts_coqui, 'tts_model'):
        generate_tts_coqui.tts_model = TTS(model_name=voice_model, progress_bar=False)

    tts = generate_tts_coqui.tts_model

    # Generowanie
    if speaker:
        tts.tts_to_file(text=text, file_path=output_path, speaker=speaker, speed=speed)
    elif language:
        tts.tts_to_file(text=text, file_path=output_path, language=language, speed=speed)
    else:
        tts.tts_to_file(text=text, file_path=output_path, speed=speed)
```

#### 3.4. Dostępne modele Coqui dla polskiego
Predefiniowane modele w kodzie:
- `tts_models/pl/mai_female/vits` - polski głos żeński (domyślny)
- `tts_models/multilingual/multi-dataset/your_tts` - multi-language (zawiera polski)
- `tts_models/multilingual/multi-dataset/xtts_v2` - XTTS v2 (najlepsza jakość, wymaga GPU)

#### 3.5. Dodatkowe flagi dla Coqui
```python
parser.add_argument('--coqui-model', default='tts_models/pl/mai_female/vits',
                    help='Model Coqui TTS (domyślnie: polski żeński)')
parser.add_argument('--coqui-speaker', help='ID mówcy (dla modeli multi-speaker)')
parser.add_argument('--coqui-list-models', action='store_true',
                    help='Wyświetl dostępne modele Coqui TTS')
```

#### 3.6. Funkcja listowania modeli
```python
def list_coqui_models():
    """Wyświetl wszystkie dostępne modele Coqui TTS"""
    from TTS.api import TTS
    models = TTS().list_models()

    print("\n=== Dostępne modele Coqui TTS ===\n")

    # Filtruj modele polskie
    polish_models = [m for m in models if '/pl/' in m]
    if polish_models:
        print("Modele polskie:")
        for model in polish_models:
            print(f"  - {model}")

    # Modele wielojęzyczne
    multi_models = [m for m in models if 'multilingual' in m]
    if multi_models:
        print("\nModele wielojęzyczne (zawierają polski):")
        for model in multi_models[:5]:  # Pokaż top 5
            print(f"  - {model}")
```

#### 3.7. Integracja w dubbing workflow
- Lokalizacja: funkcja `generate_dubbing()` lub nowa `generate_dubbing_v2()`
- Dodać warunek:
  ```python
  if args.tts_engine == 'edge':
      # istniejący kod z edge-tts
      await generate_segment_tts_edge(...)
  elif args.tts_engine == 'coqui':
      # nowy kod z Coqui
      generate_tts_coqui(segment_text, segment_path,
                         voice_model=args.coqui_model,
                         speed=speed_factor)
  elif args.tts_engine == 'piper':
      # kod dla Piper (krok 4)
      generate_tts_piper(...)
  ```

#### 3.8. Obsługa przyspieszania TTS
- Coqui obsługuje natywne przyspieszanie przez parametr `speed`
- Wykorzystać istniejącą logikę obliczania `speed_factor` z edge-tts
- Max przyspieszenie: 1.5x (50%)

#### 3.9. Aktualizacja dokumentacji
- README.md - nowa sekcja "Silniki TTS"
- Tabela porównawcza:

| Engine | Jakość | Szybkość | Wymaga internetu | Języki | GPU |
|--------|--------|----------|------------------|--------|-----|
| edge   | Dobra  | Szybka   | Tak              | Wiele  | Nie |
| coqui  | Bardzo dobra | Średnia | Nie | Wiele | Opcjonalnie |
| piper  | Dobra  | Bardzo szybka | Nie | Wiele | Nie |

### Zależności
- `TTS>=0.22.0` (Coqui TTS)
- PyTorch (już wymagany dla Whisper)

### Testy
- Test generowania pojedynczego segmentu
- Test z różnymi modelami polskimi
- Test przyspieszania TTS
- Porównanie jakości z edge-tts

---

## Krok 4: Piper TTS jako engine TTS (--tts-engine piper)

### Cel
Dodanie Piper TTS jako szybkiej i lekkiej alternatywy, działającej offline z małym footprintem pamięci.

### Implementacja

#### 4.1. Instalacja zależności
- Dodać do `requirements.txt`:
  ```
  piper-tts>=1.2.0
  ```
- Alternatywnie: użyć binarnego Piper (bez Python wrapper)

#### 4.2. Funkcja generowania TTS z Piper
```python
def generate_tts_piper(text, output_path, model_path=None, speaker=0, speed=1.0):
    """
    Generowanie TTS z Piper

    Args:
        text: tekst do syntezy
        output_path: ścieżka wyjściowa (WAV)
        model_path: ścieżka do modelu Piper (.onnx)
        speaker: ID mówcy (dla modeli multi-speaker)
        speed: prędkość mowy (0.5-2.0)
    """
    import subprocess
    import json

    # Domyślny model polski (pobierz jeśli nie istnieje)
    if not model_path:
        model_path = ensure_piper_model('pl_PL-darkman-medium')

    config_path = model_path.replace('.onnx', '.onnx.json')

    # Przygotuj tekst (escape)
    text_escaped = text.replace('"', '\\"')

    # Wywołaj Piper przez subprocess
    cmd = [
        'piper',
        '--model', model_path,
        '--config', config_path,
        '--output_file', output_path,
        '--speaker', str(speaker),
        '--length_scale', str(1.0 / speed)  # Piper używa length_scale (1.0 = normalnie)
    ]

    # Przekaż tekst przez stdin
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=text_escaped)

    if process.returncode != 0:
        raise RuntimeError(f"Piper TTS error: {stderr}")
```

#### 4.3. Pobieranie modeli Piper
```python
def ensure_piper_model(model_name='pl_PL-darkman-medium'):
    """
    Sprawdź i pobierz model Piper jeśli nie istnieje

    Modele polskie Piper:
    - pl_PL-darkman-medium (męski, średnia jakość)
    - pl_PL-mls_6892-low (niższa jakość, szybszy)
    """
    import urllib.request
    from pathlib import Path

    models_dir = Path.home() / '.local' / 'share' / 'piper' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f'{model_name}.onnx'
    config_path = models_dir / f'{model_name}.onnx.json'

    # Sprawdź czy model istnieje
    if model_path.exists() and config_path.exists():
        return str(model_path)

    # Pobierz model
    print(f"Pobieranie modelu Piper: {model_name}")
    base_url = f'https://huggingface.co/rhasspy/piper-voices/resolve/main/{model_name.replace("-", "/")}'

    urllib.request.urlretrieve(f'{base_url}/{model_name}.onnx', model_path)
    urllib.request.urlretrieve(f'{base_url}/{model_name}.onnx.json', config_path)

    print(f"Model pobrano do: {model_path}")
    return str(model_path)
```

#### 4.4. Dodatkowe flagi dla Piper
```python
parser.add_argument('--piper-model', default='pl_PL-darkman-medium',
                    help='Model Piper TTS (domyślnie: polski męski)')
parser.add_argument('--piper-speaker', type=int, default=0,
                    help='ID mówcy dla modeli multi-speaker (domyślnie: 0)')
```

#### 4.5. Dostępne modele Piper dla polskiego
Dodać do dokumentacji:
- `pl_PL-darkman-medium` - męski, średnia jakość (domyślny)
- `pl_PL-mls_6892-low` - niższa jakość, szybszy

#### 4.6. Integracja w dubbing workflow
- Dodać do warunku w `generate_dubbing_v2()`:
  ```python
  elif args.tts_engine == 'piper':
      generate_tts_piper(segment_text, segment_path,
                         model_path=None if not args.piper_model else ensure_piper_model(args.piper_model),
                         speaker=args.piper_speaker,
                         speed=speed_factor)
  ```

#### 4.7. Obsługa offline
- Piper działa w 100% offline po pobraniu modelu
- Modele są małe (20-100 MB)
- Komunikat przy pierwszym użyciu: "Pobieranie modelu Piper (jednorazowo)..."

#### 4.8. Aktualizacja dokumentacji
- README.md - rozszerzyć sekcję "Silniki TTS"
- Dodać przykłady:
  ```bash
  # Dubbing z Coqui (lepsza jakość, wolniejsze)
  python transcribe.py --local "film.mp4" --dub --tts-engine coqui

  # Dubbing z Piper (szybszy, offline)
  python transcribe.py --local "film.mp4" --dub --tts-engine piper

  # Coqui z własnym modelem
  python transcribe.py "URL" --dub --tts-engine coqui --coqui-model tts_models/multilingual/multi-dataset/xtts_v2

  # Piper z niższą jakością (szybsze)
  python transcribe.py "URL" --dub --tts-engine piper --piper-model pl_PL-mls_6892-low
  ```

### Zależności
- `piper-tts>=1.2.0` (Python wrapper)
- Lub binarne Piper: https://github.com/rhasspy/piper/releases

### Testy
- Test pobierania modelu przy pierwszym użyciu
- Test generowania TTS offline
- Test różnych modeli polskich
- Porównanie szybkości z edge-tts i coqui

---

## Podsumowanie zmian

### Nowe flagi CLI

#### Pobieranie audio
```
--download-audio-only     Pobierz tylko audio z YouTube (bez transkrypcji)
--audio-quality QUALITY   Jakość audio (best/192/128/96)
```

#### WhisperX
```
--engine whisperx         Użyj WhisperX jako silnika transkrypcji
--whisperx-align          Włącz word-level alignment
--whisperx-diarize        Włącz speaker diarization
--whisperx-min-speakers N Minimalna liczba mówców
--whisperx-max-speakers N Maksymalna liczba mówców
```

#### Silniki TTS
```
--tts-engine ENGINE       Silnik TTS (edge/coqui/piper)

# Coqui-specific
--coqui-model MODEL       Model Coqui TTS
--coqui-speaker ID        ID mówcy (multi-speaker models)
--coqui-list-models       Lista dostępnych modeli Coqui

# Piper-specific
--piper-model MODEL       Model Piper TTS
--piper-speaker ID        ID mówcy (multi-speaker models)
```

### Nowe zależności w requirements.txt
```
# Krok 2
whisperx>=3.1.1
phonemizer>=3.2.1

# Krok 3
TTS>=0.22.0

# Krok 4
piper-tts>=1.2.0
```

### Kolejność implementacji
1. **Krok 1** (najprostszy): Pobieranie audio - ~2-3 godziny
2. **Krok 2** (średni): WhisperX - ~4-6 godzin
3. **Krok 3** (średni): Coqui TTS - ~4-6 godzin
4. **Krok 4** (średni): Piper TTS - ~4-6 godzin

### Struktura plików po zmianach
```
transcribe.py                 # Główny plik (rozszerzony)
├── Sekcja 1: Imports         # Dodać: whisperx, TTS, piper
├── Sekcja 2: Funkcje TTS
│   ├── generate_tts_edge()   # Istniejące
│   ├── generate_tts_coqui()  # NOWE - Krok 3
│   └── generate_tts_piper()  # NOWE - Krok 4
├── Sekcja 3: Funkcje transkrypcji
│   ├── transcribe_with_faster_whisper()  # Istniejące
│   ├── transcribe_with_whisper()         # Istniejące
│   └── transcribe_with_whisperx()        # NOWE - Krok 2
├── Sekcja 4: Funkcje pomocnicze
│   ├── ensure_piper_model()  # NOWE - Krok 4
│   └── list_coqui_models()   # NOWE - Krok 3
└── Sekcja 5: Main workflow   # Rozszerzone warunki

requirements.txt              # Rozszerzony
README.md                     # Aktualizowany w każdym kroku
specs/plan_v4.md             # Ten plik
```

### Kompatybilność wsteczna
- ✅ Wszystkie istniejące flagi działają bez zmian
- ✅ Domyślne wartości zachowane (`--engine faster-whisper`, `--tts-engine edge`)
- ✅ Brak breaking changes w API

### Testy integracyjne (po wszystkich krokach)
```bash
# Test 1: Audio download
python transcribe.py --download-audio-only "URL"

# Test 2: WhisperX transkrypcja
python transcribe.py --local "audio.wav" --engine whisperx

# Test 3: Coqui dubbing
python transcribe.py --local "video.mp4" --dub --tts-engine coqui

# Test 4: Piper dubbing (offline)
python transcribe.py --local "video.mp4" --dub --tts-engine piper

# Test 5: Pełny workflow z WhisperX + Coqui
python transcribe.py "URL" --engine whisperx --whisperx-align --dub --tts-engine coqui

# Test 6: Porównanie silników TTS
python transcribe.py --local "test.mp4" --dub --tts-engine edge -o test_edge.mp4
python transcribe.py --local "test.mp4" --dub --tts-engine coqui -o test_coqui.mp4
python transcribe.py --local "test.mp4" --dub --tts-engine piper -o test_piper.mp4
```

---

## Załączniki

### A. Porównanie silników transkrypcji

| Silnik         | Szybkość | Dokładność | Timestamps | Word-level | Diarization | Pamięć |
|----------------|----------|------------|------------|------------|-------------|--------|
| faster-whisper | ⚡⚡⚡    | ⭐⭐⭐     | ⭐⭐       | ❌         | ❌          | 💾💾   |
| whisper        | ⚡⚡      | ⭐⭐⭐⭐   | ⭐⭐       | ❌         | ❌          | 💾💾💾 |
| whisperx       | ⚡⚡      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅         | ✅          | 💾💾💾 |

### B. Porównanie silników TTS

| Silnik | Jakość | Szybkość | Offline | Języki | Głosy | Rozmiar |
|--------|--------|----------|---------|--------|-------|---------|
| edge   | ⭐⭐⭐ | ⚡⚡⚡   | ❌      | 70+    | 200+  | 0 MB    |
| coqui  | ⭐⭐⭐⭐⭐ | ⚡⚡   | ✅      | 40+    | 100+  | 100-500 MB |
| piper  | ⭐⭐⭐ | ⚡⚡⚡⚡ | ✅      | 30+    | 50+   | 20-100 MB |

### C. Przykładowe modele Coqui TTS dla polskiego

```python
COQUI_POLISH_MODELS = {
    'mai_female': 'tts_models/pl/mai_female/vits',  # Najlepsza jakość, kobieta
    'multilingual_xtts': 'tts_models/multilingual/multi-dataset/xtts_v2',  # Multi-language, najlepsza jakość ogólna
    'your_tts': 'tts_models/multilingual/multi-dataset/your_tts',  # Multi-language, szybszy
}
```

### D. Przykładowe modele Piper TTS dla polskiego

```python
PIPER_POLISH_MODELS = {
    'darkman_medium': 'pl_PL-darkman-medium',  # Męski, średnia jakość (domyślny)
    'mls_low': 'pl_PL-mls_6892-low',  # Niższa jakość, szybszy
}
```

---

**Status:** Gotowy do implementacji
**Priorytet:** Średni
**Estimated effort:** ~16-20 godzin (wszystkie 4 kroki)
