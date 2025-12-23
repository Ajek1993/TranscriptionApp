# Transcription Tool - YouTube & Local Files to SRT

Narzędzie do automatycznej transkrypcji filmów z YouTube lub lokalnych plików wideo do formatu napisów SRT przy użyciu modelu Whisper, z wbudowanym wsparciem dla tłumaczenia i dubbingu TTS.

## Opis

Aplikacja pobiera audio z YouTube lub tworzy je z lokalnych plików wideo, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelu Whisper. Wspiera trzy silniki transkrypcji: OpenAI Whisper (domyślny, GPU), Faster-Whisper (CPU) i WhisperX (zaawansowany z alignmentem i diaryzacją). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty oraz opcjonalne tłumaczenie między polskim a angielskim. Nowa funkcjonalność dubbingu TTS pozwala na wygenerowanie polskiej ścieżki audio z synchronizacją czasową i mixowaniem z oryginalnym dźwiękiem.

## Architektura modularna

Kod aplikacji został zorganizowany w modularną architekturę:

**Główny skrypt** (katalog główny):
- **transcribe.py** - Główny orchestrator CLI, parsowanie argumentów i orchestracja pipeline'ów

**Moduły pomocnicze** (katalog data/):
- **output_manager.py** - Klasa OutputManager do formatowania komunikatów (stage headers, info, success, warnings)
- **command_builders.py** - Budowanie komend FFmpeg i yt-dlp (audio/video extraction, splitting, merging, SRT/ASS support)
- **validators.py** - Walidacja URL/plików, sprawdzanie zależności (ffmpeg, yt-dlp, TTS engines)
- **youtube_processor.py** - Pobieranie audio/wideo z YouTube, ekstrakcja audio z plików wideo
- **audio_processor.py** - Operacje na plikach audio (duration, splitting chunks)
- **device_manager.py** - Detekcja dostępności GPU/CPU, pamięć GPU
- **transcription_engines.py** - Implementacja 3 silników transkrypcji (Whisper, Faster-Whisper, WhisperX)
- **segment_processor.py** - Dzielenie długich segmentów, wypełnianie luk, formatowanie timestampów SRT
- **translation.py** - Tłumaczenie segmentów napisów (Google Translate via deep-translator)
- **srt_writer.py** - Generowanie plików SRT ze standardowym formatowaniem
- **ass_writer.py** - Generowanie plików ASS z dwujęzycznymi napisami (oryginał + tłumaczenie)
- **tts_generator.py** - Generowanie dubbingu TTS (Edge TTS + Coqui TTS), synchronizacja audio
- **audio_mixer.py** - Miksowanie ścieżek audio, tworzenie wideo z dubbingiem, wgrywanie napisów (SRT/ASS)
- **warning_suppressor.py** - System tłumienia ostrzeżeń bibliotek trzecich (TensorFlow, PyTorch, tokenizers)
- **utils.py** - Czyszczenie plików tymczasowych

Wszystkie moduły są zorganizowane jako acykliczny graf zależności (DAG), co zapewnia czytelność kodu i łatwość w rozwijaniu aplikacji.

## Funkcje

- **Transkrypcja z YouTube**: Pobieranie audio z YouTube w formacie WAV
- **Pobieranie wideo z YouTube**: Pobieranie pełnego wideo w jakości 720p-4K
- **Transkrypcja z plików lokalnych**: Obsługa MP4, MKV, AVI, MOV
- **Dubbing TTS**: Generowanie dubbingu z Microsoft Edge TTS (polskie i angielskie głosy)
- **Wgrywanie napisów do wideo**: Hardcode subtitles z customizacją stylu (SRT lub ASS)
- **Napisy dwujęzyczne (ASS)**: Wgrywanie napisów z jednoczesnym wyświetlaniem oryginału i tłumaczenia
- **Wybór silnika transkrypcji**: OpenAI Whisper (domyślny), Faster-Whisper (CPU), WhisperX (zaawansowany)
- **Docker**: Pełna dockeryzacja z CUDA 12.4 + cuDNN 9 i GPU/CPU fallback
- **Architektura modularna**: 15 wyspecjalizowanych modułów dla lepszej organizacji kodu
- Automatyczny podział długich nagrań na fragmenty (~30 minut)
- Transkrypcja z wykorzystaniem modelu Whisper (pl, en, i inne języki)
- Tłumaczenie napisów: polski ↔ angielski (deep-translator + Google Translate)
- Zaawansowane opcje narratora: kontrola segmentów, wypełnianie luk, pauzy
- Wsparcie dla GPU (CUDA) i CPU
- Generowanie plików SRT i ASS zgodnych ze standardem
- Automatyczne czyszczenie plików tymczasowych
- Wsparcie dla wielu języków transkrypcji

## Dokumentacja deweloperska

- **[Dokumentacja wszystkich funkcji](README_FUNCTIONS.md)** - Szczegółowy opis wszystkich funkcji z kodu źródłowego, parametrów, wartości zwracanych i przykładów użycia

## Docker

Aplikacja jest w pełni zdockeryzowana z automatycznym wsparciem GPU (CUDA 12.4 + cuDNN 9) i fallback na CPU.

### Szybki start

```bash
# 1. Zbuduj obraz
docker-compose build

# 2. Test GPU (opcjonalnie)
docker-compose run --rm transcribe python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 3. Użycie
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Struktura katalogów Docker

```
PROJEKT_TRANSKRYPCJA/
├── transcribe.py                # Główny orchestrator CLI (punkt wejścia)
├── data/                        # Moduły pomocnicze i pliki wyjściowe
│   ├── output_manager.py        # OutputManager class (formatowanie komunikatów)
│   ├── command_builders.py      # Budowanie komend FFmpeg/yt-dlp (SRT/ASS support)
│   ├── validators.py            # Walidacja URL/plików/zależności
│   ├── youtube_processor.py     # Pobieranie z YouTube + ekstrakcja audio
│   ├── audio_processor.py       # Operacje audio (duration, split)
│   ├── device_manager.py        # Detekcja GPU/CPU
│   ├── transcription_engines.py # Silniki transkrypcji (Whisper/Faster/WhisperX)
│   ├── segment_processor.py     # Dzielenie segmentów, timestampy
│   ├── translation.py           # Tłumaczenie napisów (Google Translate)
│   ├── srt_writer.py            # Generowanie plików SRT
│   ├── ass_writer.py            # Generowanie plików ASS (dual-language)
│   ├── tts_generator.py         # Edge TTS + Coqui TTS (dubbing)
│   ├── audio_mixer.py           # Miksowanie audio, wgrywanie napisów (SRT/ASS)
│   ├── warning_suppressor.py    # Tłumienie ostrzeżeń bibliotek trzecich
│   ├── utils.py                 # Czyszczenie plików tymczasowych
│   └── files/                   # Pliki wyjściowe (SRT, MP4, WAV)
├── docs/                        # Dokumentacja
│   ├── README_FUNCTIONS.md      # Dokumentacja funkcji
│   └── README_archive.md        # Archiwum README
├── specs/                       # Plany i specyfikacje
│   ├── refactor.md              # Plan refaktoryzacji (15 modułów)
│   └── plan_docker.md           # Plan dockeryzacji
├── Dockerfile                   # Definicja obrazu Docker
├── docker-compose.yml           # Konfiguracja Docker Compose
├── .dockerignore                # Pliki ignorowane przez Docker
├── requirements.txt             # Zależności Python (Docker)
├── requirements-windows.txt     # Zależności Python (Windows)
└── README.md                    # Ten plik
```


### Wymagania Docker

**Windows (Docker Desktop + WSL2):**

1. Windows 10/11 z WSL2
2. NVIDIA GPU Driver >= 550.54 (dla CUDA 12.4)
3. Docker Desktop z włączonym WSL2 backend

**Weryfikacja:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Linux:**

1. NVIDIA GPU Driver >= 550.54
2. Docker Engine
3. NVIDIA Container Toolkit

### GPU/CPU Fallback

Obraz automatycznie wykrywa dostępność GPU:

- Jeśli GPU dostępne → używa CUDA z float16
- Jeśli brak GPU → automatyczny fallback na CPU z int8

## Instalacja natywna (bez Docker)

### Wymagane narzędzia

1. **Python 3.7+**
2. **ffmpeg** - do przetwarzania audio
   - Windows: `winget install FFmpeg` lub `choco install ffmpeg`
3. **yt-dlp** - do pobierania z YouTube
   - Instalacja: `pip install yt-dlp`

### Instalacja

```bash
pip install -r requirements.txt
```

Dla GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Przykłady użycia - Wszystkie funkcje

### 1. Podstawowa transkrypcja

**YouTube do SRT:**
```bash
# Docker
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=VIDEO_ID"

# Natywnie
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Lokalny plik do SRT:**
```bash
# Docker (plik musi być w ./data/)
docker-compose run --rm transcribe --local /data/video.mp4

# Natywnie
python transcribe.py --local "video.mp4"
```

### 2. Pobieranie bez transkrypcji

**Pobierz tylko wideo:**
```bash
# Docker
docker-compose run --rm transcribe --download "URL" --video-quality 1080

# Natywnie
python transcribe.py --download "URL" --video-quality 1080
# Dostępne jakości: 720, 1080, 1440, 2160
```

**Pobierz tylko audio (WAV):**
```bash
# Docker
docker-compose run --rm transcribe --download-audio-only "URL"

# Natywnie
python transcribe.py --download-audio-only "URL"
```

### 3. Transkrypcja z tłumaczeniem

**Angielski film → polskie napisy:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --language en --translate en-pl

# Natywnie
python transcribe.py "URL" --language en --translate en-pl
```

**Polski film → angielskie napisy:**
```bash
# Docker
docker-compose run --rm transcribe --local /data/film.mp4 --language pl --translate pl-en

# Natywnie
python transcribe.py --local "film.mp4" --language pl --translate pl-en
```

### 4. Dubbing TTS (Polish Voice-Over)

**YouTube z polskim dubbingiem:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --dub

# Natywnie
python transcribe.py "URL" --dub
# Wynik: VIDEO_ID.srt + VIDEO_ID_dubbed.mp4
```

**Lokalny plik z dubbingiem:**
```bash
# Docker
docker-compose run --rm transcribe --local /data/video.mp4 --dub

# Natywnie
python transcribe.py --local "video.mp4" --dub
# Wynik: video.srt + video_dubbed.mp4
```

**Tylko audio dubbing (bez wideo):**
```bash
# Docker
docker-compose run --rm transcribe "URL" --dub-audio-only

# Natywnie
python transcribe.py "URL" --dub-audio-only
# Wynik: VIDEO_ID.srt + VIDEO_ID_dubbed.wav
```

**Wybór głosu TTS:**
```bash
# Docker - męski głos (domyślny)
docker-compose run --rm transcribe "URL" --dub --tts-voice pl-PL-MarekNeural

# Docker - żeński głos
docker-compose run --rm transcribe "URL" --dub --tts-voice pl-PL-ZofiaNeural

# Natywnie
python transcribe.py "URL" --dub --tts-voice pl-PL-ZofiaNeural
```

**Kontrola głośności:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --dub \
  --original-volume 0.1 \
  --tts-volume 1.2

# Natywnie
python transcribe.py "URL" --dub --original-volume 0.1 --tts-volume 1.2
# original-volume: 0.0-1.0 (domyślnie 0.2 = 20%)
# tts-volume: 0.0-2.0 (domyślnie 1.0 = 100%)
```

### 5. Angielski film → Polski dubbing (pełny workflow)

```bash
# Docker
docker-compose run --rm transcribe "URL" \
  --language en \
  --translate en-pl \
  --dub \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.15

# Natywnie
python transcribe.py "URL" \
  --language en \
  --translate en-pl \
  --dub \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.15
# Wynik: polskie napisy + wideo z polskim lektorem
```

### 6. Wgrywanie napisów do wideo (Hardcode Subtitles)

**Lokalny plik z napisami:**
```bash
# Docker
docker-compose run --rm transcribe --local /data/film.mp4 --burn-subtitles

# Natywnie
python transcribe.py --local "film.mp4" --burn-subtitles
# Wynik: film.srt + film_subtitled.mp4
```

**YouTube z napisami:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --burn-subtitles

# Natywnie
python transcribe.py "URL" --burn-subtitles
```

**Własny styl napisów:**
```bash
# Docker - żółte napisy, większa czcionka
docker-compose run --rm transcribe --local /data/film.mp4 --burn-subtitles \
  --subtitle-style "FontName=Arial,FontSize=28,PrimaryColour=&H0000FFFF,BackColour=&H80000000"

# Natywnie
python transcribe.py --local "film.mp4" --burn-subtitles \
  --subtitle-style "FontName=Arial,FontSize=28,PrimaryColour=&H0000FFFF"
```

### 7. Napisy dwujęzyczne (Dual-Language Subtitles)

**Angielski film → Napisy angielskie + polskie jednocześnie:**
```bash
# Docker
docker-compose run --rm transcribe "URL" \
  --language en \
  --translate en-pl \
  --burn-subtitles \
  --dual-language

# Natywnie
python transcribe.py "URL" \
  --language en \
  --translate en-pl \
  --burn-subtitles \
  --dual-language
# Wynik: VIDEO_ID_subtitled.mp4 z napisami ASS (żółty angielski u góry + biały polski na dole)
```

**Polski film → Napisy polskie + angielskie:**
```bash
# Docker
docker-compose run --rm transcribe --local /data/film.mp4 \
  --language pl \
  --translate pl-en \
  --burn-subtitles \
  --dual-language

# Natywnie
python transcribe.py --local "film.mp4" \
  --language pl \
  --translate pl-en \
  --burn-subtitles \
  --dual-language
# Wynik: film_subtitled.mp4 z dwujęzycznymi napisami
```

**Uwaga:** Flaga `--dual-language` wymaga użycia `--translate` i `--burn-subtitles`. Generuje plik ASS z:
- Oryginalnymi napisami (żółty tekst, u góry ekranu)
- Przetłumaczonymi napisami (biały tekst, na dole ekranu)

### 8. Wybór modelu Whisper

```bash
# Docker - szybki model (najszybszy, mniej dokładny)
docker-compose run --rm transcribe "URL" --model tiny

# Docker - domyślny
docker-compose run --rm transcribe "URL" --model base

# Docker - dokładniejszy (wolniejszy)
docker-compose run --rm transcribe "URL" --model medium

# Docker - najdokładniejszy (bardzo wolny)
docker-compose run --rm transcribe "URL" --model large

# Natywnie
python transcribe.py "URL" --model medium
```

## Silniki transkrypcji

Aplikacja obsługuje trzy silniki transkrypcji:

### 1. OpenAI Whisper (domyślny)
- **Użycie:** `--engine whisper`
- **Urządzenie:** Automatycznie GPU/CUDA jeśli dostępne
- **Zalety:** Szybki, dobra jakość, najprostszy w użyciu
- **Wady:** Podstawowe timestampy

**Użycie (domyślnie):**
```bash
# Docker
docker-compose run --rm transcribe "URL" --model base

# Natywnie
python transcribe.py "URL" --model base
```

### 2. Faster-Whisper (CPU)
- **Użycie:** `--engine faster-whisper`
- **Urządzenie:** Wymuszony CPU
- **Zalety:** Działa bez GPU
- **Wady:** Wolniejszy, wymaga więcej czasu

**Użycie:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --engine faster-whisper --model base

# Natywnie
python transcribe.py "URL" --engine faster-whisper --model base
```

### 3. WhisperX (zaawansowany)
- **Użycie:** `--engine whisperx`
- **Urządzenie:** Automatycznie GPU/CUDA jeśli dostępne
- **Zalety:**
  - Najlepsze timestampy (word-level alignment)
  - Speaker diarization (rozpoznawanie mówców)
  - Wysoką dokładność
- **Wady:** Wolniejszy niż Whisper

**Podstawowe użycie:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --engine whisperx --model base

# Natywnie
python transcribe.py "URL" --engine whisperx --model base
```

**Z word-level alignment:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --engine whisperx --model base --whisperx-align

# Natywnie
python transcribe.py "URL" --engine whisperx --model base --whisperx-align
```

**Z speaker diarization:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --engine whisperx --model base \
  --whisperx-diarize --hf-token YOUR_HF_TOKEN --whisperx-min-speakers 2

# Natywnie
python transcribe.py "URL" --engine whisperx --model base \
  --whisperx-diarize --hf-token YOUR_HF_TOKEN --whisperx-min-speakers 2
```

**Porównanie silników:**

| Silnik | Szybkość | Jakość timestampów | GPU | Diarization |
|--------|----------|-------------------|-----|-------------|
| whisper | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ Auto | ❌ |
| faster-whisper | ⚡⚡ | ⭐⭐⭐ | ❌ CPU only | ❌ |
| whisperx | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ Auto | ✅ |

## Silniki TTS (Text-to-Speech)

Aplikacja obsługuje dwa silniki TTS do generowania dubbingu:

### 1. Edge TTS (domyślny)
- **Użycie:** `--tts-engine edge`
- **Typ:** Cloudowy (Microsoft Azure)
- **Zalety:**
  - Bardzo szybki (online)
  - Darmowy
  - Nie wymaga GPU
  - Wysoka jakość głosów
- **Wady:** Wymaga połączenia internetowego

**Dostępne języki:**
- **Polski:** pl-PL-MarekNeural (męski), pl-PL-ZofiaNeural (żeński)
- **Angielski:** en-US-GuyNeural, en-US-JennyNeural, en-GB-RyanNeural, en-GB-SoniaNeural, en-AU-WilliamNeural, en-AU-NatashaNeural
- **Niemiecki:** de-DE-ConradNeural (męski), de-DE-KatjaNeural (żeński)
- **Francuski:** fr-FR-HenriNeural (męski), fr-FR-DeniseNeural (żeński)
- **Hiszpański:** es-ES-AlvaroNeural (męski), es-ES-ElviraNeural (żeński)
- **Włoski:** it-IT-DiegoNeural (męski), it-IT-ElsaNeural (żeński)
- **Rosyjski:** ru-RU-DmitryNeural (męski), ru-RU-SvetlanaNeural (żeński)
- **Japoński:** ja-JP-KeitaNeural (męski), ja-JP-NanamiNeural (żeński)
- **Chiński:** zh-CN-YunxiNeural (męski), zh-CN-XiaoxiaoNeural (żeński)
- **Koreański:** ko-KR-InJoonNeural (męski), ko-KR-SunHiNeural (żeński)
- **Ukraiński:** uk-UA-OstapNeural (męski), uk-UA-PolinaNeural (żeński)
- **Czeski:** cs-CZ-AntoninNeural (męski), cs-CZ-VlastaNeural (żeński)

**Użycie:**
```bash
# Docker - domyślny (Edge TTS)
docker-compose run --rm transcribe "URL" --dub

# Wybór głosu
docker-compose run --rm transcribe "URL" --dub --tts-voice pl-PL-ZofiaNeural

# Natywnie
python transcribe.py "URL" --dub --tts-voice de-DE-KatjaNeural
```

### 2. Coqui TTS (lokalny)
- **Użycie:** `--tts-engine coqui`
- **Typ:** Lokalny (offline)
- **Zalety:**
  - Bardzo wysoka jakość głosu
  - Działa offline
  - Pełna kontrola nad modelem
  - Wiele modeli wielojęzycznych
- **Wady:**
  - Wolniejszy niż Edge TTS
  - Wymaga GPU dla najlepszej wydajności
  - Większe zużycie pamięci (modele 100-500 MB)

**Dostępne modele polskie:**
- `tts_models/pl/mai_female/vits` - polski głos żeński (domyślny, najlepsza jakość)
- `tts_models/multilingual/multi-dataset/xtts_v2` - XTTS v2 (multi-language, wymaga GPU)
- `tts_models/multilingual/multi-dataset/your_tts` - Your TTS (szybszy)

**Użycie:**
```bash
# Docker - Coqui TTS z domyślnym modelem polskim
docker-compose run --rm transcribe "URL" --dub --tts-engine coqui

# Wybór innego modelu
docker-compose run --rm transcribe "URL" --dub --tts-engine coqui \
  --coqui-model tts_models/multilingual/multi-dataset/xtts_v2

# Multi-speaker model (jeśli model obsługuje)
docker-compose run --rm transcribe "URL" --dub --tts-engine coqui \
  --coqui-model tts_models/multilingual/multi-dataset/xtts_v2 \
  --coqui-speaker "speaker_01"

# Natywnie
python transcribe.py "URL" --dub --tts-engine coqui --coqui-model tts_models/pl/mai_female/vits
```

**Porównanie silników TTS:**

| Silnik | Jakość | Szybkość | Offline | Języki PL | GPU | Rozmiar modelu |
|--------|--------|----------|---------|-----------|-----|----------------|
| **Edge TTS** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | ❌ | 2 głosy | ❌ | 0 MB (cloud) |
| **Coqui TTS** | ⭐⭐⭐⭐⭐ | ⚡⚡ | ✅ | 40+ | Opcjonalnie | 100-500 MB |

### 9. Starsze opcje silnika transkrypcji

```bash
# Docker - openai-whisper (domyślny)
docker-compose run --rm transcribe "URL" --engine whisper

# Docker - faster-whisper (CPU fallback)
docker-compose run --rm transcribe "URL" --engine faster-whisper

# Natywnie
python transcribe.py "URL" --engine whisper
```

### 10. Wymuszone CPU (bez GPU)

```bash
# Docker - wyłącz GPU
docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe "URL"

# Przydatne gdy GPU jest zajęte lub ma problemy
```

### 11. Zaawansowane opcje dubbingu

**Kontrola segmentacji narratora:**
```bash
# Docker
docker-compose run --rm transcribe "URL" --dub \
  --max-segment-duration 8 \
  --max-segment-words 12 \
  --fill-gaps

# Natywnie
python transcribe.py "URL" --dub \
  --max-segment-duration 8 \
  --max-segment-words 12 \
  --fill-gaps
# Przydatne dla szybkiej mowy lub dialogów
```

### 12. Własne nazwy plików wyjściowych

```bash
# Docker - własna nazwa SRT
docker-compose run --rm transcribe "URL" -o moja_transkrypcja.srt

# Docker - własna nazwa dubbingu
docker-compose run --rm transcribe "URL" --dub --dub-output moj_dubbing.mp4

# Docker - własna nazwa subtitle burn
docker-compose run --rm transcribe --local /data/film.mp4 --burn-subtitles \
  --burn-output film_z_napisami.mp4

# Natywnie
python transcribe.py "URL" -o custom.srt
python transcribe.py "URL" --dub --dub-output dubbed.mp4
python transcribe.py --local "film.mp4" --burn-subtitles --burn-output output.mp4
```

### 13. Pomoc i debugowanie

```bash
# Docker - wyświetl wszystkie opcje
docker-compose run --rm transcribe --help

# Docker - interaktywny shell (debugowanie)
docker-compose run --rm --entrypoint bash transcribe

# Docker - test GPU
docker-compose run --rm transcribe python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## Szybkie Podsumowanie - Docker Compose

| Funkcja | Komenda |
|---------|---------|
| **Podstawowa transkrypcja** | `docker-compose run --rm transcribe "URL"` |
| **Lokalny plik** | `docker-compose run --rm transcribe --local /data/video.mp4` |
| **Pobierz wideo** | `docker-compose run --rm transcribe --download "URL"` |
| **Pobierz audio** | `docker-compose run --rm transcribe --download-audio-only "URL"` |
| **Tłumaczenie** | `docker-compose run --rm transcribe "URL" --language en --translate en-pl` |
| **Dubbing** | `docker-compose run --rm transcribe "URL" --dub` |
| **Dubbing audio-only** | `docker-compose run --rm transcribe "URL" --dub-audio-only` |
| **Wgraj napisy** | `docker-compose run --rm transcribe --local /data/video.mp4 --burn-subtitles` |
| **Napisy dwujęzyczne** | `docker-compose run --rm transcribe "URL" --language en --translate en-pl --burn-subtitles --dual-language` |
| **Inny model** | `docker-compose run --rm transcribe "URL" --model medium` |
| **Napisy dwujęzyczne** |  |
| **Bez GPU** | `docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe "URL"` |
| **Pomoc** | `docker-compose run --rm transcribe --help` |

## Historia zmian

### v4.3 (Obecna)

- **Napisy dwujęzyczne (ASS):** Nowa funkcjonalność `--dual-language` do wgrywania napisów z jednoczesnym wyświetlaniem oryginału i tłumaczenia
- **Format ASS:** Wsparcie dla plików ASS (Advanced SubStation Alpha) obok SRT
- **Nowe moduły:** ass_writer.py (generowanie dwujęzycznych napisów ASS), warning_suppressor.py (tłumienie ostrzeżeń bibliotek)
- **Rozszerzenie architektury:** 13 → 15 wyspecjalizowanych modułów
- **Ulepszone command_builders:** Automatyczne wykrywanie formatu napisów (SRT/ASS) i odpowiednie przetwarzanie w FFmpeg
- **Ulepszone audio_mixer:** Uniwersalna obsługa plików napisów (SRT i ASS) przy wgrywaniu do wideo
- Argument CLI: `--dual-language` (wymaga `--translate` i `--burn-subtitles`)
- Dwujęzyczne napisy ASS: żółty oryginał u góry ekranu + białe tłumaczenie na dole

### v4.2

- **Refaktoryzacja architektury:** Podział monolitycznego `transcribe.py` (2890 linii) na 13 wyspecjalizowanych modułów
- **Modularna struktura:** Acykliczny graf zależności (DAG) dla lepszej organizacji kodu
- **Nowe moduły:** output_manager, command_builders, validators, youtube_processor, audio_processor, device_manager, transcription_engines, segment_processor, translation, srt_writer, tts_generator, audio_mixer, utils
- **Lepsza czytelność:** Każdy moduł odpowiada za konkretny aspekt aplikacji (separation of concerns)
- **Łatwiejsze rozwijanie:** Modułowa struktura ułatwia dodawanie nowych funkcji i testowanie

### v4.1

- **Dwa silniki TTS:** Edge TTS (domyślny, cloudowy) i Coqui TTS (lokalny, wysokiej jakości)
- **Coqui TTS:** Lokalne generowanie dubbingu offline z modelami 100-500MB, opcjonalne GPU
- **Rozszerzone wsparcie językowe Edge TTS:** Niemiecki, Francuski, Hiszpański, Włoski, Rosyjski, Japoński, Chiński, Koreański, Ukraiński, Czeski
- Argumenty CLI: `--tts-engine`, `--coqui-model`, `--coqui-speaker`
- Automatyczne wykrywanie i inicjalizacja odpowiedniego silnika TTS
- Wsparcie dla multi-speaker modeli Coqui TTS

### v4.0

- Trzy silniki transkrypcji: OpenAI Whisper (domyślny), Faster-Whisper (CPU), WhisperX (zaawansowany)
- Zmiana domyślnego silnika z faster-whisper na whisper (szybszy GPU support)
- WhisperX z word-level alignment i speaker diarization
- Refaktoryzacja transcribe_chunk() jako dispatcher do silników
- Dockerfile zoptymalizowany: zmiana z devel (8-10GB) na runtime (3-4GB)
- Usunięcie ctranslate2 z zależności (faster-whisper teraz CPU-only)
- CUDA 12.4 + cuDNN 9 Runtime (zamiast 12.8 devel)

### v3.2

- Pełna dockeryzacja z CUDA 12.8 + cuDNN 9
- Automatyczny GPU/CPU fallback w Docker
- docker-compose dla łatwego uruchamiania

### v3.1

- Wgrywanie napisów do wideo (--burn-subtitles)
- Customizacja stylu napisów

### v3.0

- Dubbing TTS z Microsoft Edge TTS
- Pobieranie wideo z YouTube
- Wybór silnika transkrypcji

### v2.0

- Obsługa plików lokalnych
- Tłumaczenie napisów

### v1.0

- Pierwsza wersja
