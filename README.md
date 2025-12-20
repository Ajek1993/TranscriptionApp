# Transcription Tool - YouTube & Local Files to SRT

Narzędzie do automatycznej transkrypcji filmów z YouTube lub lokalnych plików wideo do formatu napisów SRT przy użyciu modelu Whisper, z wbudowanym wsparciem dla tłumaczenia i dubbingu TTS.

## Opis

Aplikacja pobiera audio z YouTube lub tworzy je z lokalnych plików wideo, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelu Whisper. Wspiera trzy silniki transkrypcji: OpenAI Whisper (domyślny, GPU), Faster-Whisper (CPU) i WhisperX (zaawansowany z alignmentem i diaryzacją). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty oraz opcjonalne tłumaczenie między polskim a angielskim. Nowa funkcjonalność dubbingu TTS pozwala na wygenerowanie polskiej ścieżki audio z synchronizacją czasową i mixowaniem z oryginalnym dźwiękiem.

## Funkcje

- **Transkrypcja z YouTube**: Pobieranie audio z YouTube w formacie WAV
- **Pobieranie wideo z YouTube**: Pobieranie pełnego wideo w jakości 720p-4K
- **Transkrypcja z plików lokalnych**: Obsługa MP4, MKV, AVI, MOV
- **Dubbing TTS**: Generowanie dubbingu z Microsoft Edge TTS (polskie i angielskie głosy)
- **Wgrywanie napisów do wideo**: Hardcode subtitles z customizacją stylu (białe napisy, ciemne tło)
- **Wybór silnika transkrypcji**: OpenAI Whisper (domyślny), Faster-Whisper (CPU), WhisperX (zaawansowany)
- **Docker**: Pełna dockeryzacja z CUDA 12.4 + cuDNN 9 i GPU/CPU fallback
- Automatyczny podział długich nagrań na fragmenty (~30 minut)
- Transkrypcja z wykorzystaniem modelu Whisper (pl, en, i inne języki)
- Tłumaczenie napisów: polski ↔ angielski (deep-translator + Google Translate)
- Zaawansowane opcje narratora: kontrola segmentów, wypełnianie luk, pauzy
- Wsparcie dla GPU (CUDA) i CPU
- Generowanie pliku SRT zgodnego ze standardem
- Automatyczne czyszczenie plików tymczasowych
- Wsparcie dla wielu języków transkrypcji

## Dokumentacja deweloperska

- **[Dokumentacja wszystkich funkcji](README_FUNCTIONS.md)** - Szczegółowy opis wszystkich funkcji z kodu źródłowego `transcribe.py`, parametrów, wartości zwracanych i przykładów użycia

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
├── data/                    # Pliki wejściowe/wyjściowe (volume mount)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── transcribe.py
├── requirements.txt
└── README.md
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

### 7. Wybór modelu Whisper

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

### 8. Starsze opcje silnika transkrypcji

```bash
# Docker - openai-whisper (domyślny)
docker-compose run --rm transcribe "URL" --engine whisper

# Docker - faster-whisper (CPU fallback)
docker-compose run --rm transcribe "URL" --engine faster-whisper

# Natywnie
python transcribe.py "URL" --engine whisper
```

### 9. Wymuszone CPU (bez GPU)

```bash
# Docker - wyłącz GPU
docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe "URL"

# Przydatne gdy GPU jest zajęte lub ma problemy
```

### 10. Zaawansowane opcje dubbingu

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

### 11. Własne nazwy plików wyjściowych

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

### 12. Pomoc i debugowanie

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
| **Inny model** | `docker-compose run --rm transcribe "URL" --model medium` |
| **Bez GPU** | `docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe "URL"` |
| **Pomoc** | `docker-compose run --rm transcribe --help` |

## Historia zmian

### v4.0 (Obecna)

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
