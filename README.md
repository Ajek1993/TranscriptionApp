# Transcription Tool - YouTube & Local Files to SRT

Narzędzie do automatycznej transkrypcji filmów z YouTube lub lokalnych plików wideo do formatu napisów SRT przy użyciu modelu Whisper, z wbudowanym wsparciem dla tłumaczenia i dubbingu TTS.

## Opis

Aplikacja pobiera audio z YouTube lub tworzy je z lokalnych plików wideo, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelem Whisper (faster-whisper). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty oraz opcjonalne tłumaczenie między polskim a angielskim. Nowa funkcjonalność dubbingu TTS pozwala na wygenerowanie polskiej ścieżki audio z synchronizacją czasową i mixowaniem z oryginalnym dźwiękiem.

## Funkcje

- **Transkrypcja z YouTube**: Pobieranie audio z YouTube w formacie WAV
- **Pobieranie wideo z YouTube**: Pobieranie pełnego wideo w jakości 720p-4K
- **Transkrypcja z plików lokalnych**: Obsługa MP4, MKV, AVI, MOV
- **Dubbing TTS**: Generowanie dubbingu z Microsoft Edge TTS (polskie i angielskie głosy)
- **Wgrywanie napisów do wideo**: Hardcode subtitles z customizacją stylu (białe napisy, ciemne tło)
- **Wybór silnika transkrypcji**: faster-whisper (szybki) lub openai-whisper (dokładny)
- **Docker**: Pełna dockeryzacja z CUDA 12.8 i GPU/CPU fallback
- Automatyczny podział długich nagrań na fragmenty (~30 minut)
- Transkrypcja z wykorzystaniem modelu Whisper (pl, en, i inne języki)
- Tłumaczenie napisów: polski ↔ angielski (deep-translator + Google Translate)
- Zaawansowane opcje narratora: kontrola segmentów, wypełnianie luk, pauzy
- Wsparcie dla GPU (CUDA) i CPU
- Generowanie pliku SRT zgodnego ze standardem
- Automatyczne czyszczenie plików tymczasowych
- Wsparcie dla wielu języków transkrypcji

## Docker

Aplikacja jest w pełni zdockeryzowana z automatycznym wsparciem GPU (CUDA 12.8) i fallback na CPU.

### Szybki start

\`\`\`bash
# 1. Zbuduj obraz
docker-compose build

# 2. Test GPU (opcjonalnie)
docker-compose run --rm transcribe python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 3. Użycie
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
\`\`\`

### Przykłady użycia Docker

\`\`\`bash
# Transkrypcja YouTube (GPU jeśli dostępne, inaczej CPU)
docker-compose run --rm transcribe "https://youtube.com/watch?v=ID" --model base

# Transkrypcja lokalnego pliku (plik musi być w ./data/)
docker-compose run --rm transcribe --local /data/video.mp4

# Dubbing z tłumaczeniem (angielski -> polski)
docker-compose run --rm transcribe "URL" --language en --translate en-pl --dub

# Wymuszone CPU (bez GPU)
docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe

# Interaktywny shell do debugowania
docker-compose run --rm --entrypoint bash transcribe

# Pomoc
docker-compose run --rm transcribe --help
\`\`\`

### Struktura katalogów Docker

\`\`\`
PROJEKT_TRANSKRYPCJA/
├── data/                    # Pliki wejściowe/wyjściowe (volume mount)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── transcribe.py
├── requirements.txt
└── README.md
\`\`\`

### Wymagania Docker

**Windows (Docker Desktop + WSL2):**
1. Windows 10/11 z WSL2
2. NVIDIA GPU Driver >= 550.54 (dla CUDA 12.8)
3. Docker Desktop z włączonym WSL2 backend

**Weryfikacja:**
\`\`\`bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
\`\`\`

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
   - Windows: \`winget install FFmpeg\` lub \`choco install ffmpeg\`
3. **yt-dlp** - do pobierania z YouTube
   - Instalacja: \`pip install yt-dlp\`

### Instalacja

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Dla GPU:
\`\`\`bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
\`\`\`

## Użycie

### Podstawowe

\`\`\`bash
# YouTube
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Plik lokalny
python transcribe.py --local "video.mp4"
\`\`\`

### Dubbing TTS

\`\`\`bash
python transcribe.py "URL" --dub
python transcribe.py --local "video.mp4" --dub --tts-voice pl-PL-ZofiaNeural
\`\`\`

### Tłumaczenie

\`\`\`bash
python transcribe.py "URL" --language en --translate en-pl
\`\`\`

### Modele

\`\`\`bash
python transcribe.py "URL" --model tiny    # najszybszy
python transcribe.py "URL" --model base    # domyślny
python transcribe.py "URL" --model medium  # dokładniejszy
python transcribe.py "URL" --model large   # najdokładniejszy
\`\`\`

## Historia zmian

### v3.2 (Obecna)
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
