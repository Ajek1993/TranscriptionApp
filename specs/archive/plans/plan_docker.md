# Plan Dockeryzacji Projektu Transkrypcji

## Podsumowanie

Dockeryzacja aplikacji CLI transkrypcji audio/video z obsługą GPU (CUDA 12.8) i automatycznym fallback na CPU.

**Wymagania:**

- GPU + CPU fallback (jeden uniwersalny obraz)
- Volume mount + CLI do podawania plików
- docker-compose do łatwiejszego uruchamiania
- CUDA 12.8 jako baza (kompatybilna z faster-whisper, openai-whisper, WhisperX i Coqui TTS)

---

## Obraz bazowy

**Wybrany:** `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04`

**Uzasadnienie:**

- CUDA 12.8 + cuDNN 9 - najnowsza stabilna wersja
- CTranslate2 >= 4.5.0 wymaga CUDA >= 12.3 + cuDNN 9
- Runtime zamiast devel - mniejszy rozmiar (~3GB vs ~8GB)
- Ubuntu 22.04 - stabilna, wsparcie do 2027
- Pełna kompatybilność z: faster-whisper, openai-whisper, WhisperX, Coqui TTS

---

## Kompatybilność bibliotek z CUDA 12.8 + cuDNN 9

| Biblioteka | Wersja | Status | Uwagi |
|---|---|---|---|
| **faster-whisper** | >=1.1.0 | ✅ OK | CTranslate2 >= 4.5.0 wymaga CUDA >= 12.3 + cuDNN 9 |
| **openai-whisper** | >=20240930 | ✅ OK | PyTorch cu124/cu128 |
| **WhisperX** | >=3.3.0 | ✅ OK | Z ctranslate2 >= 4.6.0 + torch >= 2.5.1 |
| **Coqui TTS** | >=0.27.0 | ✅ OK | PyTorch CUDA 12.x, Python 3.10-3.12 |
| **CTranslate2** | >=4.6.0 | ✅ OK | Wymaga cuDNN 9 |
| **PyTorch** | >=2.5.1 | ✅ OK | Wheel cu124 kompatybilny z CUDA 12.8 |

---

## Pliki do utworzenia

### 1. `Dockerfile`

```dockerfile
# ============================================
# DOCKERFILE - Transcription Tool with GPU Support
# ============================================

FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04 AS base

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

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.8)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY transcribe.py .

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

### 2. `docker-compose.yml`

```yaml
version: "3.8"

services:
  transcribe:
    build:
      context: .
      dockerfile: Dockerfile
    image: transcription-tool:latest
    container_name: transcribe

    # GPU configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # Volume mounts
    volumes:
      # Input/output data directory
      - ./data:/data
      # Whisper model cache (persistent)
      - whisper-models:/models

    # Environment variables
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - HF_HOME=/models
      - TORCH_HOME=/models

    # Working directory inside container
    working_dir: /data

    # Interactive mode for CLI
    stdin_open: true
    tty: true

    # Restart policy
    restart: "no"

# Persistent volumes
volumes:
  whisper-models:
    driver: local
```

### 3. `.dockerignore`

```
.venv/
__pycache__/
*.pyc
.git/
.vscode/
specs/
.claude/
*.mp4
*.mkv
*.avi
*.mov
*.wav
*.mp3
data/
models/
```

---

## Kroki implementacji

1. Utworzenie `Dockerfile` w katalogu głównym projektu
2. Utworzenie `docker-compose.yml` w katalogu głównym projektu
3. Utworzenie `.dockerignore` w katalogu głównym projektu
4. Utworzenie katalogu `data/` dla plików wejściowych/wyjściowych
5. Aktualizacja `README.md` - dodanie sekcji Docker

---

## Użycie (dla użytkownika końcowego)

### Szybki start

```bash
# 1. Sklonuj repozytorium
git clone <repo-url>
cd PROJEKT_TRANSKRYPCJA

# 2. Zbuduj obraz
docker-compose build

# 3. Test GPU (opcjonalnie)
docker-compose run --rm transcribe python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 4. Użycie
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Przykłady użycia

```bash
# Transkrypcja YouTube (GPU jeśli dostępne, inaczej CPU)
docker-compose run --rm transcribe "https://youtube.com/watch?v=ID" --model base

# Transkrypcja lokalnego pliku (plik musi być w ./data/)
docker-compose run --rm transcribe --local /data/video.mp4

# Dubbing z tłumaczeniem (angielski -> polski)
docker-compose run --rm transcribe "URL" --language en --translate en-pl --dub

# Wymuszone CPU (bez GPU)
docker-compose run --rm --gpus "" transcribe "URL"

# Interaktywny shell do debugowania
docker-compose run --rm --entrypoint bash transcribe

# Pomoc
docker-compose run --rm transcribe --help
```

### Struktura katalogów

```
PROJEKT_TRANSKRYPCJA/
├── data/                    # Pliki wejściowe/wyjściowe (volume mount)
│   ├── input/               # Pliki do transkrypcji
│   └── output/              # Wyniki (SRT, MP4)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── transcribe.py
├── requirements.txt
└── README.md
```

---

## Wymagania dla użytkownika końcowego

### Windows (Docker Desktop + WSL2)

1. Windows 10/11 z WSL2
2. NVIDIA GPU Driver >= 550.54 (dla CUDA 12.8)
3. Docker Desktop z włączonym WSL2 backend

**Weryfikacja:**

```bash
# W PowerShell/CMD
nvidia-smi

# Test w Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Linux

1. NVIDIA GPU Driver >= 550.54 (dla CUDA 12.8)
2. Docker Engine
3. NVIDIA Container Toolkit

**Instalacja NVIDIA Container Toolkit:**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## GPU/CPU Fallback

Kod w `transcribe.py` już obsługuje automatyczną detekcję:

```python
def detect_device() -> Tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", f"NVIDIA GPU ({gpu_name})"
    except ImportError:
        pass
    return "cpu", "CPU"
```

**Zachowanie:**

- Jeśli GPU dostępne → używa CUDA z `float16`
- Jeśli brak GPU → automatyczny fallback na CPU z `int8`
- Jeśli błąd GPU podczas ładowania modelu → fallback na CPU

---

## Przyszła rozszerzalność (WhisperX, Coqui TTS)

### Kompatybilność

Obraz bazowy `nvidia/cuda:12.8.0-cudnn9` jest w pełni kompatybilny z:

- **faster-whisper** - CTranslate2 >= 4.5.0 wymaga CUDA >= 12.3 + cuDNN 9 ✅
- **openai-whisper** - PyTorch cu124 ✅
- **WhisperX** - ctranslate2 >= 4.6.0 + torch >= 2.5.1 + pyannote.audio ✅
- **Coqui TTS** - PyTorch CUDA 12.x, v0.27.3 (Python 3.10-3.12) ✅

### Architektura modularnych obrazów (opcja na przyszłość)

```
docker/
├── Dockerfile              # Obecna funkcjonalność
├── Dockerfile.whisperx     # + WhisperX z diaryzacją
├── Dockerfile.coqui        # + Coqui TTS
├── docker-compose.yml
├── docker-compose.whisperx.yml
└── docker-compose.coqui.yml
```

### WhisperX - przykład rozszerzenia

```dockerfile
# Dockerfile.whisperx
FROM transcription-tool:latest AS whisperx

USER root
RUN pip install --no-cache-dir whisperx>=3.3.0 pyannote.audio
USER transcriber

ENV WHISPERX_CACHE=/models/whisperx
```

### Coqui TTS - przykład rozszerzenia

```dockerfile
# Dockerfile.coqui
FROM transcription-tool:latest AS coqui

USER root
RUN pip install --no-cache-dir coqui-tts>=0.27.0
USER transcriber

ENV COQUI_CACHE=/models/coqui
```

---

## Zarządzanie modelami Whisper

### Cache w Named Volume (domyślnie)

```yaml
volumes:
  - whisper-models:/models
```

- Persystentne między uruchomieniami
- Zarządzane przez Docker
- Modele pobierane przy pierwszym użyciu

### Bind Mount (alternatywa)

```yaml
volumes:
  - ./models:/models
```

- Łatwy dostęp z hosta
- Możliwość współdzielenia z innymi projektami

### Zmienne środowiskowe dla cache

```yaml
environment:
  - HF_HOME=/models # Hugging Face cache
  - TORCH_HOME=/models # PyTorch model cache
  - XDG_CACHE_HOME=/models # General cache
```

---

## Pliki do modyfikacji

| Plik                 | Akcja         | Opis                                   |
| -------------------- | ------------- | -------------------------------------- |
| `Dockerfile`         | UTWORZYĆ      | Nowy plik                              |
| `docker-compose.yml` | UTWORZYĆ      | Nowy plik                              |
| `.dockerignore`      | UTWORZYĆ      | Nowy plik                              |
| `requirements.txt`   | ZAKTUALIZOWAĆ | Wersje dla CUDA 12.8 + cuDNN 9         |
| `transcribe.py`      | BEZ ZMIAN     | detect_device() już obsługuje fallback |
| `README.md`          | ZAKTUALIZOWAĆ | Dodać sekcję Docker                    |

---

## Szacowany rozmiar obrazu

| Komponent                  | Rozmiar     |
| -------------------------- | ----------- |
| nvidia/cuda:12.8.0-runtime | ~3.2 GB     |
| Python 3.11 + pip          | ~200 MB     |
| PyTorch cu124              | ~2.5 GB     |
| faster-whisper + deps      | ~500 MB     |
| ffmpeg                     | ~100 MB     |
| **RAZEM**                  | **~6.5 GB** |

---

## Źródła

- [nvidia/cuda - Docker Hub](https://hub.docker.com/r/nvidia/cuda)
- [GPU support in Docker Desktop for Windows](https://docs.docker.com/desktop/features/gpu/)
- [Enable GPU support - Docker Compose](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [faster-whisper CUDA compatibility](https://github.com/SYSTRAN/faster-whisper/issues/1086)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [CTranslate2 cuDNN 9 support](https://github.com/OpenNMT/CTranslate2)
