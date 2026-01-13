# Konfiguracja GPU (CUDA) dla Whisper

## Wymagania

- **GPU:** NVIDIA RTX 3070 (lub inna karta z Compute Capability 7.5+)
- **Python:** 3.11 (zalecane, 3.10-3.12 działa)
- **Sterownik NVIDIA:** 527.41+ (sprawdź: `nvidia-smi`)
- **System:** Windows 10/11

## Instalacja krok po kroku

### 1. Stwórz środowisko wirtualne Python 3.11

```bash
# Przejdź do katalogu projektu
cd C:\Users\Arkadiusz\Desktop\ATD\PROJEKT_TRANSKRYPCJA

# Stwórz nowe środowisko .venv
py -3.11 -m venv .venv

# Aktywuj środowisko
.venv\Scripts\activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Zainstaluj podstawowe zależności

```bash
pip install -r requirements.txt
```

### 4. Zainstaluj PyTorch z CUDA 12.1 (DOPASOWANE DO DOCKER)

**WAŻNE:** Użyj PyTorch 2.3.1 + CUDA 12.1 (takie same jak Docker)!

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

**Rozmiar:** ~2.2 GB
**Czas:** ~5-10 min
**Note:** Bez torchvision (nie jest używane w projekcie, tak samo jak Docker)

### 5. (Opcjonalnie) Zainstaluj openai-whisper

```bash
pip install openai-whisper
```

## Weryfikacja

### Test 1: PyTorch CUDA

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

**Oczekiwany output:**
```
PyTorch: 2.3.1+cu121
CUDA: 12.1
GPU: True
```

### Test 2: GPU Detection

```bash
python -c "from transcribe import detect_device; device, info = detect_device(); print(f'Device: {device}'); print(f'Info: {info}')"
```

**Oczekiwany output:**
```
Device: cuda
Info: NVIDIA GPU (NVIDIA GeForce RTX 3070)
```

### Test 3: Transkrypcja z GPU

```bash
python transcribe.py "https://www.youtube.com/watch?v=KRÓTKIE_VIDEO" --model tiny --only-transcribe
```

**Sprawdź w outputcie:**
```
Używane urządzenie: NVIDIA GPU (NVIDIA GeForce RTX 3070)
```

## Troubleshooting

### Problem: `cublas64_12.dll is not found`

**Przyczyna:** Zainstalowano PyTorch z CUDA 11.8 zamiast 12.1

**Rozwiązanie:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

### Problem: `torch.cuda.is_available()` zwraca False

**Rozwiązania:**
1. Sprawdź sterownik NVIDIA: `nvidia-smi` (wymagany ≥527.41)
2. Zaktualizuj sterownik z [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx)
3. Upewnij się że używasz środowiska `.venv`:
   ```bash
   python -c "import sys; print(sys.executable)"
   ```
   Powinno pokazać: `C:\Users\Arkadiusz\Desktop\ATD\PROJEKT_TRANSKRYPCJA\.venv\Scripts\python.exe`

### Problem: Brak miejsca na dysku podczas instalacji

**Rozwiązanie:**
- Wymagane minimum 5 GB wolnego miejsca
- Wyczyść cache pip: `pip cache purge`

### Problem: Python 3.13 - brak kompatybilności

**Rozwiązanie:** Użyj Python 3.11 zamiast 3.13

## Zainstalowane wersje (działające) - Dopasowane do Docker

```
Python: 3.11.x
torch: 2.3.1+cu121 (identyczne jak Docker)
torchaudio: 2.3.1+cu121 (identyczne jak Docker)
torchvision: NIE UŻYWANE (jak w Docker)
openai-whisper: >=20240930
whisperx: >=3.1.1,<4.0.0
transformers: 4.48.0 (pinned, identyczne jak Docker)
```

## Wydajność (RTX 3070 + CUDA 12.1)

| Model | Audio 10 min | CPU      | GPU      | Przyspieszenie |
|-------|--------------|----------|----------|----------------|
| tiny  | 10 min       | ~5 min   | ~1.5 min | 3.3x           |
| base  | 10 min       | ~8 min   | ~2 min   | 4x             |
| small | 10 min       | ~15 min  | ~4 min   | 3.75x          |
| medium| 10 min       | ~30 min  | ~8 min   | 3.75x          |
| large | 10 min       | ~60 min  | ~15 min  | 4x             |

## Notatki

- **Sterownik NVIDIA** musi być ≥527.41 dla pełnej kompatybilności CUDA 12.1
- **RTX 3070** ma 8 GB VRAM - wszystkie modele Whisper (tiny-large) mieszczą się w pamięci
- Środowisko `.venv` **MUSI** być aktywowane przed każdym użyciem

## Aktywacja środowiska (za każdym razem)

```bash
cd C:\Users\Arkadiusz\Desktop\ATD\PROJEKT_TRANSKRYPCJA
.venv\Scripts\activate
```

Po aktywacji zobaczysz `(.venv)` przed promptem w terminalu.
