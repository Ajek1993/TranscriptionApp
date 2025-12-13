# YouTube to SRT Transcription Tool

Narzędzie do automatycznej transkrypcji filmów z YouTube do formatu napisów SRT przy użyciu modelu Whisper.

## Opis

Aplikacja pobiera audio z YouTube, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelu Whisper (faster-whisper). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty.

## Funkcje

- Pobieranie audio z YouTube w formacie WAV
- Automatyczny podział długich nagrań na fragmenty (~30 minut)
- Transkrypcja z wykorzystaniem modelu Whisper
- Wsparcie dla GPU (CUDA) i CPU
- Generowanie pliku SRT zgodnego ze standardem
- Automatyczne czyszczenie plików tymczasowych
- Wsparcie dla języka polskiego i innych języków

## Wymagania

### Wymagane narzędzia

1. **Python 3.7+**
2. **ffmpeg** - do przetwarzania audio
   - Windows: `winget install FFmpeg` lub `choco install ffmpeg`
   - Lub pobierz z https://ffmpeg.org/download.html
3. **yt-dlp** - do pobierania z YouTube
   - Instalacja: `pip install yt-dlp`

### Wymagane biblioteki Python

```bash
pip install faster-whisper
pip install yt-dlp
```

### Opcjonalnie (dla akceleracji GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Instalacja

1. Sklonuj lub pobierz repozytorium
2. Zainstaluj wymagane narzędzia (ffmpeg, yt-dlp)
3. Zainstaluj biblioteki Python:

```bash
pip install faster-whisper yt-dlp
```

## Użycie

### Podstawowe użycie

```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Zaawansowane opcje

```bash
# Użycie większego modelu (lepsza jakość, wolniejsze)
python transcribe.py "URL" --model medium

# Własna nazwa pliku wyjściowego
python transcribe.py "URL" -o moja_transkrypcja.srt

# Dostępne modele
python transcribe.py "URL" --model tiny    # najmniejszy, najszybszy
python transcribe.py "URL" --model base    # domyślny
python transcribe.py "URL" --model small   # średni
python transcribe.py "URL" --model medium  # duży
python transcribe.py "URL" --model large   # największy, najdokładniejszy
```

### Opcje deweloperskie

```bash
# Tylko pobierz audio (bez transkrypcji)
python transcribe.py "URL" --only-download

# Tylko podziel audio na fragmenty
python transcribe.py "URL" --only-chunk

# Tylko transkrybuj (bez generowania SRT)
python transcribe.py "URL" --only-transcribe

# Test generowania SRT z przykładowymi danymi
python transcribe.py --test-merge
```

## Przykład

```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Wynik: Plik `dQw4w9WgXcQ.srt` z napisami gotowymi do użycia w odtwarzaczu wideo.

## Architektura

Projekt składa się z 5 etapów:

1. **Etap 1: Walidacja i pobieranie audio**
   - Walidacja URL YouTube
   - Sprawdzenie zależności (ffmpeg, yt-dlp)
   - Pobieranie audio w formacie WAV (mono, 16kHz)

2. **Etap 2: Podział audio na fragmenty**
   - Wykrywanie długości audio
   - Podział długich nagrań na fragmenty po ~30 minut
   - Optymalizacja dla krótkich nagrań (brak podziału)

3. **Etap 3: Transkrypcja**
   - Wykrywanie dostępności GPU (CUDA) lub fallback na CPU
   - Ładowanie modelu Whisper
   - Transkrypcja każdego fragmentu z timestampami

4. **Etap 4: Scalanie i generowanie SRT**
   - Łączenie segmentów z wszystkich fragmentów
   - Dostosowanie timestampów
   - Generowanie pliku SRT w UTF-8

5. **Etap 5: Pipeline CLI i cleanup**
   - Zarządzanie plikami tymczasowymi
   - Automatyczne czyszczenie po zakończeniu
   - Obsługa błędów i retry dla Windows file locks

## Format wyjściowy

Wygenerowane pliki SRT są zgodne ze standardem i mogą być używane w:
- VLC Media Player
- YouTube (upload napisów)
- Inne odtwarzacze wideo z obsługą SRT

## Wsparcie dla języków

Domyślnie narzędzie używa języka polskiego. Można zmienić język modyfikując parametr `language` w kodzie:

```python
transcribe_chunk(chunk_path, model_size=args.model, language="pl")  # polski
transcribe_chunk(chunk_path, model_size=args.model, language="en")  # angielski
```

## Rozwiązywanie problemów

### "Błąd: Brakuje wymaganych narzędzi"
- Upewnij się, że ffmpeg i yt-dlp są zainstalowane i dostępne w PATH

### "Błąd: faster-whisper nie jest zainstalowany"
- Uruchom: `pip install faster-whisper`

### "Nie można użyć GPU, przełączam na CPU"
- Normalna sytuacja jeśli nie masz karty NVIDIA lub CUDA
- Transkrypcja będzie działać na CPU (wolniej)

### Błędy uprawnień przy usuwaniu plików tymczasowych (Windows)
- Narzędzie automatycznie retry
- Pliki można usunąć ręcznie z folderu tymczasowego

## Licencja

Projekt edukacyjny - dostępny do użytku zgodnie z licencjami używanych bibliotek:
- faster-whisper (MIT)
- yt-dlp (Unlicense)
- ffmpeg (LGPL/GPL)

## Autor

Projekt stworzony jako MVP do transkrypcji materiałów wideo z YouTube.
