# Transcription Tool - YouTube & Local Files to SRT

Narzędzie do automatycznej transkrypcji filmów z YouTube lub lokalnych plików wideo do formatu napisów SRT przy użyciu modelu Whisper, z wbudowanym wsparciem dla tłumaczenia.

## Opis

Aplikacja pobiera audio z YouTube lub tworzy je z lokalnych plików wideo, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelu Whisper (faster-whisper). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty oraz opcjonalne tłumaczenie między polskim a angielskim.

## Funkcje

- **Transkrypcja z YouTube**: Pobieranie audio z YouTube w formacie WAV
- **Transkrypcja z plików lokalnych**: Obsługa MP4, MKV, AVI, MOV
- Automatyczny podział długich nagrań na fragmenty (~30 minut)
- Transkrypcja z wykorzystaniem modelu Whisper (pl, en, i inne języki)
- Tłumaczenie napisów: polski ↔ angielski (deep-translator + Google Translate)
- Wsparcie dla GPU (CUDA) i CPU
- Generowanie pliku SRT zgodnego ze standardem
- Automatyczne czyszczenie plików tymczasowych
- Wsparcie dla wielu języków transkrypcji

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
pip install -r requirements.txt
```

lub ręcznie:

```bash
pip install faster-whisper yt-dlp tqdm deep-translator
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
pip install -r requirements.txt
```

4. (Opcjonalnie) Zainstaluj PyTorch dla GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Użycie

### YouTube - Podstawowe użycie

```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Pliki lokalne - Podstawowe użycie

```bash
python transcribe.py --local "C:\path\to\video.mp4"
```

Wspierane formaty: **MP4, MKV, AVI, MOV**

### Zaawansowane opcje

```bash
# Wybór modelu Whisper (lepsza jakość = wolniejsze transkrypcje)
python transcribe.py "URL" --model medium

# Zmiana języka transkrypcji (domyślnie: polski)
python transcribe.py "URL" --language en

# Tłumaczenie: polski → angielski
python transcribe.py "URL" --translate pl-en

# Tłumaczenie: angielski → polski
python transcribe.py "URL" --translate en-pl

# Własna nazwa pliku wyjściowego
python transcribe.py "URL" -o moja_transkrypcja.srt

# Dostępne modele
python transcribe.py "URL" --model tiny    # najmniejszy, najszybszy
python transcribe.py "URL" --model base    # domyślny
python transcribe.py "URL" --model small   # średni
python transcribe.py "URL" --model medium  # duży
python transcribe.py "URL" --model large   # największy, najdokładniejszy
```

### Kombinacje funkcji

```bash
# Transkrypcja lokalnego angielskiego wideo z tłumaczeniem na polski
python transcribe.py --local "movie.mp4" --language en --translate en-pl -o movie_pl.srt

# YouTube z lepszym modelem i tłumaczeniem
python transcribe.py "URL" --model medium --translate pl-en

# Transkrypcja w innym języku
python transcribe.py "URL" --language es  # španielski
python transcribe.py "URL" --language fr  # francuski
python transcribe.py "URL" --language de  # niemiecki
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

### Transkrypcja

Domyślnie narzędzie transkrybuje materiały w języku polskim. Możesz zmienić język za pomocą flagi `--language`:

```bash
python transcribe.py "URL" --language pl  # polski (domyślnie)
python transcribe.py "URL" --language en  # angielski
python transcribe.py "URL" --language es  # španielski
python transcribe.py "URL" --language fr  # francuski
python transcribe.py "URL" --language de  # niemiecki
python transcribe.py "URL" --language it  # włoski
```

Whisper obsługuje wszystkie główne języki świata.

### Tłumaczenie

Obecnie dostępne tłumaczenia:
- `pl-en`: Polski → Angielski
- `en-pl`: Angielski → Polski

Tłumaczenie odbywa się za pośrednictwem Google Translate (deep-translator library).

## Rozwiązywanie problemów

### "Błąd: Brakuje wymaganych narzędzi"
- Upewnij się, że ffmpeg i yt-dlp są zainstalowane i dostępne w PATH
- `ffmpeg --version` i `yt-dlp --version` powinny zwrócić informacje o wersji

### "Błąd: faster-whisper nie jest zainstalowany"
- Uruchom: `pip install faster-whisper`
- Lub zainstaluj wszystkie zależności: `pip install -r requirements.txt`

### "Błąd: deep-translator nie jest zainstalowany"
- Uruchom: `pip install deep-translator`
- Jest wymagany tylko jeśli używasz flagi `--translate`

### "Nie można użyć GPU, przełączam na CPU"
- Normalna sytuacja jeśli nie masz karty NVIDIA lub CUDA
- Transkrypcja będzie działać na CPU (wolniej)
- Jeśli chcesz GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Błędy uprawnień przy usuwaniu plików tymczasowych (Windows)
- Narzędzie automatycznie retry
- Pliki można usunąć ręcznie z folderu tymczasowego

### "Błąd: Plik nie istnieje" (dla lokalnych plików)
- Sprawdź czy podałeś pełną ścieżkę do pliku
- Spróbuj użyć cudzysłowów: `--local "C:\path\to\video.mp4"`
- Wspierane formaty: MP4, MKV, AVI, MOV

### "Błąd tłumaczenia"
- Tłumaczenie wymaga połączenia internetowego
- Google Translate API może czasami być niedostępne
- Transkrypcja zostanie zapisana bez tłumaczenia

## Licencja

Projekt edukacyjny - dostępny do użytku zgodnie z licencjami używanych bibliotek:
- faster-whisper (MIT)
- yt-dlp (Unlicense)
- ffmpeg (LGPL/GPL)
- deep-translator (Apache 2.0)
- tqdm (Mozilla Public License 2.0)

## Historia zmian

### v2.0 (Obecna wersja)
- **Nowa funkcja**: Obsługa transkrypcji z lokalnych plików wideo (MP4, MKV, AVI, MOV)
- **Nowa funkcja**: Tłumaczenie napisów (polski ↔ angielski)
- **Ulepszenie**: Zmiana domyślnego języka transkrypcji na konfigurowalny (`--language`)
- **Ulepszenie**: Lepsze komunikaty błędów dla lokalnych plików

### v1.0
- Pierwotna wersja z obsługą YouTube

## Autor

Projekt stworzony jako MVP do transkrypcji materiałów wideo z YouTube, teraz wspierający również pliki lokalne i tłumaczenie.
