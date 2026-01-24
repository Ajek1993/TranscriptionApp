# Plan prac MVP - Transkrypcja YouTube do SRT (Windows)

## Etap 1: Walidacja i pobieranie audio

**Cel:** Pobrać ścieżkę audio z YouTube jako plik WAV (mono, 16kHz).

**Kroki:**
1. Utworzyć plik `transcribe.py` z funkcją `validate_youtube_url(url)` używającą regex
2. Utworzyć plik `requirements.txt` z zależnościami: `yt-dlp` i `faster-whisper`
3. Dodać funkcję `check_dependencies()` sprawdzającą dostępność `ffmpeg` i `yt-dlp`
4. Dodać funkcję `download_audio(url, output_dir)` wywołującą `yt-dlp` przez subprocess
5. Dodać podstawowy `argparse` przyjmujący URL jako argument

**Edge cases do obsłużenia:**
- Brak ffmpeg → wyświetl instrukcję instalacji (`winget install FFmpeg` lub `choco install ffmpeg`)
- Brak yt-dlp → wyświetl instrukcję instalacji (`pip install yt-dlp`)
- Niedostępny film (prywatny, usunięty, geo-blocked) → czytelny błąd
- Brak internetu / przerwane pobieranie → błąd z informacją

**Uwagi Windows:**
- Użyj `pathlib.Path` dla kompatybilności ścieżek (Windows/Unix)
- ffmpeg może być w PATH lub w katalogu projektu
- Sprawdź dostępność ffmpeg przez `where ffmpeg` lub `which ffmpeg`

**Definicja "done":**
- Skrypt przyjmuje URL YouTube jako argument
- Pobiera audio i zapisuje jako WAV mono 16kHz
- Przy błędnym URL lub problemie z pobieraniem wyświetla czytelny komunikat
- Plik `requirements.txt` istnieje z poprawnymi zależnościami

**Test:**
```cmd
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --only-download
ffprobe output.wav  # powinno pokazać: mono, 16000 Hz
```

---

## Etap 2: Chunking audio

**Cel:** Podzielić długie audio na części po ~30 minut.

**Kroki:**
1. Dodać funkcję `get_audio_duration(wav_path)` używającą ffprobe
2. Dodać funkcję `split_audio(wav_path, chunk_duration_sec, output_dir)` wywołującą ffmpeg
3. Dodać flagę `--only-chunk` do argparse dla testowania (flaga developerska, nie dla użytkownika końcowego)

**Edge cases do obsłużenia:**
- Pusty/uszkodzony audio → błąd jeśli duration = 0 lub ffmpeg zwraca błąd
- Windows: Obsługa długich ścieżek (>260 znaków) przez `\\?\` prefix w razie potrzeby

**Uwagi Windows:**
- ffmpeg działa identycznie jak na macOS/Linux
- Użyj `pathlib.Path` dla nazw chunków
- Tymczasowe pliki w `%TEMP%` lub dedykowanym folderze

**Definicja "done":**
- Funkcja dzieli dowolny plik WAV na chunki po ~30 min
- Dla pliku krótszego niż chunk_duration zwraca jeden plik
- Chunki są poprawnie nazwane (chunk_001.wav, chunk_002.wav, ...)

**Test (developerski - wymaga wcześniejszego pobrania audio):**
```cmd
# Najpierw pobierz audio:
python transcribe.py "https://www.youtube.com/watch?v=LONG_VIDEO" --only-download
# Następnie przetestuj chunking wewnętrznie (sprawdź logi):
# - Dla pliku >30 min: powstają chunk_001.wav, chunk_002.wav, ...
# - Dla pliku <30 min: jeden plik bez podziału
ffprobe chunk_001.wav  # sprawdź długość ~30 min
```

---

## Etap 3: Transkrypcja

**Cel:** Przetworzyć chunk audio przez faster-whisper i uzyskać tekst z timestampami.

**Specyfikacja faster-whisper:**
- **Wywołanie:** Import jako biblioteka Python: `from faster_whisper import WhisperModel`
- **Input:** Ścieżka do pliku WAV (mono, 16kHz)
- **Output:** Generator segmentów z timestampami w formacie:
  ```python
  segments, info = model.transcribe(audio_path, language="pl", word_timestamps=True)
  # Każdy segment zawiera:
  # segment.start, segment.end (float - sekundy)
  # segment.text (str)
  # segment.words (lista słów z timestampami)
  ```
- **Model:** Użyj `base` lub `small` dla polskiego (kompromis jakość/prędkość)
- **Device:** Na Windows może używać CUDA (NVIDIA GPU) lub CPU

**Kroki:**
1. Dodać funkcję `transcribe_chunk(wav_path)` inicjalizującą WhisperModel i wywołującą transcribe
2. Sparsować output do listy `[(start_ms, end_ms, "tekst"), ...]`
3. Dodać flagę `--only-transcribe` do argparse dla testowania (flaga developerska)
4. Obsłużyć błędy faster-whisper (brak modelu, błędny format wejścia)
5. Dodać opcjonalne wykrywanie GPU (CUDA) dla przyspieszenia

**Edge cases do obsłużenia:**
- Pusty/uszkodzony audio → błąd jeśli faster-whisper nie zwraca wyników
- Brak faster-whisper → czytelny błąd z instrukcją instalacji
- Pierwsze uruchomienie → automatyczne pobieranie modelu (może zająć czas)
- Brak GPU → automatyczne przełączenie na CPU

**Uwagi Windows:**
- faster-whisper może używać CUDA jeśli jest zainstalowana obsługa NVIDIA
- Modele są pobierane do `%USERPROFILE%\.cache\huggingface`
- Dla CPU: używaj mniejszego modelu (tiny/base) dla szybszości

**Definicja "done":**
- Funkcja przyjmuje plik WAV i zwraca listę segmentów z timestampami
- Output jest poprawnie sparsowany (nie surowy tekst)
- Przy błędzie wyświetla zrozumiały komunikat
- Automatycznie wykrywa dostępność GPU

**Test (developerski):**
```cmd
# Użyj pobranego wcześniej audio lub przygotowanego pliku testowego
# Sprawdź w logach:
# - Wypisane segmenty [(start_ms, end_ms, "tekst"), ...]
# - Tekst powinien być zrozumiały (nie musi być idealny)
# - Informacja o użyciu CPU/GPU
```

---

## Etap 4: Scalanie i generowanie SRT

**Cel:** Połączyć segmenty z wielu chunków w jeden plik SRT z poprawnymi timestampami.

**Kroki:**
1. Dodać funkcję `merge_segments(chunks_segments, chunk_offsets)` przesuwającą timestampy
2. Dodać funkcję `format_srt_timestamp(ms)` konwertującą milisekundy na format SRT (`HH:MM:SS,mmm`)
3. Dodać funkcję `write_srt(segments, output_path)` zapisującą plik SRT z encoding UTF-8
4. Dodać flagę `--test-merge` z hardcoded danymi testowymi (flaga developerska)

**Edge cases do obsłużenia:**
- Brak segmentów → pusty plik SRT z ostrzeżeniem
- Windows: Zapis pliku z UTF-8 BOM dla kompatybilności z Notatnikiem

**Uwagi Windows:**
- Jawnie określ encoding='utf-8' przy zapisie pliku
- Opcjonalnie dodaj BOM dla starszych edytorów Windows
- Testuj plik SRT w VLC lub innym odtwarzaczu Windows

**Definicja "done":**
- Funkcja łączy segmenty z wielu chunków
- Timestampy są poprawnie przesunięte (chunk 2 zaczyna się od końca chunk 1)
- Plik SRT ma poprawny format (numery, timestampy, tekst, puste linie)
- Plik SRT zapisany z UTF-8 encoding

**Test:**
```cmd
python transcribe.py --test-merge
# Sprawdź: powstał test_output.srt
# Otwórz w Notatniku/VS Code - polskie znaki wyświetlają się poprawnie
# Timestampy rosną monotonicznie
# Otwórz w VLC - napisy się wyświetlają
```

---

## Etap 5: Pipeline CLI i cleanup

**Cel:** Połączyć wszystkie etapy w jeden przepływ z argumentami CLI i automatycznym cleanup.

**Kroki:**
1. Dodać główną funkcję `main()` łączącą: walidacja → download → chunk → transcribe → merge → srt
2. Dodać argument `-o/--output` dla nazwy pliku wyjściowego (domyślnie: video_id.srt)
3. Dodać cleanup plików tymczasowych (tempfile + shutil) w bloku finally
4. Zostawić flagi `--only-*` i `--test-merge` do debugowania (ukryte w help)

**Edge cases do obsłużenia:**
- Wszystkie z poprzednich etapów (propagacja błędów)
- Cleanup musi działać nawet przy błędzie (finally block)
- Windows: Obsługa zablokowanych plików (retry mechanizm dla cleanup)

**Uwagi Windows:**
- Użyj `tempfile.mkdtemp()` dla katalogu tymczasowego
- Przy cleanup: może być potrzebne `time.sleep(0.1)` przed usunięciem (file locks)
- Obsłuż wyjątek `PermissionError` przy cleanup gracefully

**Definicja "done":**
- Jedno polecenie pobiera, transkrybuje i zapisuje SRT
- Pliki tymczasowe są usuwane po zakończeniu (sukces lub błąd)
- Nazwa pliku wyjściowego to video_id.srt lub podana przez użytkownika

**Test:**
```cmd
python transcribe.py "https://www.youtube.com/watch?v=xyz123"
# Sprawdź: powstał xyz123.srt
dir %TEMP%  # brak plików tymczasowych związanych ze skryptem
python transcribe.py "https://www.youtube.com/watch?v=xyz123" -o custom_name.srt
# Sprawdź: powstał custom_name.srt
```

---

## Test integracyjny

### Scenariusz sukcesu:
```cmd
# Użytkownik uruchamia:
python transcribe.py "https://www.youtube.com/watch?v=VALID_VIDEO_ID"

# Oczekiwany rezultat:
# 1. Wyświetla: "Pobieranie audio..."
# 2. Wyświetla: "Dzielenie na chunki..." (jeśli >30 min)
# 3. Wyświetla: "Transkrypcja chunk 1/N..." (+ info o CPU/GPU)
# 4. Wyświetla: "Zapisano: VALID_VIDEO_ID.srt"
# 5. Plik SRT istnieje i ma poprawny format
# 6. Brak plików tymczasowych
```

### Scenariusz błędu - niepoprawny URL:
```cmd
python transcribe.py "https://example.com/not-youtube"

# Oczekiwany rezultat:
# "Błąd: Niepoprawny URL YouTube. Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID"
# Exit code: 1
```

### Scenariusz błędu - niedostępny film:
```cmd
python transcribe.py "https://www.youtube.com/watch?v=DELETED_VIDEO"

# Oczekiwany rezultat:
# "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie."
# Exit code: 1
```

### Scenariusz błędu - brak ffmpeg:
```cmd
# (przy braku ffmpeg w systemie)
python transcribe.py "https://www.youtube.com/watch?v=xyz"

# Oczekiwany rezultat:
# "Błąd: ffmpeg nie jest zainstalowany. Zainstaluj:"
# "  - winget install FFmpeg"
# "  - lub choco install ffmpeg"
# "  - lub pobierz z https://ffmpeg.org/download.html"
# Exit code: 1
```

### Scenariusz błędu - brak yt-dlp:
```cmd
# (przy braku yt-dlp w systemie)
python transcribe.py "https://www.youtube.com/watch?v=xyz"

# Oczekiwany rezultat:
# "Błąd: yt-dlp nie jest zainstalowany. Zainstaluj: pip install yt-dlp"
# Exit code: 1
```

---

## Podsumowanie różnic Windows vs macOS

| Aspekt | macOS (oryginalny plan) | Windows (ten plan) |
|--------|------------------------|-------------------|
| **Silnik transkrypcji** | parakeet-mlx (tylko Apple Silicon) | faster-whisper (uniwersalny) |
| **Akceleracja** | Metal (GPU Apple) | CUDA (NVIDIA GPU) lub CPU |
| **Instalacja ffmpeg** | `brew install ffmpeg` | `winget install FFmpeg` / `choco install ffmpeg` |
| **Ścieżki plików** | Unix-style (/) | Obsługiwane przez pathlib (\\) |
| **Katalog tymczasowy** | `/tmp/` | `%TEMP%\` |
| **Cache modeli** | `~/.cache/` | `%USERPROFILE%\.cache\` |
| **Encoding plików** | UTF-8 (domyślne) | UTF-8 (jawnie określone) |
| **Cleanup** | Prosty shutil.rmtree | + obsługa file locks |

## Podsumowanie

| Etap | Nazwa | Kroki | Zależności |
|------|-------|-------|------------|
| 1 | Walidacja i pobieranie | 5 | - |
| 2 | Chunking | 3 | Etap 1 |
| 3 | Transkrypcja | 5 | Etap 2 |
| 4 | Scalanie SRT | 4 | Etap 3 |
| 5 | Pipeline CLI | 4 | Etap 1-4 |

**Łącznie:** 21 kroków, 5 etapów

**Ścieżka krytyczna:** Etap 3 (transkrypcja) - bez działającego faster-whisper nic nie zadziała.

**Minimalna wersja (bez chunkingu):** Etapy 1, 3, 4, 5 - dla filmów <30 min.

**Zalecana konfiguracja dla Windows:**
- Python 3.8+ (z Microsoft Store lub python.org)
- ffmpeg (przez winget lub chocolatey)
- Opcjonalnie: NVIDIA GPU z CUDA dla szybszej transkrypcji
- Wystarczająco RAM: 4GB minimum, 8GB+ zalecane dla dłuższych filmów
