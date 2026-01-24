# Plan prac MVP - Transkrypcja YouTube do SRT

## Etap 1: Walidacja i pobieranie audio

**Cel:** Pobrać ścieżkę audio z YouTube jako plik WAV (mono, 16kHz).

**Kroki:**
1. Utworzyć plik `transcribe.py` z funkcją `validate_youtube_url(url)` używającą regex
2. Utworzyć plik `requirements.txt` z zależnościami: `yt-dlp` i `parakeet-mlx`
3. Dodać funkcję `check_dependencies()` sprawdzającą dostępność `ffmpeg` i `yt-dlp`
4. Dodać funkcję `download_audio(url, output_dir)` wywołującą `yt-dlp` przez subprocess
5. Dodać podstawowy `argparse` przyjmujący URL jako argument

**Edge cases do obsłużenia:**
- Brak ffmpeg → wyświetl instrukcję instalacji (`brew install ffmpeg`)
- Brak yt-dlp → wyświetl instrukcję instalacji (`pip install yt-dlp`)
- Niedostępny film (prywatny, usunięty, geo-blocked) → czytelny błąd
- Brak internetu / przerwane pobieranie → błąd z informacją

**Definicja "done":**
- Skrypt przyjmuje URL YouTube jako argument
- Pobiera audio i zapisuje jako WAV mono 16kHz
- Przy błędnym URL lub problemie z pobieraniem wyświetla czytelny komunikat
- Plik `requirements.txt` istnieje z poprawnymi zależnościami

**Test:**
```bash
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

**Definicja "done":**
- Funkcja dzieli dowolny plik WAV na chunki po ~30 min
- Dla pliku krótszego niż chunk_duration zwraca jeden plik
- Chunki są poprawnie nazwane (chunk_001.wav, chunk_002.wav, ...)

**Test (developerski - wymaga wcześniejszego pobrania audio):**
```bash
# Najpierw pobierz audio:
python transcribe.py "https://www.youtube.com/watch?v=LONG_VIDEO" --only-download
# Następnie przetestuj chunking wewnętrznie (sprawdź logi):
# - Dla pliku >30 min: powstają chunk_001.wav, chunk_002.wav, ...
# - Dla pliku <30 min: jeden plik bez podziału
ffprobe chunk_001.wav  # sprawdź długość ~30 min
```

---

## Etap 3: Transkrypcja

**Cel:** Przetworzyć chunk audio przez parakeet-mlx i uzyskać tekst z timestampami.

**Specyfikacja parakeet-mlx:**
- **Wywołanie:** Import jako biblioteka Python: `from parakeet_mlx import transcribe`
- **Input:** Ścieżka do pliku WAV (mono, 16kHz)
- **Output:** Lista słów z timestampami w formacie:
  ```python
  [
      {"word": "tekst", "start": 0.0, "end": 0.5},
      {"word": "słowo", "start": 0.5, "end": 1.0},
      ...
  ]
  ```
- **Agregacja:** Słowa należy zagregować w segmenty (np. po ~10 słów lub na podstawie pauz >0.5s)

**Kroki:**
1. Dodać funkcję `transcribe_chunk(wav_path)` importującą i wywołującą parakeet-mlx
2. Dodać funkcję `aggregate_words_to_segments(words)` grupującą słowa w segmenty
3. Sparsować output do listy `[(start_ms, end_ms, "tekst"), ...]`
4. Dodać flagę `--only-transcribe` do argparse dla testowania (flaga developerska)
5. Obsłużyć błędy parakeet-mlx (brak modelu, błędny format wejścia)

**Edge cases do obsłużenia:**
- Pusty/uszkodzony audio → błąd jeśli parakeet-mlx nie zwraca wyników
- Brak parakeet-mlx → czytelny błąd z instrukcją instalacji

**Definicja "done":**
- Funkcja przyjmuje plik WAV i zwraca listę segmentów z timestampami
- Output jest poprawnie sparsowany (nie surowy tekst)
- Przy błędzie wyświetla zrozumiały komunikat

**Test (developerski):**
```bash
# Użyj pobranego wcześniej audio lub przygotowanego pliku testowego
# Sprawdź w logach:
# - Wypisane segmenty [(start_ms, end_ms, "tekst"), ...]
# - Tekst powinien być zrozumiały (nie musi być idealny)
```

---

## Etap 4: Scalanie i generowanie SRT

**Cel:** Połączyć segmenty z wielu chunków w jeden plik SRT z poprawnymi timestampami.

**Kroki:**
1. Dodać funkcję `merge_segments(chunks_segments, chunk_offsets)` przesuwającą timestampy
2. Dodać funkcję `format_srt_timestamp(ms)` konwertującą milisekundy na format SRT (`HH:MM:SS,mmm`)
3. Dodać funkcję `write_srt(segments, output_path)` zapisującą plik SRT
4. Dodać flagę `--test-merge` z hardcoded danymi testowymi (flaga developerska)

**Edge cases do obsłużenia:**
- Brak segmentów → pusty plik SRT z ostrzeżeniem

**Definicja "done":**
- Funkcja łączy segmenty z wielu chunków
- Timestampy są poprawnie przesunięte (chunk 2 zaczyna się od końca chunk 1)
- Plik SRT ma poprawny format (numery, timestampy, tekst, puste linie)

**Test:**
```bash
python transcribe.py --test-merge
# Sprawdź: powstał test_output.srt
# Otwórz w edytorze - format poprawny
# Timestampy rosną monotonicznie
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

**Definicja "done":**
- Jedno polecenie pobiera, transkrybuje i zapisuje SRT
- Pliki tymczasowe są usuwane po zakończeniu (sukces lub błąd)
- Nazwa pliku wyjściowego to video_id.srt lub podana przez użytkownika

**Test:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=xyz123"
# Sprawdź: powstał xyz123.srt
ls /tmp/  # brak plików tymczasowych
python transcribe.py "https://www.youtube.com/watch?v=xyz123" -o custom_name.srt
# Sprawdź: powstał custom_name.srt
```

---

## Test integracyjny

### Scenariusz sukcesu:
```bash
# Użytkownik uruchamia:
python transcribe.py "https://www.youtube.com/watch?v=VALID_VIDEO_ID"

# Oczekiwany rezultat:
# 1. Wyświetla: "Pobieranie audio..."
# 2. Wyświetla: "Dzielenie na chunki..." (jeśli >30 min)
# 3. Wyświetla: "Transkrypcja chunk 1/N..."
# 4. Wyświetla: "Zapisano: VALID_VIDEO_ID.srt"
# 5. Plik SRT istnieje i ma poprawny format
# 6. Brak plików tymczasowych
```

### Scenariusz błędu - niepoprawny URL:
```bash
python transcribe.py "https://example.com/not-youtube"

# Oczekiwany rezultat:
# "Błąd: Niepoprawny URL YouTube. Podaj link w formacie: https://www.youtube.com/watch?v=VIDEO_ID"
# Exit code: 1
```

### Scenariusz błędu - niedostępny film:
```bash
python transcribe.py "https://www.youtube.com/watch?v=DELETED_VIDEO"

# Oczekiwany rezultat:
# "Błąd: Nie można pobrać filmu. Film może być prywatny, usunięty lub niedostępny w Twoim regionie."
# Exit code: 1
```

### Scenariusz błędu - brak ffmpeg:
```bash
# (przy braku ffmpeg w systemie)
python transcribe.py "https://www.youtube.com/watch?v=xyz"

# Oczekiwany rezultat:
# "Błąd: ffmpeg nie jest zainstalowany. Zainstaluj: brew install ffmpeg"
# Exit code: 1
```

### Scenariusz błędu - brak yt-dlp:
```bash
# (przy braku yt-dlp w systemie)
python transcribe.py "https://www.youtube.com/watch?v=xyz"

# Oczekiwany rezultat:
# "Błąd: yt-dlp nie jest zainstalowany. Zainstaluj: pip install yt-dlp"
# Exit code: 1
```

---

## Podsumowanie

| Etap | Nazwa | Kroki | Zależności |
|------|-------|-------|------------|
| 1 | Walidacja i pobieranie | 5 | - |
| 2 | Chunking | 3 | Etap 1 |
| 3 | Transkrypcja | 5 | Etap 2 |
| 4 | Scalanie SRT | 4 | Etap 3 |
| 5 | Pipeline CLI | 4 | Etap 1-4 |

**Łącznie:** 21 kroków, 5 etapów

**Ścieżka krytyczna:** Etap 3 (transkrypcja) - bez działającego parakeet-mlx nic nie zadziała.

**Minimalna wersja (bez chunkingu):** Etapy 1, 3, 4, 5 - dla filmów <30 min.
