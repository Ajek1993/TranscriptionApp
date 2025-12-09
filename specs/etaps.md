# Etapy realizacji MVP

## ETAP 1: Pobieranie audio

**Co robi:** Pobiera ścieżkę audio z YouTube i zapisuje jako WAV.

**Input:** URL YouTube (string)
**Output:** Plik WAV (mono, 16kHz) w katalogu tymczasowym

**Test:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --only-download
# Sprawdź: powstał plik .wav, można go odtworzyć
ffprobe output.wav  # powinno pokazać: mono, 16000 Hz
```

---

## ETAP 2: Chunking audio

**Co robi:** Dzieli długi plik audio na części po ~30 minut.

**Input:** Plik WAV (dowolnej długości)
**Output:** Lista plików WAV (chunk_001.wav, chunk_002.wav, ...)

**Test:**
```bash
# Weź dowolny plik WAV > 1h
python transcribe.py test_audio.wav --only-chunk
# Sprawdź: powstały pliki chunk_*.wav, każdy ~30 min
# Dla pliku 1h30m powinny być 3 chunki
```

---

## ETAP 3: Transkrypcja

**Co robi:** Przetwarza jeden chunk audio przez parakeet-mlx, zwraca tekst z timestampami.

**Input:** Plik WAV (chunk)
**Output:** Lista segmentów: `[(start_ms, end_ms, "tekst"), ...]`

**Test:**
```bash
# Weź krótki plik WAV (~1-2 min) z mową polską
python transcribe.py test_chunk.wav --only-transcribe
# Sprawdź: wypisane segmenty z timestampami i tekstem
# Tekst powinien być zrozumiały (nie musi być idealny)
```

---

## ETAP 4: Scalanie i SRT

**Co robi:** Łączy segmenty z wielu chunków, przesuwa timestampy, zapisuje jako SRT.

**Input:** Lista list segmentów + offsety czasowe chunków
**Output:** Plik .srt

**Test:**
```bash
# Użyj danych testowych (hardcoded segmenty)
python transcribe.py --test-merge
# Sprawdź: powstał plik .srt
# Otwórz w edytorze - format SRT poprawny
# Timestampy rosną monotonicznie
```

---

## ETAP 5: Pipeline CLI

**Co robi:** Łączy wszystkie etapy w jeden przepływ + obsługa błędów + cleanup.

**Input:** URL YouTube (argument CLI)
**Output:** Plik .srt (nazwa = video_id.srt)

**Test:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=xyz123"
# Sprawdź: powstał xyz123.srt
# Brak plików tymczasowych (cleanup zadziałał)
# Przy błędnym URL - czytelny komunikat
```

---

## Podsumowanie

### Kolejność realizacji
```
1. Pobieranie audio  ──→  2. Chunking  ──→  3. Transkrypcja  ──→  4. Scalanie SRT  ──→  5. Pipeline CLI
   [fundamenty]            [przygotowanie]   [core]               [output]              [integracja]
```

### Etap KRYTYCZNY
**ETAP 3: Transkrypcja** - bez działającej transkrypcji cały projekt nie ma sensu. To jedyna część, która wymaga zewnętrznego modelu ML i może mieć niespodziewane problemy (format wyjścia parakeet-mlx, obsługa polskiego).

### Etap do POMINIĘCIA w pierwszej wersji
**ETAP 2: Chunking** - dla krótkich filmów (<30 min) chunking nie jest potrzebny. Pierwsza działająca wersja może transkrybować całe audio na raz. Chunking dodać gdy podstawowy pipeline działa.

### Minimalna działająca wersja (3 etapy)
```
1. Pobieranie audio  ──→  3. Transkrypcja  ──→  4. Scalanie SRT
```
To wystarczy dla filmów <30 min. Chunking i pełny CLI to rozszerzenia.
