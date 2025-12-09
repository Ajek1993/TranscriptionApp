# Brief MVP - Transkrypcja YouTube do SRT

## Co boli
Ręczne transkrybowanie długich (2-3h) filmów z YouTube jest czasochłonne, a istniejące narzędzia online mają limity lub słabą jakość dla polskiego.

## Dla kogo
Dla mnie - do lokalnej transkrypcji polskojęzycznych materiałów z YouTube na Macu.

## Dlaczego CLI
Proste w użyciu, łatwe do zintegrowania z agentami AI, bez zbędnego GUI. Jedno polecenie = gotowy plik SRT.

## Core funkcja MVP

**Input:** link do YouTube
**Output:** plik .srt z transkrypcją

### Krok po kroku:

1. **Walidacja URL** - sprawdź czy to poprawny link YouTube
2. **Pobranie audio** - użyj `yt-dlp` do wyciągnięcia ścieżki audio (format: WAV mono 16kHz)
3. **Podział na chunki** - podziel audio na części po ~30-45 minut (dla filmów 2-3h = 3-4 części)
4. **Transkrypcja** - dla każdego chunka uruchom `parakeet-mlx`, zbierz wyniki
5. **Scalenie** - połącz wszystkie fragmenty w jeden plik SRT (z poprawionymi timestampami)
6. **Zapis** - zapisz plik .srt (nazwa: tytuł filmu lub ID)
7. **Cleanup** - usuń pliki tymczasowe (audio, chunki)

### Przykład użycia:
```bash
python transcribe.py "https://www.youtube.com/watch?v=xyz123"
# Output: xyz123.srt
```

## TECHNOLOGIA

### Dlaczego Python
Python to naturalny wybór - yt-dlp i parakeet-mlx to narzędzia Pythonowe, a cała logika to prosty pipeline bez wymagań wydajnościowych.

### Biblioteki

**STDLIB (wbudowane, preferowane):**
- `argparse`: parsowanie argumentów CLI
- `subprocess`: uruchamianie yt-dlp i ffmpeg
- `pathlib`: operacje na ścieżkach plików
- `tempfile`: katalog tymczasowy na audio/chunki
- `re`: walidacja URL YouTube
- `shutil`: cleanup plików tymczasowych

**ZEWNĘTRZNE (tylko jeśli stdlib nie wystarczy):**
- `yt-dlp`: pobieranie audio z YouTube - stdlib nie ma możliwości pobierania z YouTube
- `parakeet-mlx`: transkrypcja speech-to-text na Apple Silicon - stdlib nie ma modelu STT

### Rezygnuję z pydub
Cięcie audio przez `subprocess` + `ffmpeg` to 5 linii kodu. Pydub to nadmiarowa zależność dla jednej operacji.

### Struktura plików (MVP):
```
PROJEKT_TRANSKRYPCJA/
├── transcribe.py      # główny skrypt (cała logika)
├── requirements.txt   # yt-dlp, parakeet-mlx
└── specs/
    └── brief_v3.md
```

Jeden plik `transcribe.py` wystarczy - to prosty pipeline ~150-200 linii. Podział na moduły to przedwczesna abstrakcja.

### Jak uruchomić:
```bash
# Instalacja zależności
pip install yt-dlp parakeet-mlx

# Użycie
python transcribe.py "https://www.youtube.com/watch?v=xyz123"

# Z własną nazwą pliku wyjściowego
python transcribe.py "https://www.youtube.com/watch?v=xyz123" -o moj_film.srt
```

## Edge cases (max 4)

1. **Niedostępny film** (prywatny, usunięty, geo-blocked) → wyświetl czytelny błąd i zakończ
2. **Brak ffmpeg** → sprawdź na starcie, wyświetl instrukcję instalacji (`brew install ffmpeg`)
3. **Brak internetu / przerwane pobieranie** → błąd z informacją co poszło nie tak
4. **Pusty/uszkodzony audio** → błąd jeśli yt-dlp nie wyciągnie audio

## Czego NIE robimy w MVP

- ❌ Obsługa lokalnych plików (tylko YouTube)
- ❌ Wiele formatów wyjściowych (tylko SRT)
- ❌ Cache / zapisywanie postępu
- ❌ Plik konfiguracyjny
- ❌ Logowanie do pliku
- ❌ Progress bar
- ❌ Batch processing (wiele linków naraz)
- ❌ Automatyczne wykrywanie języka (zakładamy polski)
- ❌ Obsługa prywatnych filmów (wymagających cookies/logowania)
- ❌ Chunking z zakładką (akceptujemy ryzyko ucięcia słowa)
- ❌ Sprawdzanie duplikatów
- ❌ Metadane w output (tytuł, URL, data)
