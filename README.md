# YouTube to SRT Transcription Tool

Narzędzie do automatycznej transkrypcji filmów z YouTube do formatu plików napisów SRT z wykorzystaniem lokalnego modelu AI (parakeet-mlx).

## Funkcjonalności

- 🎥 Pobieranie audio z filmów YouTube
- 🔊 Automatyczna konwersja do formatu mono 16kHz WAV
- ✂️ Podział długich nagrań na segmenty (~30 minut)
- 🤖 Transkrypcja z wykorzystaniem modelu parakeet-mlx (Apple Silicon)
- ⏱️ Generowanie napisów z precyzyjnymi timestampami
- 📝 Eksport do formatu SRT
- 🧹 Automatyczne czyszczenie plików tymczasowych

## Wymagania

### Zależności systemowe

- **Python 3.8+**
- **ffmpeg** - do przetwarzania audio
- **yt-dlp** - do pobierania filmów z YouTube

### Instalacja zależności systemowych

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
choco install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

### Zależności Python

Zainstaluj wymagane pakiety:

```bash
pip install -r requirements.txt
```

Plik `requirements.txt` zawiera:
- `yt-dlp` - pobieranie wideo z YouTube
- `parakeet-mlx` - model transkrypcji dla Apple Silicon

**Uwaga:** `parakeet-mlx` działa tylko na urządzeniach Apple z chipem M1/M2/M3 (Apple Silicon).

## Użycie

### Podstawowe użycie

Transkrypcja filmu YouTube do pliku SRT:

```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Wynik zostanie zapisany jako `VIDEO_ID.srt`.

### Własna nazwa pliku wyjściowego

```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" -o moj_plik.srt
```

### Flagi deweloperskie (ukryte)

Narzędzie zawiera kilka flag do testowania poszczególnych etapów:

**Tylko pobieranie audio:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --only-download
```

**Pobieranie + podział na chunki:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --only-chunk
```

**Pełny proces bez zapisu SRT:**
```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --only-transcribe
```

**Test generowania SRT:**
```bash
python transcribe.py --test-merge
```

## Jak to działa

Pipeline składa się z 5 etapów:

1. **Walidacja i pobieranie audio**
   - Walidacja URL YouTube
   - Sprawdzenie zależności systemowych (ffmpeg, yt-dlp)
   - Pobieranie audio i konwersja do mono 16kHz WAV

2. **Chunking audio**
   - Podział długich nagrań na segmenty po ~30 minut
   - Dla krótszych nagrań pomijany

3. **Transkrypcja**
   - Przetwarzanie każdego segmentu przez model parakeet-mlx
   - Generowanie segmentów tekstowych z timestampami

4. **Scalanie i generowanie SRT**
   - Łączenie segmentów z wielu chunków
   - Korekta timestampów (przesunięcie dla kolejnych chunków)
   - Formatowanie do standardu SRT

5. **Cleanup**
   - Automatyczne usuwanie plików tymczasowych
   - Działanie nawet w przypadku błędu (blok `finally`)

## Format wyjściowy

Plik SRT w standardowym formacie:

```
1
00:00:00,000 --> 00:00:05,000
Pierwszy segment transkrypcji.

2
00:00:05,000 --> 00:00:10,500
Drugi segment transkrypcji.

3
00:00:10,500 --> 00:00:15,000
Trzeci segment transkrypcji.
```

## Obsługa błędów

Narzędzie obsługuje różne przypadki błędów z czytelnymi komunikatami:

- ❌ Niepoprawny URL YouTube
- ❌ Brak ffmpeg lub yt-dlp
- ❌ Niedostępny film (prywatny, usunięty, geo-blocked)
- ❌ Brak połączenia z internetem
- ❌ Brak biblioteki parakeet-mlx
- ❌ Błędy podczas transkrypcji

## Przykłady

### Przykład 1: Krótki film
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Wynik:
- Pobrane audio: `dQw4w9WgXcQ.wav`
- Brak podziału (film krótszy niż 30 min)
- Transkrypcja jednego segmentu
- Plik wyjściowy: `dQw4w9WgXcQ.srt`

### Przykład 2: Długi podcast
```bash
python transcribe.py "https://www.youtube.com/watch?v=LONG_VIDEO" -o podcast.srt
```

Wynik:
- Pobrane audio: `LONG_VIDEO.wav`
- Podział na 3 chunki po 30 min
- Transkrypcja 3 segmentów z postępem
- Plik wyjściowy: `podcast.srt`

### Przykład 3: Test
```bash
python transcribe.py --test-merge
```

Wynik:
- Utworzenie `test_output.srt` z przykładowymi danymi
- Weryfikacja poprawności formatu SRT

## Struktura projektu

```
PROJEKT_TRANSKRYPCJA/
├── transcribe.py          # Główny skrypt
├── requirements.txt       # Zależności Python
├── README.md             # Dokumentacja
└── specs/
    └── plan.md           # Szczegółowa specyfikacja etapów
```

## Ograniczenia

- **Apple Silicon only**: Model parakeet-mlx wymaga urządzenia z chipem M1/M2/M3
- **Czas przetwarzania**: Transkrypcja może zająć kilka minut w zależności od długości filmu
- **Jakość transkrypcji**: Zależy od jakości audio i akcentu mówcy
- **Język**: Model wspiera przede wszystkim język angielski

## Rozwój

### Uruchomienie testów

Test generowania SRT:
```bash
python transcribe.py --test-merge
```

Weryfikacja składni:
```bash
python -m py_compile transcribe.py
```

### Etapy rozwoju

- ✅ Etap 1: Walidacja i pobieranie audio
- ✅ Etap 2: Chunking audio
- ✅ Etap 3: Transkrypcja
- ✅ Etap 4: Scalanie i generowanie SRT
- ✅ Etap 5: Pipeline CLI i cleanup

## Licencja

Projekt edukacyjny - użyj na własną odpowiedzialność.

## Autor

Projekt stworzony w ramach nauki automatyzacji transkrypcji wideo.

## Wsparcie

W przypadku problemów:

1. Sprawdź, czy wszystkie zależności są zainstalowane
2. Upewnij się, że masz Apple Silicon (M1/M2/M3)
3. Sprawdź, czy URL YouTube jest poprawny
4. Sprawdź logi błędów w terminalu

---

**Wskazówka:** Pierwsze uruchomienie może zająć więcej czasu, ponieważ parakeet-mlx musi pobrać model AI (~600MB).
