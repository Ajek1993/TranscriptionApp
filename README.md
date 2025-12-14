# Transcription Tool - YouTube & Local Files to SRT

Narzędzie do automatycznej transkrypcji filmów z YouTube lub lokalnych plików wideo do formatu napisów SRT przy użyciu modelu Whisper, z wbudowanym wsparciem dla tłumaczenia i dubbingu TTS.

## Opis

Aplikacja pobiera audio z YouTube lub tworzy je z lokalnych plików wideo, przetwarza je i generuje plik napisów SRT z wykorzystaniem modelem Whisper (faster-whisper). Wspiera długie materiały audio poprzez automatyczny podział na fragmenty oraz opcjonalne tłumaczenie między polskim a angielskim. Nowa funkcjonalność dubbingu TTS pozwala na wygenerowanie polskiej ścieżki audio z synchronizacją czasową i mixowaniem z oryginalnym dźwiękiem.

## Funkcje

- **Transkrypcja z YouTube**: Pobieranie audio z YouTube w formacie WAV
- **Pobieranie wideo z YouTube**: Pobieranie pełnego wideo w jakości 1080p (lub najlepszej dostępnej)
- **Transkrypcja z plików lokalnych**: Obsługa MP4, MKV, AVI, MOV
- **Dubbing TTS**: Generowanie polskiego dubbingu z Microsoft Edge TTS
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
   - Lub pobierz z [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
3. **yt-dlp** - do pobierania z YouTube
   - Instalacja: `pip install yt-dlp`

### Wymagane biblioteki Python

```bash
pip install -r requirements.txt
```

lub ręcznie:

```bash
pip install faster-whisper yt-dlp tqdm deep-translator edge-tts
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

### Dubbing TTS

Dubbing działa zarówno z plikami lokalnymi jak i YouTube.

```bash
# Lokalny plik wideo
python transcribe.py --local "video.mp4" --dub

# YouTube - automatycznie pobierze wideo w 1080p
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --dub

# YouTube z niestandardową jakością
python transcribe.py "URL" --dub --video-quality 720

# YouTube: angielski film z tłumaczeniem i dubbingiem
python transcribe.py "URL" --language en --translate en-pl --dub

# Własne ustawienia głosu i głośności
python transcribe.py "URL" --dub \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.1 \
  --tts-volume 1.2 \
  --dub-output moj_dubbing.mp4

# Dostępne głosy TTS
--tts-voice pl-PL-MarekNeural   # Męski głos (domyślnie)
--tts-voice pl-PL-ZofiaNeural   # Żeński głos

# Dostępne jakości wideo (YouTube)
--video-quality 720    # HD
--video-quality 1080   # Full HD (domyślnie)
--video-quality 1440   # 2K
--video-quality 2160   # 4K
```

#### Parametry dubbingu

| Parametr            | Opis                                        | Domyślna wartość      |
| ------------------- | ------------------------------------------- | --------------------- |
| `--dub`             | Włącz generowanie dubbingu                  | Wyłączone             |
| `--video-quality`   | Jakość wideo z YouTube (720/1080/1440/2160) | 1080                  |
| `--tts-voice`       | Wybór głosu (MarekNeural lub ZofiaNeural)   | pl-PL-MarekNeural     |
| `--tts-volume`      | Głośność TTS (0.0-2.0)                      | 1.0                   |
| `--original-volume` | Głośność oryginalnego audio (0.0-1.0)       | 0.2 (ściszone do 20%) |
| `--dub-output`      | Nazwa pliku wyjściowego z dubbingiem        | video_dubbed.mp4      |

**Uwaga**: Przy dubbingu z YouTube wideo jest pobierane do katalogu tymczasowego i automatycznie usuwane po zakończeniu.

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
python transcribe.py "URL" --language es  # špánělski
python transcribe.py "URL" --language fr  # francuski
python transcribe.py "URL" --language de  # niemiecki

# Pełny workflow: angielskie wideo z YouTube → polski dubbing
python transcribe.py "URL" \
  --language en \
  --translate en-pl \
  --dub \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.15 \
  --dub-output polski_dubbing.mp4
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

## Przykłady

### Podstawowa transkrypcja

```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Wynik: Plik `dQw4w9WgXcQ.srt` z napisami gotowymi do użycia w odtwarzaczu wideo.

### Dubbing lokalnego filmu

```bash
python transcribe.py --local "film.mp4" --dub
```

Wynik:

- `film.srt` - napisy
- `film_dubbed.mp4` - wideo z polskim dubbingiem

### Dubbing filmu z YouTube

```bash
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --dub
```

Wynik:

- `VIDEO_ID.srt` - napisy
- `VIDEO_ID_dubbed.mp4` - wideo z polskim dubbingiem

## Architektura

Projekt składa się z 6 etapów:

1. **Etap 1: Walidacja i pobieranie audio/wideo**

   - Walidacja URL YouTube lub ścieżki lokalnej
   - Sprawdzenie zależności (ffmpeg, yt-dlp)
   - Pobieranie audio/wideo w odpowiednim formacie

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

5. **Etap 5: Tłumaczenie (opcjonalnie)**

   - Tłumaczenie tekstów napisów przy zachowaniu timestampów
   - Wsparcie dla kierunków: pl→en, en→pl

6. **Etap 6: Dubbing TTS (opcjonalnie)**
   - Generowanie TTS dla każdego segmentu z Microsoft Edge TTS
   - Automatyczne przyspieszanie TTS gdy przekracza slot czasowy
   - Łączenie segmentów TTS z synchronizacją czasową (ffmpeg concat)
   - Mixowanie oryginalnego audio (ściszone) z TTS
   - Tworzenie finalnego wideo z nową ścieżką audio

### Jak działa dubbing TTS

1. **Pobieranie wideo**: Jeśli YouTube + --dub, pobiera pełne wideo do temp
2. **Generowanie segmentów**: Dla każdego napisu generowany jest plik MP3 z polskim lektorem
3. **Kontrola tempa**: Jeśli TTS jest dłuższy niż slot czasowy, automatycznie przyspieszany jest do max +50%
4. **Synchronizacja**: Segmenty TTS są łączone z użyciem concat filter (cisza → TTS → cisza → TTS...)
5. **Mixowanie**: Oryginalne audio jest ściszane (domyślnie do 20%), a TTS nakładany w pełnej głośności
6. **Finalizacja**: Oryginalne wideo + nowa ścieżka audio = film z dubbingiem

## Format wyjściowy

### Pliki SRT

Wygenerowane pliki SRT są zgodne ze standardem i mogą być używane w:

- VLC Media Player
- YouTube (upload napisów)
- Inne odtwarzacze wideo z obsługą SRT

### Pliki wideo z dubbingiem

- Format: MP4
- Kodek wideo: copy (bez rekodowania)
- Kodek audio: AAC, 192 kbps, stereo, 44.1kHz
- Zawartość: oryginalne wideo + zmixowane audio (oryginał + TTS)

## Wsparcie dla języków

### Transkrypcja

Domyślnie narzędzie transkrybuje materiały w języku polskim. Możesz zmienić język za pomocą flagi `--language`:

```bash
python transcribe.py "URL" --language pl  # polski (domyślnie)
python transcribe.py "URL" --language en  # angielski
python transcribe.py "URL" --language es  # špánělski
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

### Głosy TTS

Dostępne polskie głosy Microsoft Edge TTS:

- **pl-PL-MarekNeural**: Męski głos (domyślny)
- **pl-PL-ZofiaNeural**: Żeński głos

## Rozwiązywanie problemów

### "Błąd: Brakuje wymaganych narzędzi"

- Upewnij się, że ffmpeg i yt-dlp są zainstalowane i dostępne w PATH
- `ffmpeg --version` i `yt-dlp --version` powinny zwrócić informacje o wersji

### "Błąd: faster-whisper nie jest zainstalowany"

- Uruchom: `pip install faster-whisper`
- Lub zainstaluj wszystkie zależności: `pip install -r requirements.txt`

### "Błąd: edge-tts nie jest zainstalowany"

- Uruchom: `pip install edge-tts`
- Jest wymagany tylko jeśli używasz flagi `--dub`

### "Błąd: deep-translator nie jest zainstalowany"

- Uruchom: `pip install deep-translator`
- Jest wymagany tylko jeśli używasz flagi `--translate`

### "Nie można użyć GPU, przełączam na CPU"

- Normalna sytuacja jeśli nie masz karty NVIDIA lub CUDA
- Transkrypcja będzie działać na CPU (wolniej)
- Jeśli chcesz GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### TTS za długi dla slotu czasowego

- Narzędzie automatycznie przyspiesza TTS do maksymalnie +50%
- Jeśli nadal za długi, segment zostanie przycięty
- Możesz dostosować głośności: `--original-volume 0.1` (cichsze tło) lub `--tts-volume 1.2` (głośniejszy TTS)

### Błędy sieciowe przy generowaniu TTS

- Edge TTS wymaga połączenia internetowego
- Narzędzie automatycznie retry (3 próby z exponential backoff)
- Sprawdź połączenie z internetem

### Problemy z pobieraniem wideo z YouTube

- Sprawdź czy wideo jest publicznie dostępne
- Niektóre wideo mogą być zablokowane geograficznie
- Spróbuj zaktualizować yt-dlp: `pip install --upgrade yt-dlp`

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

## Wydajność

### Transkrypcja

| Model  | Jakość       | Szybkość (CPU) | Szybkość (GPU) | Pamięć |
| ------ | ------------ | -------------- | -------------- | ------ |
| tiny   | Podstawowa   | Bardzo szybka  | Bardzo szybka  | ~1 GB  |
| base   | Dobra        | Szybka         | Szybka         | ~1 GB  |
| small  | Bardzo dobra | Średnia        | Szybka         | ~2 GB  |
| medium | Doskonała    | Wolna          | Średnia        | ~5 GB  |
| large  | Najlepsza    | Bardzo wolna   | Wolna          | ~10 GB |

### Dubbing TTS

- Generowanie TTS: ~1-3 sekundy na segment (zależy od połączenia internetowego)
- Mixowanie audio: zależy od długości materiału
- Dla 60-minutowego filmu: ~10-20 minut całkowitego czasu przetwarzania

### Pobieranie z YouTube

- Jakość 1080p: ~2-5 minut dla filmu 10-minutowego (zależy od prędkości internetu)
- Jakość 720p: ~1-3 minuty dla filmu 10-minutowego

## Technologie

- **faster-whisper**: Transkrypcja audio na tekst
- **yt-dlp**: Pobieranie z YouTube
- **ffmpeg**: Przetwarzanie audio i wideo
- **deep-translator**: Tłumaczenie Google Translate
- **edge-tts**: Microsoft Edge Text-to-Speech
- **tqdm**: Progress bary
- **asyncio**: Asynchroniczne generowanie TTS

## Licencja

Projekt edukacyjny - dostępny do użytku zgodnie z licencjami używanych bibliotek:

- faster-whisper (MIT)
- yt-dlp (Unlicense)
- ffmpeg (LGPL/GPL)
- deep-translator (Apache 2.0)
- edge-tts (GPL-3.0)
- tqdm (Mozilla Public License 2.0)

## Historia zmian

### v3.0 (Obecna wersja)

- **Nowa funkcja**: Pobieranie pełnego wideo z YouTube (720p/1080p/1440p/4K)
- **Nowa funkcja**: Dubbing TTS z YouTube (automatyczne pobieranie wideo do temp)
- **Nowa funkcja**: Dubbing TTS z Microsoft Edge TTS (głosy polskie)
- **Nowa funkcja**: Automatyczne przyspieszanie TTS dla dopasowania do slotów czasowych
- **Nowa funkcja**: Mixowanie oryginalnego audio z TTS (konfigurowalne głośności)
- **Nowa funkcja**: Generowanie wideo z dubbingiem (MP4 z nową ścieżką audio)
- **Ulepszenie**: Zamiana amix na concat filter aby zapobiec dynamicznym zmianom głośności TTS
- **Ulepszenie**: Retry mechanizm dla błędów sieciowych TTS
- **Ulepszenie**: Progress bary dla generowania TTS

### v2.0

- **Nowa funkcja**: Obsługa transkrypcji z lokalnych plików wideo (MP4, MKV, AVI, MOV)
- **Nowa funkcja**: Tłumaczenie napisów (polski ↔ angielski)
- **Ulepszenie**: Zmiana domyślnego języka transkrypcji na konfigurowalny (`--language`)
- **Ulepszenie**: Lepsze komunikaty błędów dla lokalnych plików

### v1.0

- Pierwotna wersja z obsługą YouTube

## Autor

Projekt stworzony jako MVP do transkrypcji materiałów wideo z YouTube, rozszerzony o obsługę plików lokalnych, tłumaczenie i dubbing TTS.

## Wsparcie

W przypadku problemów lub pytań:

1. Sprawdź sekcję "Rozwiązywanie problemów" powyżej
2. Upewnij się że wszystkie zależności są zainstalowane
3. Sprawdź logi błędów w konsoli

---

**Przykładowy workflow: YouTube → Polski Dubbing**

```bash
# 1. Pobierz angielski film z YouTube, przetłumacz i dodaj polski dubbing
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --language en \
  --translate en-pl \
  --dub \
  --video-quality 1080 \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.15

# Wynik:
# - VIDEO_ID.srt (polskie napisy)
# - VIDEO_ID_dubbed.mp4 (wideo z polskim lektorem)
```
