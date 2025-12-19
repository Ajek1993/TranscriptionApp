# Plan: Lokalne pliki wideo + Tłumaczenie

## Podsumowanie

Dodanie dwóch nowych funkcji do `transcribe.py`:

1. **Transkrypcja lokalnych plików wideo** (MP4, MKV, AVI, MOV)
2. **Tłumaczenie napisów** (polski ↔ angielski) przy użyciu deep-translator

## Pliki do modyfikacji

- `transcribe.py` - główna logika (3 nowe funkcje + modyfikacje CLI i main())
- `requirements.txt` - nowa zależność deep-translator
- `README.md` - aktualizacja dokumentacji

---

## Etap 1: Nowe importy i stałe

**Lokalizacja:** linie 11-21 w `transcribe.py`

```python
# Dodać:
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov'}

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
```

---

## Etap 2: Funkcja validate_video_file()

**Lokalizacja:** po `validate_youtube_url()` (~linia 36)

```python
def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """Walidacja lokalnego pliku wideo."""
    path = Path(file_path)
    if not path.exists():
        return False, f"Blad: Plik nie istnieje: {file_path}"
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"Blad: Nieobslugiwany format. Wspierane: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
    return True, "OK"
```

---

## Etap 3: Funkcja extract_audio_from_video()

**Lokalizacja:** po `download_audio()` (~linia 149)

```python
def extract_audio_from_video(video_path: str, output_dir: str = ".") -> Tuple[bool, str, str]:
    """Ekstrakcja audio z lokalnego pliku wideo do WAV (mono, 16kHz, PCM 16-bit)."""
    # 1. Walidacja pliku
    # 2. Generowanie nazwy wyjściowej: {stem}.wav
    # 3. Komenda ffmpeg:
    #    ffmpeg -i input -vn -acodec pcm_s16le -ar 16000 -ac 1 -y output.wav
    # 4. Obsługa błędów (wzorzec z download_audio)
```

---

## Etap 4: Funkcja translate_segments()

**Lokalizacja:** przed `write_srt()` (~linia 304)

```python
def translate_segments(
    segments: List[Tuple[int, int, str]],
    source_lang: str,
    target_lang: str,
    batch_size: int = 50
) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
    """Tłumaczenie tekstu segmentów z zachowaniem timestampów."""
    # 1. Sprawdzenie TRANSLATOR_AVAILABLE
    # 2. GoogleTranslator(source=src, target=tgt)
    # 3. Batch translation z progress bar (tqdm)
    # 4. Zachowanie timestampów, podmiana tekstu
```

---

## Etap 5: Modyfikacje CLI

**Lokalizacja:** ~linie 453-469

Nowe argumenty:

```python
parser.add_argument('-l', '--local', type=str,
                   help='Sciezka do lokalnego pliku wideo (MP4, MKV, AVI, MOV)')
parser.add_argument('-t', '--translate', type=str, choices=['pl-en', 'en-pl'],
                   help='Tlumaczenie (pl-en: polski->angielski, en-pl: angielski->polski)')
parser.add_argument('--language', type=str, default='pl',
                   help='Jezyk transkrypcji (domyslnie: pl)')
```

---

## Etap 6: Modyfikacja main() - rozgałęzienie źródła

**Lokalizacja:** ~linia 525 (po sprawdzeniu zależności)

```python
if args.local:
    # Tryb lokalny
    is_valid, error_msg = validate_video_file(args.local)
    if not is_valid:
        print(error_msg)
        return 1
    success, message, audio_path = extract_audio_from_video(args.local, temp_dir)
    input_stem = Path(args.local).stem
else:
    # Tryb YouTube (istniejąca logika)
    # ... walidacja URL, download_audio ...
    input_stem = video_id
```

---

## Etap 7: Modyfikacja main() - krok tłumaczenia

**Lokalizacja:** ~linia 638 (po transkrypcji, przed generowaniem SRT)

```python
if args.translate:
    print(f"\n=== Etap: Tlumaczenie ===")
    src_lang, tgt_lang = args.translate.split('-')
    success, message, translated_segments = translate_segments(
        all_segments, src_lang, tgt_lang
    )
    if not success:
        print(message)
        return 1
    all_segments = translated_segments
```

---

## Etap 8: Użycie args.language w transkrypcji

**Lokalizacja:** ~linia 606

Zmiana z hardcoded `"pl"` na `args.language`:

```python
success, message, segments = transcribe_chunk(
    chunk_path,
    model_size=args.model,
    language=args.language,  # zamiast "pl"
    segment_progress_bar=segment_pbar
)
```

---

## Etap 9: Aktualizacja requirements.txt

Dodać:

```
deep-translator>=1.11.0
```

---

## Etap 10: Aktualizacja README.md

Dodać sekcje:

- Nowe opcje CLI (`--local`, `--translate`, `--language`)
- Przykłady użycia dla lokalnych plików
- Przykłady użycia tłumaczenia

---

## Przykłady użycia po implementacji

```bash
# Lokalne wideo
python transcribe.py --local film.mp4

# YouTube + tłumaczenie na angielski
python transcribe.py "https://youtube.com/..." --translate pl-en

# Lokalne + tłumaczenie + własna nazwa
python transcribe.py --local film.mkv --translate pl-en -o subtitles_en.srt

# Angielskie wideo z tłumaczeniem na polski
python transcribe.py --local movie.mp4 --language en --translate en-pl
```

---

## Kolejność implementacji

1. Dodać importy i stałe
2. Dodać `validate_video_file()`
3. Dodać `extract_audio_from_video()`
4. Dodać `translate_segments()`
5. Zmodyfikować argumenty CLI
6. Zmodyfikować `main()` - rozgałęzienie źródła wejścia
7. Zmodyfikować `main()` - krok tłumaczenia
8. Zmienić hardcoded language na `args.language`
9. Zaktualizować `requirements.txt`
10. Zaktualizować `README.md`
