# Plan GUI dla Aplikacji Transkrypcyjnej

## Framework: Gradio (webowy interfejs)

---

## 1. Struktura plikow

```
TranscriptionApp/
|-- gui.py                  # NOWY: Glowny plik uruchomieniowy GUI
|-- data/
|   |-- gui/                # NOWY: Modul GUI
|       |-- __init__.py
|       |-- config.py       # Konfiguracja (opcje, wartosci domyslne)
|       |-- handlers.py     # Handlery zdarzen (logika przetwarzania)
```

---

## 2. Layout interfejsu - 3 zakladki

### Zakladka 1: Transkrypcja (glowna)

```
+------------------------------------------------------------------+
|  === ZRODLO ===                                                   |
|  ( ) YouTube URL    [________________________]                    |
|  ( ) Plik lokalny   [Wybierz plik...]                             |
+------------------------------------------------------------------+
|  === PODSTAWOWE USTAWIENIA === (klikane)                          |
|  Model:         [Dropdown: base | tiny | small | medium | large]  |
|  Jezyk:         [Dropdown: auto | pl | en | de | fr | ...]        |
|  Silnik:        [Dropdown: whisper | whisperx]                    |
+------------------------------------------------------------------+
|  === TLUMACZENIE ===                                              |
|  [x] Wlacz tlumaczenie                                            |
|  Z: [auto] -> Na: [pl | en | de | fr | ...]                       |
+------------------------------------------------------------------+
|  [v] Zaawansowane ustawienia (klikni aby rozwinac)                |
|  +--------------------------------------------------------------+ |
|  | Urzadzenie:    [auto | cuda | cpu]                           | |
|  | Timeout (s):   [____1800____]                                 | |
|  | WhisperX:      [ ] Align  [ ] Diarization                    | |
|  | HF Token:      [______________________]                       | |
|  +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
|              [ ROZPOCZNIJ TRANSKRYPCJE ]                          |
+------------------------------------------------------------------+
```

### Zakladka 2: Dubbing / Napisy

```
+------------------------------------------------------------------+
|  === ZRODLO ===                                                   |
|  ( ) YouTube URL    [________________________]                    |
|  ( ) Plik lokalny   [Wybierz plik...]                             |
|  ( ) Uzyj pliku SRT [Wybierz SRT...]                              |
+------------------------------------------------------------------+
|  === PODSTAWOWE === (klikane)                                     |
|  [x] Dubbing TTS     [x] Wpal napisy do wideo                     |
|  Typ dubbingu:  ( ) Wideo  ( ) Tylko audio (WAV)                  |
|  [ ] Napisy dwujezyczne                                           |
+------------------------------------------------------------------+
|  === USTAWIENIA TTS === (klikane)                                 |
|  Silnik:     [Dropdown: edge | coqui]                             |
|  Glos:       [Dropdown: pl-PL-MarekNeural | pl-PL-ZofiaNeural...] |
|  Glosnosc TTS:      [====|====] 1.0                               |
|  Glosnosc oryginalu:[==|======] 0.3                               |
+------------------------------------------------------------------+
|  === TRYB LEKTORA ===                                             |
|  [ ] Tryb lektora (laczy segmenty dla naturalniejszego czytania)  |
|  Merge gap (ms): [____300____]                                    |
+------------------------------------------------------------------+
|  [v] Zaawansowane (do wpisania recznie)                           |
|  +--------------------------------------------------------------+ |
|  | Maks. dlugosc segmentu (s): [__10__]                         | |
|  | Maks. slow w segmencie:     [__15__]                         | |
|  | [x] Wypelnij luki   Min. pauza: [__300__] ms                 | |
|  | Maks. luka:         [__2000__] ms                            | |
|  | Jakosc wideo:  [1080 | 720 | 1440 | 2160]                    | |
|  +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
|              [ ROZPOCZNIJ PRZETWARZANIE ]                         |
+------------------------------------------------------------------+
```

### Zakladka 3: Pobieranie

```
+------------------------------------------------------------------+
|  === POBIERANIE Z YOUTUBE ===                                     |
|  URL: [____________________________________________]               |
|                                                                   |
|  Co pobrac:                                                       |
|  ( ) Wideo           Jakosc: [1080 | 720 | 1440 | 2160]           |
|  ( ) Tylko audio     Jakosc: [best | 192 | 128 | 96]              |
+------------------------------------------------------------------+
|              [ POBIERZ ]                                          |
+------------------------------------------------------------------+
```

### Sekcja Output (zawsze widoczna na dole)

```
+------------------------------------------------------------------+
|  === PROGRESS ===                                                 |
|  [=================>              ] 65%  Transkrypcja...          |
|                                                                   |
|  === LOG ===                                                      |
|  | [INFO] Ladowanie modelu whisper base...                       |
|  | [INFO] Transkrypcja: audio.wav...                             |
|  | [OK] Gotowe: 156 segmentow                                    |
|                                                                   |
|  === PLIKI WYJSCIOWE ===                                          |
|  [Download: output.srt]  [Download: output_dubbed.mp4]            |
+------------------------------------------------------------------+
```

---

## 3. Komponenty Gradio do uzycia

| Funkcja | Komponent | Typ |
|---------|-----------|-----|
| URL YouTube | `gr.Textbox` | input |
| Plik lokalny | `gr.File` | upload |
| Model whisper | `gr.Dropdown` | select |
| Jezyk | `gr.Dropdown` | select |
| Wlacz tlumaczenie | `gr.Checkbox` | bool |
| Dubbing | `gr.Checkbox` | bool |
| Typ dubbingu | `gr.Radio` | select |
| Glosnosc | `gr.Slider` | float |
| Tryb lektora | `gr.Checkbox` | bool |
| Timeout, merge gap | `gr.Number` | int |
| Progress | `gr.Progress` | display |
| Log | `gr.Textbox` (readonly) | display |
| Pliki wyjsciowe | `gr.Files` | download |
| Zaawansowane | `gr.Accordion` | container |

---

## 4. Mapowanie opcji CLI -> GUI

### Podstawowe (klikane):
- `--model` -> Dropdown
- `--language` -> Dropdown
- `--engine` -> Dropdown
- `--translate` -> Checkbox + 2x Dropdown
- `--dub` -> Checkbox
- `--burn-subtitles` -> Checkbox
- `--tts-engine` -> Dropdown
- `--tts-voice` -> Dropdown
- `--tts-volume` -> Slider
- `--original-volume` -> Slider
- `--narrator-mode` -> Checkbox

### Zaawansowane (input recznie):
- `--device` -> Dropdown w accordion
- `--transcription-timeout` -> Number
- `--whisperx-align` -> Checkbox
- `--whisperx-diarize` -> Checkbox
- `--hf-token` -> Textbox (password)
- `--max-segment-duration` -> Number
- `--max-segment-words` -> Number
- `--fill-gaps` -> Checkbox
- `--min-pause` -> Number
- `--max-gap-fill` -> Number
- `--merge-gap` -> Number
- `--video-quality` -> Dropdown

---

## 5. Konfiguracja (config.py)

```python
MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEFAULT_MODEL = "base"

ENGINES = ["whisper", "whisperx"]

LANGUAGES = {
    "auto": "Automatycznie",
    "pl": "Polski", "en": "Angielski", "de": "Niemiecki",
    "fr": "Francuski", "es": "Hiszpanski", "it": "Wloski",
    "pt": "Portugalski", "ru": "Rosyjski", "uk": "Ukrainski",
    "cs": "Czeski", "nl": "Holenderski", "ja": "Japonski",
    "zh-cn": "Chinski", "ko": "Koreanski", "tr": "Turecki",
    "ar": "Arabski", "hu": "Wegierski"
}

VOICES_EDGE = {
    "pl-PL-MarekNeural": "Polski (Marek)",
    "pl-PL-ZofiaNeural": "Polski (Zofia)",
    "en-US-GuyNeural": "Angielski US (Guy)",
    "en-US-JennyNeural": "Angielski US (Jenny)",
    "de-DE-ConradNeural": "Niemiecki (Conrad)",
    "de-DE-KatjaNeural": "Niemiecki (Katja)",
    # ... pozostale glosy
}

DEVICES = ["auto", "cuda", "cpu"]
VIDEO_QUALITIES = ["720", "1080", "1440", "2160"]
AUDIO_QUALITIES = ["best", "192", "128", "96"]

# Wartosci domyslne
DEFAULT_TTS_VOLUME = 1.0
DEFAULT_ORIGINAL_VOLUME = 0.3
DEFAULT_TIMEOUT = 1800
DEFAULT_MERGE_GAP = 300
```

---

## 6. Pliki krytyczne do modyfikacji/reuzucia

| Plik | Funkcje do reuzucia |
|------|---------------------|
| `data/validators.py` | `validate_youtube_url()`, `validate_video_file()` |
| `data/youtube_processor.py` | `download_audio()`, `download_video()` |
| `data/transcription_engines.py` | `transcribe_chunk()` |
| `data/translation.py` | `translate_segments()` |
| `data/tts_generator.py` | `generate_tts_segments()` |
| `data/srt_writer.py` | `write_srt()` |
| `data/ass_writer.py` | `write_dual_language_ass()` |
| `data/audio_mixer.py` | `mix_audio_tracks()`, `burn_subtitles()` |
| `data/device_manager.py` | `detect_device()`, `clear_cuda_cache()` |

---

## 7. Kolejnosc implementacji

### Faza 1: Podstawowa struktura
1. Utworz `gui.py` z pustymi zakladkami
2. Utworz `data/gui/__init__.py`, `config.py`
3. Test uruchomienia: `python gui.py`

### Faza 2: Zakladka Transkrypcja
1. Komponenty input (URL, plik)
2. Dropdowny (model, engine, language)
3. Handler transkrypcji w `handlers.py`
4. Output/progress

### Faza 3: Zakladka Dubbing
1. Komponenty TTS (silnik, glos, slidery)
2. Checkboxy (dubbing, napisy, tryb lektora)
3. Handler dubbingu

### Faza 4: Zakladka Pobieranie
1. Prosty handler download

### Faza 5: Zaawansowane
1. Accordion z opcjami
2. WhisperX, kontrola segmentow

---

## 8. Weryfikacja

### Test 1: Prosta transkrypcja
1. Uruchom `python gui.py`
2. Wklej krotki URL YouTube (< 1 min)
3. Wybierz model `tiny`, jezyk `auto`
4. Kliknij "Rozpocznij"
5. Sprawdz czy SRT zostal wygenerowany

### Test 2: Dubbing
1. Uzyj tego samego URL
2. Wlacz "Dubbing TTS", wybierz glos
3. Sprawdz czy MP4 z dubbingiem zostal wygenerowany

### Test 3: Tryb lektora
1. Wlacz tryb lektora
2. Sprawdz czy segmenty sa laczone (mniej plikow TTS)

---

## 9. Przyklad minimalnego gui.py

```python
import gradio as gr

def transcribe(url, model, language, progress=gr.Progress()):
    progress(0.1, desc="Walidacja...")
    # ... logika ...
    progress(1.0, desc="Gotowe!")
    return "Transkrypcja zakonczona", "output.srt"

with gr.Blocks(title="Transkrypcja Video") as app:
    gr.Markdown("# Transkrypcja Video")

    with gr.Tabs():
        with gr.TabItem("Transkrypcja"):
            url = gr.Textbox(label="YouTube URL")
            model = gr.Dropdown(["tiny","base","small"], value="base", label="Model")
            lang = gr.Dropdown(["auto","pl","en"], value="auto", label="Jezyk")

            with gr.Accordion("Zaawansowane", open=False):
                device = gr.Dropdown(["auto","cuda","cpu"], value="auto")
                timeout = gr.Number(value=1800, label="Timeout (s)")

            btn = gr.Button("Rozpocznij", variant="primary")

        with gr.TabItem("Dubbing"):
            gr.Markdown("TODO: Dubbing options")

        with gr.TabItem("Pobieranie"):
            gr.Markdown("TODO: Download options")

    # Output
    log = gr.Textbox(label="Log", lines=5)
    files = gr.Files(label="Pliki")

    btn.click(transcribe, [url, model, lang], [log, files])

if __name__ == "__main__":
    app.launch()
```

---

## 10. Uwagi techniczne

1. **Pamiec GPU**: Wywoluj `clear_cuda_cache()` po kazdej operacji
2. **Pliki tymczasowe**: Uzywaj `tempfile.mkdtemp()`, sprzataj po zakonczeniu
3. **Threading**: Gradio async, ale moduly synchroniczne - rozważ `asyncio.to_thread()`
4. **Walidacja URL**: Dodaj walidacje "on change" z debouncem
