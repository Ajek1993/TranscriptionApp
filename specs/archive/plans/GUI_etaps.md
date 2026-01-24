# Etapy realizacji GUI - Aplikacja Transkrypcyjna

## Etap 1: Struktura bazowa i konfiguracja

### 1.1 Utworzenie struktury katalogow
- [x] Utworz katalog `data/gui/`
- [x] Utworz plik `data/gui/__init__.py`
- [x] Utworz plik `data/gui/config.py`
- [x] Utworz plik `data/gui/handlers.py`
- [x] Utworz plik `gui.py` (glowny punkt wejscia)

### 1.2 Konfiguracja (`config.py`)
- [x] Zdefiniuj liste modeli Whisper: `MODELS`
- [x] Zdefiniuj silniki: `ENGINES`
- [x] Zdefiniuj slownik jezykow: `LANGUAGES`
- [x] Zdefiniuj glosy Edge TTS: `VOICES_EDGE`
- [x] Zdefiniuj urzadzenia: `DEVICES`
- [x] Zdefiniuj jakosci wideo/audio: `VIDEO_QUALITIES`, `AUDIO_QUALITIES`
- [x] Ustaw wartosci domyslne (volume, timeout, merge_gap)

### 1.3 Weryfikacja etapu 1
- [x] Uruchom `python gui.py` - powinno otworzyc pusta strone Gradio

---

## Etap 2: Zakladka Transkrypcja

### 2.1 Komponenty zrodla
- [x] Radio button: wybor YouTube URL / Plik lokalny
- [x] Textbox: pole na URL YouTube
- [x] File upload: wybor pliku lokalnego
- [x] Logika ukrywania/pokazywania pol w zaleznosci od wyboru

### 2.2 Podstawowe ustawienia
- [x] Dropdown: Model whisper (tiny, base, small, medium, large, large-v2, large-v3)
- [x] Dropdown: Jezyk (auto + lista jezykow z config)
- [x] Dropdown: Silnik (whisper, whisperx)

### 2.3 Sekcja tlumaczenia
- [x] Checkbox: Wlacz tlumaczenie
- [x] Dropdown: Jezyk zrodlowy (auto + lista)
- [x] Dropdown: Jezyk docelowy (lista bez auto)
- [x] Logika ukrywania gdy checkbox odznaczony

### 2.4 Zaawansowane ustawienia (Accordion)
- [x] Dropdown: Urzadzenie (auto, cuda, cpu)
- [x] Number: Timeout (domyslnie 1800s)
- [x] Checkbox: WhisperX Align
- [x] Checkbox: WhisperX Diarization
- [x] Textbox (password): HuggingFace Token

### 2.5 Handler transkrypcji (`handlers.py`)
- [x] Funkcja `handle_transcription()`:
  - Walidacja URL/pliku (reuse `validators.py`)
  - Pobranie audio (reuse `youtube_processor.py`)
  - Transkrypcja (reuse `transcription_engines.py`)
  - Opcjonalne tlumaczenie (reuse `translation.py`)
  - Zapis SRT (reuse `srt_writer.py`)
  - Zwrot sciezki do pliku

### 2.6 Weryfikacja etapu 2
- [ ] Test: Transkrypcja krotkiego URL YouTube (model tiny)
- [ ] Test: Transkrypcja pliku lokalnego
- [ ] Test: Transkrypcja z tlumaczeniem

---

## Etap 3: Zakladka Dubbing / Napisy

### 3.1 Komponenty zrodla
- [x] Radio button: YouTube URL / Plik lokalny / Plik SRT
- [x] Textbox: URL
- [x] File upload: plik wideo/audio
- [x] File upload: plik SRT

### 3.2 Podstawowe opcje
- [x] Checkbox: Dubbing TTS
- [x] Checkbox: Wpal napisy do wideo
- [x] Radio: Typ dubbingu (Wideo / Tylko audio WAV)
- [x] Checkbox: Napisy dwujezyczne

### 3.3 Ustawienia TTS
- [x] Dropdown: Silnik TTS (edge, coqui)
- [x] Dropdown: Glos (dynamicznie ladowany z config)
- [x] Slider: Glosnosc TTS (0.0 - 2.0, domyslnie 1.0)
- [x] Slider: Glosnosc oryginalu (0.0 - 1.0, domyslnie 0.3)

### 3.4 Tryb lektora
- [x] Checkbox: Tryb lektora
- [x] Number: Merge gap (ms) - widoczne gdy tryb lektora wlaczony

### 3.5 Zaawansowane (Accordion)
- [x] Number: Maks. dlugosc segmentu (s)
- [x] Number: Maks. slow w segmencie
- [x] Checkbox: Wypelnij luki
- [x] Number: Min. pauza (ms)
- [x] Number: Maks. luka (ms)
- [x] Dropdown: Jakosc wideo

### 3.6 Handler dubbingu (`handlers.py`)
- [x] Funkcja `handle_dubbing()`:
  - Walidacja wejscia
  - Ladowanie/generowanie SRT
  - Generowanie TTS (reuse `tts_generator.py`)
  - Mixowanie audio (reuse `audio_mixer.py`)
  - Opcjonalne wypalanie napisow (reuse `audio_mixer.py`)
  - Zwrot sciezki do pliku wyjsciowego

### 3.7 Weryfikacja etapu 3
- [ ] Test: Dubbing z YouTube URL
- [ ] Test: Dubbing z pliku lokalnego
- [ ] Test: Dubbing z gotowego SRT
- [ ] Test: Tryb lektora
- [ ] Test: Napisy dwujezyczne

---

## Etap 4: Zakladka Pobieranie

### 4.1 Komponenty
- [x] Textbox: URL YouTube
- [x] Radio: Co pobrac (Wideo / Tylko audio)
- [x] Dropdown: Jakosc wideo (dynamicznie widoczne)
- [x] Dropdown: Jakosc audio (dynamicznie widoczne)

### 4.2 Handler pobierania (`handlers.py`)
- [x] Funkcja `handle_download()`:
  - Walidacja URL
  - Pobranie wideo/audio (reuse `youtube_processor.py`)
  - Zwrot sciezki do pliku

### 4.3 Weryfikacja etapu 4
- [ ] Test: Pobranie wideo w roznych jakosciach
- [ ] Test: Pobranie tylko audio

---

## Etap 5: Sekcja Output i Progress

### 5.1 Komponenty wspolne (widoczne na dole)
- [x] Progress bar: pasek postepu z opisem
- [x] Textbox (readonly): Log operacji
- [x] Files: Lista plikow do pobrania

### 5.2 Integracja z handlerami
- [x] Dodaj `gr.Progress()` do wszystkich handlerow
- [x] Implementuj logowanie krokow do textboxa
- [x] Zwracaj liste plikow wyjsciowych

### 5.3 Weryfikacja etapu 5
- [ ] Test: Progress bar aktualizuje sie podczas transkrypcji
- [ ] Test: Logi wyswietlaja sie poprawnie
- [ ] Test: Pliki wyjsciowe sa dostepne do pobrania

---

## Etap 6: Polishing i optymalizacja

### 6.1 Walidacja na biezaco
- [x] Walidacja URL YouTube przy wpisywaniu (debounce)
- [x] Walidacja rozszerzenia pliku przy uploadzie
- [x] Komunikaty bledow w UI

### 6.2 Zarzadzanie pamieca
- [x] Wywolanie `clear_cuda_cache()` po kazdej operacji
- [x] Czyszczenie plikow tymczasowych

### 6.3 UX
- [x] Blokowanie przycisku podczas przetwarzania
- [x] Pokazywanie szacowanego czasu (opcjonalne)
- [x] Responsywnosc layoutu

### 6.4 Weryfikacja koncowa
- [ ] Test pelnego pipeline: URL -> transkrypcja -> tlumaczenie -> dubbing
- [ ] Test na roznych przegladarkach
- [ ] Test na roznych rozmiarach okna

---

## Zaleznosci miedzy etapami

```
Etap 1 (Struktura)
    |
    v
Etap 2 (Transkrypcja) --> Etap 3 (Dubbing)
    |                         |
    v                         v
Etap 4 (Pobieranie)     Etap 5 (Output)
    |                         |
    +-----------+-------------+
                |
                v
          Etap 6 (Polishing)
```

---

## Pliki do reuzucia z istniejacego kodu

| Etap | Plik zrodlowy | Funkcje |
|------|---------------|---------|
| 2 | `data/validators.py` | `validate_youtube_url()`, `validate_video_file()` |
| 2 | `data/youtube_processor.py` | `download_audio()` |
| 2 | `data/transcription_engines.py` | `transcribe_chunk()` |
| 2 | `data/translation.py` | `translate_segments()` |
| 2 | `data/srt_writer.py` | `write_srt()` |
| 3 | `data/tts_generator.py` | `generate_tts_segments()` |
| 3 | `data/audio_mixer.py` | `mix_audio_tracks()`, `burn_subtitles()` |
| 3 | `data/ass_writer.py` | `write_dual_language_ass()` |
| 4 | `data/youtube_processor.py` | `download_video()` |
| 6 | `data/device_manager.py` | `detect_device()`, `clear_cuda_cache()` |

---

## Estymacja zlozonosci

| Etap | Zlozonosc | Opis |
|------|-----------|------|
| 1 | Niska | Tworzenie struktury i konfiguracji |
| 2 | Wysoka | Najwiecej logiki, integracja z transkrypcja |
| 3 | Wysoka | Wiele opcji, integracja TTS i mixowania |
| 4 | Niska | Prosta funkcjonalnosc pobierania |
| 5 | Srednia | Integracja progressu i logow |
| 6 | Srednia | Dopracowanie UX i optymalizacja |
