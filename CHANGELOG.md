# Changelog

Wszystkie ważne zmiany w projekcie będą dokumentowane w tym pliku.

## [5.0.2] - 2026-02-01

### Naprawiono
- Dodano automatyczne czyszczenie plików tymczasowych Gradio (AppData/Local/Temp/gradio) po zakończeniu transkrypcji, dubbingu oraz przy zamykaniu aplikacji

## [5.0.1] - 2026-02-01

### Zmieniono
- Zaktualizowano komendy yt-dlp dla kompatybilności z YouTube 2026 (nowe player_client: android_vr, web_safari)
- Rozszerzono formaty pobierania audio/wideo o dodatkowe warianty (webm fallback)
- Podniesiono minimalną wersję yt-dlp do >=2026.01.31

## [5.0.0] - 2025-01-25

### Dodano
- **Nowy interfejs GUI (Gradio)** z trzema zakładkami:
  - **Transkrypcja**: Pełna obsługa transkrypcji z YouTube i plików lokalnych
  - **Dubbing/Napisy**: Generowanie dubbingu TTS i wgrywanie napisów
  - **Pobieranie**: Pobieranie wideo i audio z YouTube

- **Nowe moduły**:
  - `gui.py` - Główny plik GUI oparty na Gradio
  - `data/gui/config.py` - Konfiguracja (modele, języki, głosy TTS)
  - `data/gui/handlers.py` - Handlery zdarzeń dla wszystkich zakładek
  - `data/validators.py` - Walidacja URL YouTube, plików wideo, plików SRT

- **Walidacja w czasie rzeczywistym**:
  - Sprawdzanie poprawności URL YouTube podczas wpisywania
  - Walidacja rozszerzeń plików lokalnych
  - Weryfikacja plików SRT

- **Dynamiczne formularze**:
  - Pokazywanie/ukrywanie pól w zależności od wyboru
  - Aktualizacja dropdownów (np. głosy TTS zależne od silnika)

### Zmieniono
- Rozszerzono `command_builders.py` o nowe funkcje dla GUI
- Zaktualizowano `tts_generator.py` dla lepszej integracji z GUI
- Zaktualizowano `validators.py` o nowe funkcje walidacji

### Naprawiono
- Poprawki w obsłudze plików tymczasowych w GUI
- Lepsze zarządzanie pamięcią CUDA w handlerach GUI

---

## [4.4.0] - 2025-01-23

### Dodano
- **Tryb lektora (narrator mode)**: Nowy tryb optymalizujący synchronizację dubbingu TTS
  - Flaga `--narrator-mode` aktywuje tryb lektora
  - Parametr `--merge-gap` kontroluje łączenie segmentów (50-300 ms)
- **Zarządzanie pamięcią GPU**: Funkcja `clear_cuda_cache()` do czyszczenia pamięci CUDA
- **Sprawdzanie VRAM**: Funkcja `check_vram_for_model()` weryfikuje dostępność pamięci
- **Predykcja overflowu**: Ostrzeżenia przed przepełnieniem VRAM

### Usunięto
- **faster-whisper**: Zastąpienie problematycznego faster-whisper przez OpenAI Whisper

### Zmieniono
- Rozszerzenie tłumacza o dodatkowe języki
- Poprawki dla modelu Coqui XTTS v2
- Nowy parametr `--speed` dla kontroli prędkości
- Nowy moduł `srt_reader.py` do wczytywania plików SRT

---

## [4.3.0] - 2025-01-15

### Dodano
- **Napisy dwujęzyczne (ASS)**: Flagi `--dual-language` do wgrywania napisów z oryginałem i tłumaczeniem
- **Format ASS**: Wsparcie dla plików ASS obok SRT
- **Nowe moduły**:
  - `ass_writer.py` - Generowanie dwujęzycznych napisów ASS
  - `warning_suppressor.py` - Tłumienie ostrzeżeń bibliotek trzecich

### Zmieniono
- Rozszerzenie architektury: 13 → 15 wyspecjalizowanych modułów
- Ulepszone `command_builders.py`: Automatyczne wykrywanie formatu napisów
- Ulepszone `audio_mixer.py`: Uniwersalna obsługa SRT i ASS

---

## [4.2.0] - 2025-01-10

### Dodano
- **Refaktoryzacja architektury**: Podział monolitu na 13 wyspecjalizowanych modułów
- **Moduły**: output_manager, command_builders, validators, youtube_processor, audio_processor,
  device_manager, transcription_engines, segment_processor, translation, srt_writer,
  tts_generator, audio_mixer, utils

---

## [4.1.0] - 2025-01-05

### Dodano
- **Coqui TTS**: Lokalny silnik TTS z modelami 100-500MB
- **Argumenty CLI**: `--tts-engine`, `--coqui-model`, `--coqui-speaker`
- **Rozszerzone języki Edge TTS**: Niemiecki, Francuski, Hiszpański, Włoski, Rosyjski,
  Japoński, Chiński, Koreański, Ukraiński, Czeski

---

## [4.0.0] - 2024-12-20

### Dodano
- **WhisperX**: Zaawansowany silnik z word-level alignment i speaker diarization
- **Refaktoryzacja**: `transcribe_chunk()` jako dispatcher do silników

### Zmieniono
- **Dockerfile**: Zmiana z devel (8-10GB) na runtime (3-4GB)
- **CUDA**: Aktualizacja do 12.4 + cuDNN 9 Runtime

---

## [3.2.0] - 2024-12-10

### Dodano
- **Pełna dockeryzacja**: CUDA 12.8 + cuDNN 9
- **GPU/CPU fallback**: Automatyczne wykrywanie w Docker
- **docker-compose**: Łatwe uruchamianie

---

## [3.1.0] - 2024-12-05

### Dodano
- **Wgrywanie napisów**: Flaga `--burn-subtitles`
- **Customizacja stylu**: Parametr `--subtitle-style`

---

## [3.0.0] - 2024-11-20

### Dodano
- **Dubbing TTS**: Microsoft Edge TTS
- **Pobieranie wideo**: Flaga `--download`
- **Wybór silnika**: Parametr `--engine`

---

## [2.0.0] - 2024-11-10

### Dodano
- **Pliki lokalne**: Flaga `--local`
- **Tłumaczenie**: Flaga `--translate`

---

## [1.0.0] - 2024-11-01

### Dodano
- Pierwsza wersja
- Podstawowa transkrypcja z YouTube
- Generowanie plików SRT
