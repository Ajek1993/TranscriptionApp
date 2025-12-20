# Dokumentacja funkcji - transcribe.py

Szczegółowy opis wszystkich funkcji w pliku `transcribe.py`.

## Klasy pomocnicze

### OutputManager

Centralizacja zarządzania komunikatami użytkownika.

**Metody:**

- `stage_header(stage_num: int, stage_name: str)` - Wyświetla nagłówek etapu
- `info(message: str, use_tqdm_safe: bool = False)` - Wyświetla komunikat informacyjny
- `success(message: str)` - Wyświetla komunikat sukcesu z checkmarkiem
- `warning(message: str, use_tqdm_safe: bool = False)` - Wyświetla ostrzeżenie
- `error(message: str)` - Wyświetla komunikat błędu
- `detail(message: str, use_tqdm_safe: bool = False)` - Wyświetla szczegóły (wcięcie)
- `mode_header(mode_name: str, details: dict = None)` - Wyświetla nagłówek trybu z konfiguracją

## Command Builders

Funkcje budujące komendy dla narzędzi zewnętrznych (ffmpeg, ffprobe, yt-dlp).

### build_ffprobe_audio_info_cmd(file_path: str) → list

Buduje komendę ffprobe do pobrania informacji o audio (kanały, sample rate).

**Parametry:**
- `file_path` - Ścieżka do pliku audio

**Zwraca:**
- Lista z komendą ffprobe

### build_ffprobe_video_info_cmd(file_path: str) → list

Buduje komendę ffprobe do pobrania informacji o wideo (szerokość, wysokość, kodek).

**Parametry:**
- `file_path` - Ścieżka do pliku wideo

**Zwraca:**
- Lista z komendą ffprobe

### build_ffprobe_duration_cmd(file_path: str) → list

Buduje komendę ffprobe do pobrania długości pliku.

**Parametry:**
- `file_path` - Ścieżka do pliku

**Zwraca:**
- Lista z komendą ffprobe

### build_ffmpeg_audio_extraction_cmd(input_path: str, output_path: str, sample_rate: int = 16000, channels: int = 1) → list

Buduje komendę ffmpeg do ekstrakcji audio jako WAV.

**Parametry:**
- `input_path` - Ścieżka pliku wejściowego
- `output_path` - Ścieżka pliku wyjściowego
- `sample_rate` - Częstotliwość próbkowania (domyślnie 16000)
- `channels` - Liczba kanałów (domyślnie 1 - mono)

**Zwraca:**
- Lista z komendą ffmpeg

### build_ffmpeg_audio_split_cmd(input_path: str, output_path: str, start_time: int, duration: int) → list

Buduje komendę ffmpeg do podziału audio na fragmenty.

**Parametry:**
- `input_path` - Ścieżka pliku wejściowego
- `output_path` - Ścieżka pliku wyjściowego
- `start_time` - Czas rozpoczęcia w sekundach
- `duration` - Długość fragmentu w sekundach

**Zwraca:**
- Lista z komendą ffmpeg

### build_ffmpeg_video_merge_cmd(video_path: str, audio_path: str, output_path: str) → list

Buduje komendę ffmpeg do połączenia wideo ze ścieżką audio.

**Parametry:**
- `video_path` - Ścieżka do pliku wideo
- `audio_path` - Ścieżka do pliku audio
- `output_path` - Ścieżka pliku wyjściowego

**Zwraca:**
- Lista z komendą ffmpeg

### build_ffmpeg_subtitle_burn_cmd(video_path: str, srt_path: str, output_path: str, subtitle_style: str) → list

Buduje komendę ffmpeg do wgrania napisów do wideo.

**Parametry:**
- `video_path` - Ścieżka do pliku wideo
- `srt_path` - Ścieżka do pliku SRT
- `output_path` - Ścieżka pliku wyjściowego
- `subtitle_style` - Styl napisów w formacie ASS

**Zwraca:**
- Lista z komendą ffmpeg

**Uwagi:**
- Automatycznie konwertuje ścieżki do formatu absolutnego
- Escape'uje znaki specjalne dla ffmpeg filter

### build_ytdlp_audio_download_cmd(url: str, output_file: str) → list

Buduje komendę yt-dlp do pobierania tylko audio.

**Parametry:**
- `url` - URL YouTube
- `output_file` - Ścieżka pliku wyjściowego

**Zwraca:**
- Lista z komendą yt-dlp

### build_ytdlp_video_download_cmd(url: str, output_file: str, quality: str = "1080") → list

Buduje komendę yt-dlp do pobierania wideo.

**Parametry:**
- `url` - URL YouTube
- `output_file` - Ścieżka pliku wyjściowego
- `quality` - Jakość wideo (720, 1080, 1440, 2160)

**Zwraca:**
- Lista z komendą yt-dlp

## Pipeline Functions

Funkcje głównych pipeline'ów przetwarzania.

### run_dubbing_pipeline(segments, audio_path, original_video_path, input_stem, args, temp_dir) → Tuple[bool, str]

Uruchamia kompletny pipeline dubbingu TTS.

**Obsługuje:**
- Generowanie TTS dla wszystkich segmentów
- Łączenie ścieżek TTS
- Mixowanie audio
- Tworzenie wideo (jeśli nie audio-only)

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `audio_path` - Ścieżka do oryginalnego audio
- `original_video_path` - Ścieżka do oryginalnego wideo (lub None)
- `input_stem` - Nazwa bazowa pliku wejściowego
- `args` - Argumenty CLI
- `temp_dir` - Katalog tymczasowy

**Zwraca:**
- Tuple (success, error_msg)

**Tryby:**
- `--dub-audio-only`: Tworzy tylko plik audio WAV
- `--dub`: Tworzy pełne wideo MP4 z dubbingiem

### generate_srt_output(segments, input_stem, args, temp_dir) → Tuple[bool, str, str]

Generuje plik SRT z segmentów transkrypcji.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `input_stem` - Nazwa bazowa pliku wejściowego
- `args` - Argumenty CLI
- `temp_dir` - Katalog tymczasowy

**Zwraca:**
- Tuple (success, error_msg, srt_filename)

**Logika nazewnictwa:**
- Jeśli `--output` podane: użyj tej nazwy
- Jeśli `--burn-subtitles`: zapisz w temp_dir
- Inaczej: zapisz w bieżącym katalogu

### burn_subtitles_to_video_pipeline(original_video_path, srt_filename, input_stem, args, temp_dir) → Tuple[bool, str]

Obsługuje kompletny pipeline wgrywania napisów do wideo.

**Parametry:**
- `original_video_path` - Ścieżka do oryginalnego wideo
- `srt_filename` - Ścieżka do pliku SRT
- `input_stem` - Nazwa bazowa pliku wejściowego
- `args` - Argumenty CLI
- `temp_dir` - Katalog tymczasowy

**Zwraca:**
- Tuple (success, error_msg)

**Uwagi:**
- Automatycznie pobiera wideo z YouTube jeśli brak lokalnego
- Wspiera customizację stylu napisów przez `--subtitle-style`

### run_transcription_pipeline(audio_path, args, temp_dir) → Tuple[bool, str, List[Tuple[int, int, str]]]

Uruchamia kompletny pipeline transkrypcji.

**Etapy:**
1. Wykrywanie długości audio
2. Podział na fragmenty jeśli > 30 minut
3. Transkrypcja każdego fragmentu
4. Scalanie segmentów z dostosowaniem timestampów
5. Opcjonalne tłumaczenie

**Parametry:**
- `audio_path` - Ścieżka do pliku audio WAV
- `args` - Argumenty CLI
- `temp_dir` - Katalog tymczasowy

**Zwraca:**
- Tuple (success, error_msg, segments)

**Optymalizacje:**
- Dla audio < 30 min: transkrypcja bez podziału
- Dla audio > 30 min: podział na fragmenty ~30 min każdy

## Input Processing

Funkcje przetwarzania źródeł wejściowych.

### process_input_source(args, temp_dir) → Tuple[bool, str, str, str, str]

Przetwarza źródło wejściowe (YouTube lub plik lokalny).

**Parametry:**
- `args` - Argumenty CLI
- `temp_dir` - Katalog tymczasowy

**Zwraca:**
- Tuple (success, error_msg, audio_path, original_video_path, input_stem)

**Logika:**
- Jeśli `args.local`: przetwarza plik lokalny
- Jeśli URL: przetwarza YouTube
- Inaczej: błąd

### handle_video_download_mode(args) → int

Obsługuje tryb pobierania wideo z YouTube (--download).

**Parametry:**
- `args` - Argumenty CLI

**Zwraca:**
- Kod wyjścia (0 sukces, 1 błąd)

**Funkcjonalność:**
- Pobiera pełne wideo z YouTube
- Wspiera jakości: 720p, 1080p, 1440p, 2160p
- Zapisuje jako MP4 w bieżącym katalogu

### handle_audio_download_mode(args) → int

Obsługuje tryb pobierania audio z YouTube (--download-audio-only).

**Parametry:**
- `args` - Argumenty CLI

**Zwraca:**
- Kod wyjścia (0 sukces, 1 błąd)

**Funkcjonalność:**
- Pobiera tylko audio z YouTube
- Konwertuje do WAV (mono, 16kHz)
- Zapisuje w bieżącym katalogu

### handle_test_merge_mode(args) → int

Obsługuje tryb testowy dla generowania SRT (--test-merge).

**Parametry:**
- `args` - Argumenty CLI

**Zwraca:**
- Kod wyjścia (0 sukces, 1 błąd)

**Funkcjonalność:**
- Tworzy przykładowe segmenty transkrypcji
- Testuje generowanie pliku SRT
- Używane podczas developmentu

## Validation Functions

Funkcje walidacji wejść i zależności.

### validate_youtube_url(url: str) → bool

Waliduje URL YouTube.

**Parametry:**
- `url` - URL do walidacji

**Zwraca:**
- True jeśli URL jest poprawny, False w przeciwnym razie

**Obsługiwane formaty:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

### validate_video_file(file_path: str) → Tuple[bool, str]

Waliduje czy plik wideo istnieje i ma wspierane rozszerzenie.

**Parametry:**
- `file_path` - Ścieżka do pliku

**Zwraca:**
- Tuple (is_valid, error_msg)

**Wspierane formaty:**
- MP4, MKV, AVI, MOV

### check_dependencies() → Tuple[bool, str]

Sprawdza czy wymagane narzędzia są zainstalowane.

**Wymagane narzędzia:**
- ffmpeg
- yt-dlp

**Zwraca:**
- Tuple (all_present, error_msg)

**Uwagi:**
- Uruchamia `ffmpeg -version` i `yt-dlp --version`
- Zwraca szczegółowy komunikat o brakujących narzędziach

### check_edge_tts_dependency() → Tuple[bool, str]

Sprawdza czy edge-tts jest zainstalowane.

**Zwraca:**
- Tuple (is_available, error_msg)

**Uwagi:**
- Wymagane tylko dla dubbingu TTS (--dub)

## Download Functions

Funkcje pobierania z YouTube i ekstrakcji audio.

### download_audio(url: str, output_dir: str = ".") → Tuple[bool, str, str]

Pobiera audio z YouTube jako WAV.

**Parametry:**
- `url` - URL YouTube
- `output_dir` - Katalog wyjściowy

**Zwraca:**
- Tuple (success, error_msg, wav_path)

**Format wyjściowy:**
- WAV, mono, 16kHz, PCM 16-bit
- Optymalizowany dla Whisper

**Retry:**
- 3 próby z 2s opóźnieniem
- Obsługa błędów sieciowych

### download_video(url: str, output_dir: str = ".", quality: str = "1080") → Tuple[bool, str, str]

Pobiera wideo z YouTube.

**Parametry:**
- `url` - URL YouTube
- `output_dir` - Katalog wyjściowy
- `quality` - Jakość wideo (720, 1080, 1440, 2160)

**Zwraca:**
- Tuple (success, error_msg, video_path)

**Format wyjściowy:**
- MP4 z najlepszym dostępnym kodekiem
- Merge wideo + audio

**Retry:**
- 3 próby z 2s opóźnieniem

### extract_audio_from_video(video_path: str, output_dir: str = ".") → Tuple[bool, str, str]

Ekstrahuje audio z pliku wideo lokalnego.

**Parametry:**
- `video_path` - Ścieżka do pliku wideo
- `output_dir` - Katalog wyjściowy

**Zwraca:**
- Tuple (success, error_msg, wav_path)

**Format wyjściowy:**
- WAV, mono, 16kHz, PCM 16-bit

**Obsługiwane formaty:**
- MP4, MKV, AVI, MOV

## Audio Processing

Funkcje przetwarzania audio.

### get_audio_duration(wav_path: str) → Tuple[bool, float]

Pobiera długość audio w sekundach.

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV

**Zwraca:**
- Tuple (success, duration_seconds)

**Metoda:**
- Używa ffprobe z parsowaniem JSON

### get_audio_duration_ms(audio_path: str) → Tuple[bool, int]

Pobiera długość audio w milisekundach.

**Parametry:**
- `audio_path` - Ścieżka do pliku audio

**Zwraca:**
- Tuple (success, duration_ms)

**Uwagi:**
- Precyzyjniejsze dla synchronizacji TTS

### split_audio(wav_path: str, chunk_duration_sec: int = 1800, output_dir: str = ".") → Tuple[bool, str, List[str]]

Dzieli długie audio na fragmenty.

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV
- `chunk_duration_sec` - Długość fragmentu w sekundach (domyślnie 1800 = 30 min)
- `output_dir` - Katalog wyjściowy

**Zwraca:**
- Tuple (success, error_msg, chunk_paths)

**Optymalizacja:**
- Dla audio < 30 min: zwraca oryginalny plik bez podziału
- Dla audio > 30 min: dzieli na fragmenty ~30 min każdy

**Format fragmentów:**
- WAV, mono, 16kHz

## Transcription Functions

Funkcje transkrypcji audio.

### detect_device() → Tuple[str, str]

Wykrywa dostępne urządzenie (GPU/CPU) dla transkrypcji.

**Zwraca:**
- Tuple (device, compute_type)

**Urządzenia:**
- `cuda` + `float16` - jeśli GPU dostępne
- `cpu` + `int8` - jeśli brak GPU

**Uwagi:**
- Automatyczny fallback na CPU
- Wspiera CUDA dla NVIDIA GPU

### get_gpu_memory_info() → str

Pobiera informacje o pamięci GPU.

**Zwraca:**
- String z informacjami o GPU lub komunikat o błędzie

**Metoda:**
- Używa PyTorch do sprawdzenia CUDA memory

### transcribe_with_whisper(wav_path, model_size, language, segment_progress_bar, timeout_seconds) → Tuple[bool, str, List[Tuple[int, int, str]]]

Transkrybuje audio przy użyciu OpenAI Whisper.

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV
- `model_size` - Rozmiar modelu Whisper (tiny/base/small/medium/large)
- `language` - Język transkrypcji (pl, en, etc.)
- `segment_progress_bar` - Progress bar dla segmentów
- `timeout_seconds` - Timeout w sekundach (0 = brak timeoutu)

**Zwraca:**
- Tuple (success, error_msg, segments)

**Funkcjonalność:**
- Automatyczne wykrywanie GPU/CUDA
- Fallback na CPU jeśli GPU niedostępne
- Timeout handling
- Progress tracking
- Optymalizuje dla GPU z float16

**Zalety:**
- Szybki (GPU-accelerated)
- Dobra jakość
- Wbudowany fallback na CPU

### transcribe_with_faster_whisper(wav_path, model_size, language, segment_progress_bar, timeout_seconds) → Tuple[bool, str, List[Tuple[int, int, str]]]

Transkrybuje audio przy użyciu Faster-Whisper (CPU-only).

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV
- `model_size` - Rozmiar modelu (tiny/base/small/medium/large)
- `language` - Język transkrypcji (pl, en, etc.)
- `segment_progress_bar` - Progress bar dla segmentów
- `timeout_seconds` - Timeout w sekundach (0 = brak timeoutu)

**Zwraca:**
- Tuple (success, error_msg, segments)

**Funkcjonalność:**
- Wymuszony CPU (bez CUDA)
- Timeout handling
- Progress tracking
- Optymalizuje dla CPU z int8

**Zalety:**
- Działa bez GPU
- Mniejsze zużycie pamięci

**Wady:**
- Wolniejszy niż Whisper
- Wymaga więcej czasu obliczeniowego

### transcribe_with_whisperx(wav_path, model_size, language, segment_progress_bar, timeout_seconds, align=False, diarize=False, min_speakers=None, max_speakers=None, hf_token=None) → Tuple[bool, str, List[Tuple[int, int, str]]]

Transkrybuje audio przy użyciu WhisperX (zaawansowany).

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV
- `model_size` - Rozmiar modelu (tiny/base/small/medium/large)
- `language` - Język transkrypcji (pl, en, etc.)
- `segment_progress_bar` - Progress bar dla segmentów
- `timeout_seconds` - Timeout w sekundach (0 = brak timeoutu)
- `align` - Włączaj word-level alignment (domyślnie False)
- `diarize` - Włączaj speaker diarization (domyślnie False)
- `min_speakers` - Minimalna liczba mówców (opcjonalnie)
- `max_speakers` - Maksymalna liczba mówców (opcjonalnie)
- `hf_token` - HuggingFace token dla diarization (opcjonalnie)

**Zwraca:**
- Tuple (success, error_msg, segments)

**Funkcjonalność:**
- Automatyczne wykrywanie GPU/CUDA
- Word-level alignment (dokładne timestampy)
- Speaker diarization (rozpoznawanie mówców)
- Timeout handling
- Progress tracking
- Fallback na CPU jeśli GPU niedostępne

**Zaawansowane opcje:**
- `--whisperx-align`: Włącza dokładne timestampy na poziomie słów
- `--whisperx-diarize`: Włącza rozpoznawanie mówców (wymaga HF token)
- `--whisperx-min-speakers`, `--whisperx-max-speakers`: Ograniczenia liczby mówców

**Zalety:**
- Najdokładniejsze timestampy
- Speaker diarization
- Najwyższa jakość
- Automatyczne GPU/CPU

**Wady:**
- Wolniejszy niż Whisper
- Diarization wymaga HuggingFace token

### transcribe_chunk(wav_path, model_size, language, engine, segment_progress_bar, timeout_seconds, whisperx_align=False, whisperx_diarize=False, whisperx_min_speakers=None, whisperx_max_speakers=None, hf_token=None) → Tuple[bool, str, List[Tuple[int, int, str]]]

Transkrybuje pojedynczy fragment audio (dispatcher do silników).

**Parametry:**
- `wav_path` - Ścieżka do pliku WAV
- `model_size` - Rozmiar modelu (tiny/base/small/medium/large)
- `language` - Język transkrypcji (pl, en, etc.)
- `engine` - Silnik transkrypcji (whisper, faster-whisper, whisperx)
- `segment_progress_bar` - Progress bar dla segmentów
- `timeout_seconds` - Timeout w sekundach (0 = brak timeoutu)
- `whisperx_align` - Włącz word-level alignment (tylko WhisperX)
- `whisperx_diarize` - Włącz speaker diarization (tylko WhisperX)
- `whisperx_min_speakers` - Minimalna liczba mówców (opcjonalnie)
- `whisperx_max_speakers` - Maksymalna liczba mówców (opcjonalnie)
- `hf_token` - HuggingFace token (opcjonalnie)

**Zwraca:**
- Tuple (success, error_msg, segments)

**Funkcjonalność (Dispatcher):**
- Wybiera odpowiedni silnik transkrypcji na podstawie `engine`
- Kieruje parametry do engine-specific funkcji
- `whisper` → transcribe_with_whisper()
- `faster-whisper` → transcribe_with_faster_whisper()
- `whisperx` → transcribe_with_whisperx()
- Wspólne dla wszystkich silników: timeout handling, progress tracking

**Silniki dostępne:**
- `whisper` (domyślny): GPU-accelerated, szybki, dobra jakość
- `faster-whisper`: CPU-only, wolniejszy, brak GPU
- `whisperx`: Zaawansowany, najdokładniejsze timestampy, speaker diarization

**Backward compatibility:**
- Domyślnie używa `whisper` jeśli `engine` nie podany
- WhisperX parametry są opcjonalne (ignorowane dla innych silników)

## Segment Processing

Funkcje przetwarzania segmentów transkrypcji.

### split_long_segments(segments, max_duration_sec, max_words) → List[Tuple[int, int, str]]

Dzieli długie segmenty na krótsze dla lepszej synchronizacji TTS.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `max_duration_sec` - Maksymalna długość segmentu w sekundach
- `max_words` - Maksymalna liczba słów w segmencie

**Zwraca:**
- Lista podzielonych segmentów

**Strategia podziału:**
1. Dzieli po znakach interpunkcyjnych (., !, ?)
2. Jeśli nadal za długi: dzieli po przecinkach
3. Jeśli nadal za długi: dzieli po spacji (maks. słów)

**Uwagi:**
- Zachowuje proporcjonalne timestampy
- Używane przed generowaniem TTS

### fill_timestamp_gaps(segments, min_pause_ms, max_gap_fill_ms) → List[Tuple[int, int, str]]

Wypełnia luki w timestampach dla płynniejszego dubbingu.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `min_pause_ms` - Minimalna pauza między segmentami (domyślnie 300ms)
- `max_gap_fill_ms` - Maksymalna luka do wypełnienia (domyślnie 2000ms)

**Zwraca:**
- Lista segmentów z wypełnionymi lukami

**Strategia:**
- Dla luk < max_gap_fill_ms: rozszerza poprzedni segment
- Dla luk > max_gap_fill_ms: zostawia lukę
- Zachowuje minimalną pauzę między segmentami

## Translation Functions

Funkcje tłumaczenia napisów.

### translate_segments(segments, direction) → Tuple[bool, str, List[Tuple[int, int, str]]]

Tłumaczy teksty segmentów zachowując timestampy.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `direction` - Kierunek tłumaczenia (pl-en, en-pl)

**Zwraca:**
- Tuple (success, error_msg, translated_segments)

**Funkcjonalność:**
- Używa Google Translate przez deep-translator
- Progress bar dla tłumaczenia
- Retry mechanizm (3 próby)
- Zachowuje timestampy
- Fallback na oryginalny tekst przy błędzie

**Obsługiwane kierunki:**
- `pl-en`: Polski → Angielski
- `en-pl`: Angielski → Polski

## SRT Functions

Funkcje generowania plików SRT.

### format_srt_timestamp(ms: int) → str

Formatuje milisekundy do formatu SRT timestamp.

**Parametry:**
- `ms` - Czas w milisekundach

**Zwraca:**
- String w formacie `HH:MM:SS,mmm`

**Przykład:**
- 125000 ms → "00:02:05,000"

### write_srt(segments, output_path) → Tuple[bool, str]

Zapisuje segmenty do pliku SRT.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `output_path` - Ścieżka pliku wyjściowego

**Zwraca:**
- Tuple (success, error_msg)

**Format SRT:**
```
1
00:00:00,000 --> 00:00:05,000
Tekst pierwszego napisu

2
00:00:05,500 --> 00:00:10,000
Tekst drugiego napisu
```

**Uwagi:**
- Kodowanie UTF-8 z BOM
- Zgodny ze standardem SRT
- Wspiera polskie znaki

## TTS Functions

Funkcje generowania dubbingu TTS.

### generate_tts_for_segment(segment_index, start_ms, end_ms, text, output_dir, voice) → Tuple[bool, str, str, int]

Generuje TTS dla pojedynczego segmentu (async).

**Parametry:**
- `segment_index` - Numer segmentu
- `start_ms` - Timestamp rozpoczęcia
- `end_ms` - Timestamp zakończenia
- `text` - Tekst do przekształcenia na mowę
- `output_dir` - Katalog wyjściowy
- `voice` - Głos TTS (np. pl-PL-MarekNeural)

**Zwraca:**
- Tuple (success, error_msg, tts_file_path, duration_ms)

**Funkcjonalność:**
- Generuje MP3 z Microsoft Edge TTS
- Automatyczne przyspieszanie jeśli TTS > slot czasowy
- Max przyspieszenie: +50%
- Retry mechanizm (3 próby z exponential backoff)

**Uwagi:**
- Wymaga połączenia internetowego
- Asynchroniczna (async/await)

### generate_tts_segments(segments, output_dir, voice) → Tuple[bool, str, List[Tuple[int, int, str, int]]]

Generuje TTS dla wszystkich segmentów z progress barem.

**Parametry:**
- `segments` - Lista segmentów (start_ms, end_ms, text)
- `output_dir` - Katalog wyjściowy
- `voice` - Głos TTS

**Zwraca:**
- Tuple (success, error_msg, tts_files)

**Format tts_files:**
- Lista tuple (start_ms, end_ms, tts_file_path, duration_ms)

**Funkcjonalność:**
- Progress bar z licznikiem segmentów
- Asynchroniczne przetwarzanie
- Raportowanie błędów z kontekstem

### create_tts_audio_track(tts_files, total_duration_ms, output_path) → Tuple[bool, str]

Łączy segmenty TTS w jedną ścieżkę audio z synchronizacją czasową.

**Parametry:**
- `tts_files` - Lista (start_ms, end_ms, tts_file_path, duration_ms)
- `total_duration_ms` - Całkowita długość ścieżki
- `output_path` - Ścieżka pliku wyjściowego

**Zwraca:**
- Tuple (success, error_msg)

**Strategia:**
- Używa ffmpeg concat filter
- Dla każdego segmentu: cisza → TTS → cisza
- Precyzyjna synchronizacja timestampów
- Wypełnia końcową ciszę do total_duration

**Format wyjściowy:**
- WAV, stereo, 44.1kHz

### mix_audio_tracks(original_audio_path, tts_audio_path, output_path, original_volume, tts_volume) → Tuple[bool, str]

Mixuje oryginalną ścieżkę audio z TTS.

**Parametry:**
- `original_audio_path` - Ścieżka do oryginalnego audio
- `tts_audio_path` - Ścieżka do TTS audio
- `output_path` - Ścieżka pliku wyjściowego
- `original_volume` - Głośność oryginału (0.0-1.0, domyślnie 0.2)
- `tts_volume` - Głośność TTS (0.0-2.0, domyślnie 1.0)

**Zwraca:**
- Tuple (success, error_msg)

**Funkcjonalność:**
- Ścisza oryginał (domyślnie do 20%)
- Nakłada TTS w pełnej głośności
- Używa ffmpeg volume filter

**Format wyjściowy:**
- WAV, stereo, 44.1kHz

## Video Functions

Funkcje tworzenia i modyfikacji wideo.

### create_dubbed_video(original_video_path, mixed_audio_path, output_path) → Tuple[bool, str]

Tworzy wideo z dubbingiem (oryginalne wideo + zmixowane audio).

**Parametry:**
- `original_video_path` - Ścieżka do oryginalnego wideo
- `mixed_audio_path` - Ścieżka do zmixowanego audio
- `output_path` - Ścieżka pliku wyjściowego

**Zwraca:**
- Tuple (success, error_msg)

**Funkcjonalność:**
- Kopiuje strumień wideo (bez rekodowania)
- Zastępuje ścieżkę audio
- Kodek audio: AAC, 192 kbps

**Format wyjściowy:**
- MP4

### burn_subtitles_to_video(video_path, srt_path, output_path, subtitle_style) → Tuple[bool, str]

Wgrywa napisy na stałe do wideo (hardcode subtitles).

**Parametry:**
- `video_path` - Ścieżka do wideo
- `srt_path` - Ścieżka do pliku SRT
- `output_path` - Ścieżka pliku wyjściowego
- `subtitle_style` - Styl napisów w formacie ASS

**Zwraca:**
- Tuple (success, error_msg)

**Funkcjonalność:**
- Re-enkoduje wideo z napisami
- Kodek: H.264, CRF 23, preset medium
- Customizacja stylu (czcionka, kolor, tło)

**Domyślny styl:**
- Białe napisy
- Półprzezroczyste ciemne tło
- Czcionka Arial, 24px

**Format wyjściowy:**
- MP4

## Utility Functions

Funkcje pomocnicze.

### cleanup_temp_files(temp_dir: str, retries: int = 3, delay: float = 0.2) → None

Usuwa katalog tymczasowy z retry mechaniką.

**Parametry:**
- `temp_dir` - Ścieżka do katalogu tymczasowego
- `retries` - Liczba prób (domyślnie 3)
- `delay` - Opóźnienie między próbami w sekundach

**Funkcjonalność:**
- Retry mechanizm dla Windows file locking
- Cicha obsługa błędów
- Logowanie błędów

**Uwagi:**
- Windows może blokować pliki przez chwilę po zakończeniu ffmpeg

### main()

Główna funkcja aplikacji.

**Funkcjonalność:**
1. Parsowanie argumentów CLI
2. Wykrywanie trybu (download, transcribe, dub, etc.)
3. Walidacja zależności
4. Uruchomienie odpowiedniego pipeline'u
5. Cleanup plików tymczasowych

**Obsługiwane tryby:**
- `--download`: Pobieranie wideo z YouTube
- `--download-audio-only`: Pobieranie audio z YouTube
- `--test-merge`: Test generowania SRT
- Normalny tryb: Transkrypcja + opcjonalnie dubbing/subtitle burning

**Obsługa błędów:**
- Graceful shutdown
- Cleanup nawet przy błędach
- Szczegółowe komunikaty błędów

---

## Struktura kodu

### Główne etapy przetwarzania

1. **Etap 1: Walidacja i pobieranie** (`process_input_source`)
   - Walidacja URL/pliku lokalnego
   - Sprawdzenie zależności
   - Pobieranie audio/wideo

2. **Etap 2: Podział audio** (`run_transcription_pipeline`)
   - Wykrywanie długości
   - Podział na fragmenty jeśli > 30 min

3. **Etap 3: Transkrypcja** (`_transcribe_all_chunks`)
   - Wykrywanie GPU/CPU
   - Transkrypcja z progress bars
   - Scalanie segmentów

4. **Etap 4: Generowanie SRT** (`generate_srt_output`)
   - Opcjonalne tłumaczenie
   - Zapis do pliku SRT

5. **Etap 5: Dubbing TTS** (`run_dubbing_pipeline`, opcjonalny)
   - Generowanie TTS
   - Łączenie segmentów
   - Mixowanie audio
   - Tworzenie wideo

6. **Etap 6: Subtitle Burning** (`burn_subtitles_to_video_pipeline`, opcjonalny)
   - Wgrywanie napisów do wideo
   - Customizacja stylu

### Dependency Graph

```
main()
├── handle_video_download_mode()
│   └── download_video()
├── handle_audio_download_mode()
│   └── download_audio()
├── handle_test_merge_mode()
│   └── write_srt()
└── Normal mode
    ├── check_dependencies()
    ├── process_input_source()
    │   ├── _process_local_file()
    │   │   └── extract_audio_from_video()
    │   └── _process_youtube_url()
    │       ├── download_audio()
    │       └── download_video() (jeśli --dub)
    ├── run_transcription_pipeline()
    │   ├── split_audio()
    │   ├── _transcribe_all_chunks()
    │   │   └── transcribe_chunk()
    │   │       ├── split_long_segments()
    │   │       └── fill_timestamp_gaps()
    │   └── _translate_segments_if_requested()
    │       └── translate_segments()
    ├── generate_srt_output()
    │   └── write_srt()
    ├── run_dubbing_pipeline() (jeśli --dub)
    │   ├── generate_tts_segments()
    │   │   └── generate_tts_for_segment()
    │   ├── create_tts_audio_track()
    │   ├── mix_audio_tracks()
    │   └── create_dubbed_video()
    ├── burn_subtitles_to_video_pipeline() (jeśli --burn-subtitles)
    │   └── burn_subtitles_to_video()
    └── cleanup_temp_files()
```

### Zarządzanie plikami tymczasowymi

**Lokalizacja:** System temp directory (`tempfile.mkdtemp()`)

**Struktura:**
```
temp_XXXXXX/
├── audio.wav              # Wyekstrahowane audio
├── chunk_001.wav          # Fragmenty audio (jeśli długie)
├── chunk_002.wav
├── input_stem.srt         # SRT (jeśli --burn-subtitles)
├── tts/                   # Katalog TTS
│   ├── segment_001.mp3
│   ├── segment_002.mp3
│   └── ...
├── tts_combined.wav       # Połączone TTS
└── mixed_audio.wav        # Zmixowane audio
```

**Cleanup:**
- Automatyczny na końcu pipeline'u
- Retry mechanizm dla Windows
- Logowanie błędów
