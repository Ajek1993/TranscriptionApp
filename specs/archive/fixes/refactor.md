# Plan Refaktoryzacji transcribe.py

## Cel

Rozbić monolityczny plik `transcribe.py` (2890 linii) na mniejsze, wyspecjalizowane moduły. Plik `transcribe.py` powinien zawierać tylko parsowanie argumentów CLI i orchestrację pipeline'ów.

## Wybrana Strategia

- **Struktura**: Płaska w folderze `data/` (wszystkie moduły obok `transcribe.py`)
- **OutputManager**: Wydzielony do osobnego modułu
- **Zakres**: Konserwatywny - 13 modułów tematycznych, minimalizacja ryzyka

## Obecna Struktura

```
data/transcribe.py - 2890 linii
├── 1 klasa: OutputManager
├── 46 funkcji:
│   ├── Command builders (9 funkcji): ffmpeg/yt-dlp
│   ├── Pipeline'y (5 funkcji): dubbing, transcription, SRT
│   ├── Transkrypcja (4 funkcje): Whisper/Faster/WhisperX
│   ├── TTS (4 funkcje): Edge TTS + Coqui
│   ├── Wideo (3 funkcje): YouTube download, extract audio
│   ├── Audio (3 funkcje): duration, split
│   ├── Segmenty (3 funkcje): split, fill gaps, format
│   ├── Walidacja (5 funkcji): URL, files, dependencies
│   ├── Tłumaczenie (1 funkcja)
│   ├── SRT (1 funkcja)
│   ├── Miksowanie (3 funkcje): mix, create video, burn subs
│   ├── Utils (2 funkcje): cleanup, device detection
│   └── Main (1 funkcja): CLI + orchestration
└── Stałe: SUPPORTED_VIDEO_EXTENSIONS, *_AVAILABLE, XTTS_SUPPORTED_LANGUAGES
```

## Docelowa Struktura

```
data/
├── transcribe.py              # CLI + orchestration (~400 linii)
├── output_manager.py          # OutputManager class
├── command_builders.py        # Command builders (ffmpeg/yt-dlp)
├── validators.py              # Walidacja + sprawdzanie dependencies
├── youtube_processor.py       # YouTube download + extract audio
├── audio_processor.py         # Audio operations (duration, split)
├── device_manager.py          # GPU/CPU detection
├── transcription_engines.py   # Whisper/Faster-Whisper/WhisperX
├── segment_processor.py       # Split, fill gaps, format timestamps
├── translation.py             # Google Translator
├── srt_writer.py              # SRT generation
├── tts_generator.py           # Edge TTS + Coqui TTS
├── audio_mixer.py             # Mix audio + create video + burn subs
└── utils.py                   # Cleanup temp files
```

## Mapa Zależności (DAG - brak cykli)

```
transcribe.py (główny orchestrator)
    ├─→ output_manager.py (niezależny)
    ├─→ validators.py (niezależny)
    ├─→ command_builders.py (niezależny)
    ├─→ utils.py (niezależny)
    ├─→ device_manager.py (niezależny)
    ├─→ segment_processor.py (niezależny)
    ├─→ srt_writer.py → segment_processor
    ├─→ youtube_processor.py → command_builders, output_manager
    ├─→ audio_processor.py → command_builders
    ├─→ translation.py (niezależny)
    ├─→ transcription_engines.py → device_manager, output_manager
    ├─→ tts_generator.py → device_manager, audio_processor
    └─→ audio_mixer.py → command_builders
```

## Szczegółowy Plan Modułów

### 1. output_manager.py (~120 linii)

**Funkcje/klasy:**

- `OutputManager` class (linie 64-115 z transcribe.py)

**Interfejs publiczny:**

- `OutputManager.stage_header(stage_num, stage_name)`
- `OutputManager.info(message, use_tqdm_safe=False)`
- `OutputManager.success(message)`
- `OutputManager.warning(message, use_tqdm_safe=False)`
- `OutputManager.error(message)`
- `OutputManager.detail(message, use_tqdm_safe=False)`
- `OutputManager.mode_header(mode_name, details=None)`

**Zależności:** Brak (tylko stdlib: tqdm)

---

### 2. command_builders.py (~180 linii)

**Funkcje:**

- `build_ffprobe_audio_info_cmd(file_path) -> list`
- `build_ffprobe_video_info_cmd(file_path) -> list`
- `build_ffprobe_duration_cmd(file_path) -> list`
- `build_ffmpeg_audio_extraction_cmd(input_path, output_path, ...) -> list`
- `build_ffmpeg_audio_split_cmd(input_path, output_dir, ...) -> list`
- `build_ffmpeg_video_merge_cmd(video_path, audio_path, output_path) -> list`
- `build_ffmpeg_subtitle_burn_cmd(video_path, srt_path, output_path) -> list`
- `build_ytdlp_audio_download_cmd(url, output_template) -> list`
- `build_ytdlp_video_download_cmd(url, output_template, quality) -> list`

**Zależności:** Brak (tylko stdlib: pathlib)

---

### 3. validators.py (~150 linii)

**Funkcje:**

- `validate_youtube_url(url) -> bool`
- `validate_video_file(file_path) -> Tuple[bool, str]`
- `check_dependencies() -> Tuple[bool, str]`
- `check_edge_tts_dependency() -> Tuple[bool, str]`
- `check_coqui_tts_dependency() -> Tuple[bool, str]`

**Stałe:**

- `SUPPORTED_VIDEO_EXTENSIONS`
- `TRANSLATOR_AVAILABLE`
- `EDGE_TTS_AVAILABLE`
- `COQUI_TTS_AVAILABLE`

**Zależności:** Conditional imports (deep_translator, edge_tts, TTS)

---

### 4. youtube_processor.py (~280 linii)

**Funkcje:**

- `download_audio(url, output_dir) -> Tuple[bool, str, str]`
- `download_video(url, output_dir, quality='best') -> Tuple[bool, str, str]`
- `extract_audio_from_video(video_path, output_dir) -> Tuple[bool, str, str]`

**Importy:**

```python
from command_builders import (
    build_ytdlp_audio_download_cmd,
    build_ytdlp_video_download_cmd,
    build_ffprobe_audio_info_cmd,
    build_ffprobe_video_info_cmd,
    build_ffmpeg_audio_extraction_cmd
)
from output_manager import OutputManager
```

---

### 5. audio_processor.py (~280 linii)

**Funkcje:**

- `get_audio_duration(wav_path) -> Tuple[bool, float]`
- `get_audio_duration_ms(audio_path) -> Tuple[bool, int]`
- `split_audio(wav_path, chunk_duration_sec, output_dir) -> Tuple[bool, str, List[str]]`

**Importy:**

```python
from command_builders import build_ffprobe_duration_cmd, build_ffmpeg_audio_split_cmd
from tqdm import tqdm
```

---

### 6. device_manager.py (~120 linii)

**Funkcje:**

- `get_gpu_memory_info() -> str`
- `detect_device(force_device='auto') -> Tuple[str, str]`

**Stałe:**

- `WHISPER_MODEL_MEMORY_REQUIREMENTS`

**Zależności:** torch (conditional import)

---

### 7. transcription_engines.py (~700 linii) - KLUCZOWY

**Funkcje:**

- `transcribe_with_whisper(...) -> Tuple[bool, str, List[Tuple[int, int, str]]]`
- `transcribe_with_faster_whisper(...) -> Tuple[bool, str, List[Tuple[int, int, str]]]`
- `transcribe_with_whisperx(...) -> Tuple[bool, str, List[Tuple[int, int, str]]]`
- `transcribe_chunk(...) -> Tuple[bool, str, List[Tuple[int, int, str]]]`
- `_run_transcription_with_timeout(...) -> Tuple[bool, str, List]` (prywatna)

**Importy:**

```python
from device_manager import detect_device, get_gpu_memory_info
from output_manager import OutputManager
# Conditional: whisper, faster_whisper, whisperx, torch
```

**Uwaga:** `_transcribe_all_chunks()` może pozostać w `transcribe.py` jako część pipeline

---

### 8. segment_processor.py (~250 linii)

**Funkcje:**

- `split_long_segments(segments, max_length_ms=10000) -> List[Tuple[int, int, str]]`
- `fill_timestamp_gaps(segments, total_duration_ms) -> List[Tuple[int, int, str]]`
- `format_srt_timestamp(milliseconds) -> str`

**Zależności:** Brak (tylko stdlib + typing)

---

### 9. translation.py (~100 linii)

**Funkcje:**

- `translate_segments(segments, source_lang, target_lang, batch_size=50) -> Tuple[bool, str, List[Tuple[int, int, str]]]`

**Zależności:** deep_translator (conditional), tqdm

---

### 10. srt_writer.py (~100 linii)

**Funkcje:**

- `write_srt(segments, output_path) -> Tuple[bool, str]`

**Importy:**

```python
from segment_processor import format_srt_timestamp
```

---

### 11. tts_generator.py (~550 linii) - KLUCZOWY

**Funkcje:**

- `generate_tts_segments(segments, output_dir, voice, engine, ...) -> Tuple[bool, str, List[Tuple[int, int, str, float]]]`
- `create_tts_audio_track(tts_files, total_duration_ms, output_path) -> Tuple[bool, str]`
- `determine_tts_target_language(transcription_language, translation_spec) -> str`
- `_generate_tts_coqui_for_segment(...)` (prywatna)

**Stałe:**

- `XTTS_SUPPORTED_LANGUAGES`

**Importy:**

```python
from device_manager import detect_device
from audio_processor import get_audio_duration
# Conditional: edge_tts, asyncio, TTS
```

---

### 12. audio_mixer.py (~250 linii)

**Funkcje:**

- `mix_audio_tracks(original_audio, tts_audio, output_path, tts_volume_ratio=1.2, original_volume_ratio=0.3) -> Tuple[bool, str]`
- `create_dubbed_video(video_path, dubbed_audio_path, output_path) -> Tuple[bool, str]`
- `burn_subtitles_to_video(video_path, srt_path, output_path) -> Tuple[bool, str]`

**Importy:**

```python
from command_builders import build_ffmpeg_video_merge_cmd, build_ffmpeg_subtitle_burn_cmd
```

---

### 13. utils.py (~50 linii)

**Funkcje:**

- `cleanup_temp_files(temp_dir, retries=3, delay=0.2) -> None`

**Zależności:** shutil, time, pathlib

---

### 14. transcribe.py (docelowo ~400 linii)

**Pozostają:**

- `main()` - parsowanie CLI (~290 linii)
- Pipeline orchestrators:
  - `run_transcription_pipeline()` (~150 linii)
  - `run_dubbing_pipeline()` (~140 linii)
  - `generate_srt_output()` (~35 linii)
  - `burn_subtitles_to_video_pipeline()` (~50 linii)
  - `process_input_source()` (~20 linii)
- Special modes:
  - `handle_video_download_mode()` (~35 linii)
  - `handle_audio_download_mode()` (~35 linii)
  - `handle_test_merge_mode()` (~55 linii)
- Helpers:
  - `_transcribe_all_chunks()` (~90 linii)
  - `_translate_segments_if_requested()` (~35 linii)

**Wszystkie importy z nowych modułów:**

```python
from output_manager import OutputManager
from command_builders import *
from validators import (
    validate_youtube_url, validate_video_file,
    check_dependencies, check_edge_tts_dependency,
    check_coqui_tts_dependency, TRANSLATOR_AVAILABLE
)
from youtube_processor import download_audio, download_video, extract_audio_from_video
from audio_processor import get_audio_duration, get_audio_duration_ms, split_audio
from device_manager import detect_device
from transcription_engines import transcribe_chunk
from segment_processor import split_long_segments, fill_timestamp_gaps
from translation import translate_segments
from srt_writer import write_srt
from tts_generator import (
    generate_tts_segments, create_tts_audio_track,
    determine_tts_target_language
)
from audio_mixer import mix_audio_tracks, create_dubbed_video, burn_subtitles_to_video
from utils import cleanup_temp_files
```

## Kolejność Implementacji

### FAZA 1: Podstawowe moduły (niskie ryzyko)

1. **output_manager.py** - Wydziel OutputManager class
2. **command_builders.py** - Wydziel 9 funkcji build\_\*
3. **utils.py** - Wydziel cleanup_temp_files
4. **validators.py** - Wydziel walidację + stałe

**Test po Fazie 1:**

```bash
python data/transcribe.py --help
python data/transcribe.py --test-merge
```

### FAZA 2: Moduły przetwarzania (średnie ryzyko)

5. **segment_processor.py** - Split/fill/format funkcje
6. **srt_writer.py** - write_srt + import z segment_processor
7. **device_manager.py** - detect_device + GPU info
8. **audio_processor.py** - Duration + split audio

**Test po Fazie 2:**

```bash
python data/transcribe.py "https://youtube.com/..." --only-download
```

### FAZA 3: Zewnętrzne źródła (średnie ryzyko)

9. **youtube_processor.py** - Download/extract funkcje
10. **translation.py** - translate_segments

**Test po Fazie 3:**

```bash
python data/transcribe.py "https://youtube.com/..." --model tiny --translate pl-en
```

### FAZA 4: Główna logika (wysokie ryzyko)

11. **transcription_engines.py** - Wszystkie silniki transkrypcji
12. **tts_generator.py** - Edge TTS + Coqui
13. **audio_mixer.py** - Mix/merge/burn funkcje

**Test po Fazie 4:**

```bash
# Pełny pipeline transkrypcji
python data/transcribe.py "https://youtube.com/..." --model base

# Pełny dubbing
python data/transcribe.py --local test.mp4 --dub --tts-engine edge
```

### FAZA 5: Cleanup

14. Usuń wszystkie przeniesione funkcje z transcribe.py
15. Dodaj docstringi do każdego nowego modułu
16. Test pełnej integracji

**Test finalny:**

```bash
python data/transcribe.py "https://youtube.com/..." --model base --translate pl-en --dub --burn-subtitles
```

## Potencjalne Pułapki i Rozwiązania

### 1. Circular Dependencies

**Rozwiązanie:** Mapa zależności jest DAG (acykliczna). Base modules nie importują nic z projektu.

### 2. Globalne stałe (\*\_AVAILABLE)

**Rozwiązanie:** Wszystkie w `validators.py`, importowane gdzie potrzebne.

### 3. OutputManager wszędzie używany

**Rozwiązanie:** Pierwszy moduł do wydzielenia, potem wszystkie inne mogą importować.

### 4. Conditional imports (TTS, translator)

**Rozwiązanie:** Conditional imports TYLKO w `validators.py`. W modułach TTS/translation zakładamy dostępność bibliotek.

### 5. Importy względne

**Rozwiązanie:** Użyj prostych importów (same-directory):

```python
from output_manager import OutputManager
```

### 6. Windows paths

**Rozwiązanie:** `pathlib.Path` konsekwentnie, `.replace('\\', '/')` tylko dla ffmpeg filters.

## Backup i Bezpieczeństwo

Przed rozpoczęciem:

```bash
cp data/transcribe.py data/transcribe.py.backup
```

Commituj po każdej fazie:

```bash
git add data/
git commit -m "Refactor: Faza X - [nazwa modułów]"
```

## Pliki do Modyfikacji/Utworzenia

### Plik źródłowy:

- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/transcribe.py` (modyfikacja)

### Nowe pliki:

- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/output_manager.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/command_builders.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/validators.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/youtube_processor.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/audio_processor.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/device_manager.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/transcription_engines.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/segment_processor.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/translation.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/srt_writer.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/tts_generator.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/audio_mixer.py`
- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/utils.py`

### Backup:

- `C:/Users/Arkadiusz/Desktop/backupPC/CODING/ATD/PROJEKT_TRANSKRYPCJA/data/transcribe.py.backup`

## Podsumowanie

Ten plan rozbija monolityczny `transcribe.py` na 13 wyspecjalizowanych modułów + zredukowany orchestrator. Konserwatywne podejście minimalizuje ryzyko, a jasna mapa zależności (DAG) zapobiega circular dependencies. Implementacja w 5 fazach z testami po każdej fazie zapewnia bezpieczną migrację.
