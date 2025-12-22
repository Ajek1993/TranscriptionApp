# Plan: Dubbing TTS dla transcribe.py

## Cel
Dodanie funkcjonalności Text-to-Speech (TTS) do generowania polskiego dubbingu z synchronizacją czasową i mixowaniem z oryginalnym audio.

## Wymagania
- **Silnik TTS**: Edge TTS (Microsoft, darmowy, wysoka jakość)
- **Język**: Polski (głos `pl-PL-MarekNeural` lub `pl-PL-ZofiaNeural`)
- **Audio**: Mix - oryginalne audio ściszone (20%) + TTS (100%)
- **Synchronizacja**: TTS dopasowany do timestampów napisów

---

## Pliki do modyfikacji

| Plik | Zmiany |
|------|--------|
| `transcribe.py` | Nowe funkcje TTS, argumenty CLI, etap 6 w pipeline |
| `requirements.txt` | Dodanie `edge-tts>=7.0.0` |
| `README.md` | Dokumentacja nowych flag i przykłady użycia |

---

## Implementacja

### 1. Nowa zależność

```
edge-tts>=7.0.0
```

### 2. Nowe funkcje w transcribe.py

#### 2.1 Import i sprawdzenie dostępności (~linia 27)
```python
try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
```

#### 2.2 `check_edge_tts_dependency()` (~linia 112)
Sprawdza czy edge-tts jest zainstalowany.

#### 2.3 `generate_tts_for_segment()` (async, ~linia 470)
```python
async def generate_tts_for_segment(
    text: str,
    output_path: str,
    voice: str = "pl-PL-MarekNeural",
    rate: str = "+0%"
) -> Tuple[bool, str, float]:
```
Generuje pojedynczy segment TTS z kontrolą tempa.

#### 2.4 `generate_tts_segments()` (~linia 500)
```python
def generate_tts_segments(
    segments: List[Tuple[int, int, str]],
    output_dir: str,
    voice: str = "pl-PL-MarekNeural"
) -> Tuple[bool, str, List[Tuple[int, str, float]]]:
```
Generuje TTS dla wszystkich segmentów z automatycznym przyspieszaniem gdy TTS jest dłuższy niż slot czasowy.

#### 2.5 `create_tts_audio_track()` (~linia 550)
```python
def create_tts_audio_track(
    tts_files: List[Tuple[int, str, float]],
    total_duration_ms: int,
    output_path: str
) -> Tuple[bool, str]:
```
Łączy segmenty TTS w jedną ścieżkę z użyciem ffmpeg `adelay` + `amix`.

#### 2.6 `mix_audio_tracks()` (~linia 600)
```python
def mix_audio_tracks(
    original_audio_path: str,
    tts_audio_path: str,
    output_path: str,
    original_volume: float = 0.2,
    tts_volume: float = 1.0
) -> Tuple[bool, str]:
```
Mixuje oryginalne audio (ściszone) z TTS.

#### 2.7 `create_dubbed_video()` (~linia 640)
```python
def create_dubbed_video(
    original_video_path: str,
    mixed_audio_path: str,
    output_video_path: str
) -> Tuple[bool, str]:
```
Tworzy końcowe wideo z dubbingiem (video copy + nowe audio AAC).

### 3. Nowe argumenty CLI (~linia 641)

```python
parser.add_argument('--dub', action='store_true',
                   help='Generuj dubbing TTS')
parser.add_argument('--tts-voice', type=str, default='pl-PL-MarekNeural',
                   choices=['pl-PL-MarekNeural', 'pl-PL-ZofiaNeural'],
                   help='Glos TTS')
parser.add_argument('--tts-volume', type=float, default=1.0,
                   help='Glosnosc TTS (0.0-2.0)')
parser.add_argument('--original-volume', type=float, default=0.2,
                   help='Glosnosc oryginalnego audio (0.0-1.0)')
parser.add_argument('--dub-output', type=str,
                   help='Nazwa pliku wyjsciowego z dubbingiem')
```

### 4. Etap 6 w pipeline (~linia 882)

Po wygenerowaniu SRT, jeśli `--dub`:
1. Generuj TTS dla każdego segmentu
2. Połącz segmenty TTS z odpowiednimi opóźnieniami
3. Zmixuj oryginalne audio (20%) + TTS (100%)
4. Utwórz wideo z nowym audio

---

## Komendy ffmpeg

### Łączenie segmentów TTS z opóźnieniami
```bash
ffmpeg -y -i tts_0.mp3 -i tts_1.mp3 -i tts_2.mp3 \
  -filter_complex "[0]adelay=0|0[d0];[1]adelay=5000|5000[d1];[2]adelay=12000|12000[d2];[d0][d1][d2]amix=inputs=3:duration=longest[out]" \
  -map "[out]" -ar 44100 -ac 2 tts_combined.wav
```

### Mixowanie audio
```bash
ffmpeg -y -i original.wav -i tts_combined.wav \
  -filter_complex "[0:a]volume=0.2[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first[out]" \
  -map "[out]" -ar 44100 -ac 2 mixed.wav
```

### Tworzenie wideo z dubbingiem
```bash
ffmpeg -y -i original.mp4 -i mixed.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest output_dubbed.mp4
```

---

## Synchronizacja TTS

Problem: TTS może być dłuższy niż slot czasowy.

Rozwiązanie:
1. Generuj TTS z normalnym tempem
2. Zmierz długość wygenerowanego audio
3. Jeśli dłuższe niż slot → wylicz wymagane przyspieszenie
4. Regeneruj z parametrem `rate="+X%"` (max +50%)
5. Umieść segment w odpowiednim miejscu za pomocą `adelay`

---

## Przykłady użycia

```bash
# Podstawowy dubbing
python transcribe.py --local "video.mp4" --dub

# Dubbing angielskiego filmu z tłumaczeniem na polski
python transcribe.py --local "english.mp4" --language en --translate en-pl --dub

# Własne ustawienia głosu i głośności
python transcribe.py --local "video.mp4" --dub \
  --tts-voice pl-PL-ZofiaNeural \
  --original-volume 0.1 \
  --tts-volume 1.2 \
  --dub-output moj_dubbing.mp4
```

---

## Obsługa błędów

| Przypadek | Rozwiązanie |
|-----------|-------------|
| Pusty tekst segmentu | Pomiń generowanie TTS |
| TTS za długi (>150% slotu) | Przyspiesz do +50%, przytnij jeśli nadal za długi |
| Błąd sieci (edge-tts) | Retry 3x z exponential backoff |
| Timeout ffmpeg | Zwiększony timeout (30 min) dla długich wideo |
| Brak ścieżki wideo | Błąd: dubbing wymaga pliku wideo |
