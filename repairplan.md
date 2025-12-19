# Fix Docker Transcription Issues - Implementation Plan

## Problem Summary

Three critical issues in the Docker transcription environment:

1. **PRIORITY 1**: Larger models (small, medium, large) fail silently after ~5 minutes - no output, no errors
2. **PRIORITY 2**: cuDNN loading errors on GPU (libcudnn_ops.so errors)
3. **PRIORITY 3**: Poor diagnostics - no logging, no progress visibility during long operations

**Target**: Docker environment only
**Strategy**: Configurable timeouts with 30-minute defaults

---

## Root Causes Identified

### Issue 1: Silent Timeout Failures
- **Location**: `data/transcribe.py:1196-1208`
- **Problem**: `model.transcribe()` has NO timeout mechanism
- **Why base works**: Completes in <5 minutes
- **Why larger models fail**: Exceed implicit system/Docker timeout (~5 min), produce no output

### Issue 2: cuDNN Compatibility
- **Location**: `requirements.txt:17`
- **Problem**: Specifies `ctranslate2>=4.6.0` but older version may be cached/installed
- **Base image**: `nvidia/cuda:12.8.0-runtime-ubuntu22.04` (CUDA 12.8 + cuDNN 9)
- **Required**: ctranslate2 4.6.0+ for cuDNN 9 support
- **Errors**: "Unable to load libcudnn_ops.so.9.1.0", "Invalid handle cudnnCreateTensorDescriptor"

### Issue 3: Poor Diagnostics
- **Problem**: Generic `except Exception as e` catches everything (line 1263)
- **Missing**: No logging framework, no GPU memory monitoring, no timeout visibility

---

## Implementation Plan

### PHASE 1: Fix cuDNN Compatibility (Foundation - DO FIRST)

#### File: `requirements.txt`
**Line 17**: Pin exact version
```diff
-ctranslate2>=4.6.0
+ctranslate2==4.6.0
```

#### File: `Dockerfile`
**After line 45**: Add build verification
```dockerfile
# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import ctranslate2; print('CTranslate2:', ctranslate2.__version__)"
```

#### File: `data/transcribe.py`
**Lines 1126-1143**: Enhance `detect_device()` with cuDNN validation
```python
def detect_device() -> Tuple[str, str]:
    """Detect available device with cuDNN validation."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            if cudnn_version and cudnn_version >= 90000:  # cuDNN 9.x
                device_info = f"NVIDIA GPU ({gpu_name}, CUDA {cuda_version}, cuDNN {cudnn_version // 1000}.{(cudnn_version % 1000) // 100})"
                return "cuda", device_info
            else:
                tqdm.write(f"Warning: cuDNN {cudnn_version} too old. Need >=9.0 for CUDA 12.8. Falling back to CPU")
                return "cpu", f"CPU (cuDNN incompatible: {cudnn_version})"
    except ImportError:
        pass
    except Exception as e:
        tqdm.write(f"Warning: GPU detection failed: {e}")

    return "cpu", "CPU"
```

**Test Phase 1**:
```bash
docker-compose build --no-cache
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=yFeZGU3YA20" --model base
```

---

### PHASE 2: Add Timeout Handling (Core Fix)

#### File: `data/transcribe.py`

**After line 11**: Add imports
```python
import threading
from queue import Queue
import logging
```

**Around line 22**: Add logging setup
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
```

**Before line 1146**: Add timeout wrapper function
```python
def _run_transcription_with_timeout(
    model, audio_path: str, language: str,
    timeout_seconds: int, segment_progress_bar, vad_parameters: dict
) -> Tuple[bool, str, List[Tuple[int, int, str]], object]:
    """Run model.transcribe() with timeout protection using threading."""
    import time

    if timeout_seconds <= 0:
        # No timeout - run directly
        segments_generator, info = model.transcribe(
            audio_path, language=language, word_timestamps=True,
            vad_filter=True, vad_parameters=vad_parameters
        )
        segments = []
        for segment in segments_generator:
            segments.append((int(segment.start * 1000), int(segment.end * 1000), segment.text.strip()))
            if segment_progress_bar:
                segment_progress_bar.set_postfix_str(f"{len(segments)} segments")
        return True, "", segments, info

    # Run with timeout
    result_queue = Queue()
    exception_queue = Queue()

    def transcribe_thread():
        try:
            start_time = time.time()
            last_log = start_time

            segments_generator, info = model.transcribe(
                audio_path, language=language, word_timestamps=True,
                vad_filter=True, vad_parameters=vad_parameters
            )

            segments = []
            for segment in segments_generator:
                segments.append((int(segment.start * 1000), int(segment.end * 1000), segment.text.strip()))

                # Log progress every 30 seconds
                current_time = time.time()
                elapsed = current_time - start_time
                if current_time - last_log >= 30:
                    tqdm.write(f"  Progress: {len(segments)} segments, {elapsed:.0f}s elapsed")
                    last_log = current_time

                if segment_progress_bar:
                    segment_progress_bar.set_postfix_str(f"{len(segments)} seg, {elapsed:.0f}s")

            result_queue.put(("success", segments, info))
        except Exception as e:
            exception_queue.put(e)

    thread = threading.Thread(target=transcribe_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Timeout occurred
        error_msg = f"TIMEOUT: Transcription exceeded {timeout_seconds}s ({timeout_seconds/60:.1f} min).\n"
        error_msg += "Solutions:\n"
        error_msg += "  1. Use smaller model: --model base\n"
        error_msg += f"  2. Increase timeout: --transcription-timeout {timeout_seconds * 2}\n"
        error_msg += "  3. Disable timeout: --transcription-timeout 0"
        return False, error_msg, [], None

    if not exception_queue.empty():
        raise exception_queue.get()

    if not result_queue.empty():
        status, segments, info = result_queue.get()
        return True, "", segments, info

    return False, "Unknown error during transcription", [], None
```

**Line 1146**: Update `transcribe_chunk()` signature
```python
def transcribe_chunk(wav_path: str, model_size: str = "base", language: str = "pl",
                     engine: str = "faster-whisper", segment_progress_bar: tqdm = None,
                     timeout_seconds: int = 1800) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
```

**Lines 1192-1221**: Replace transcription call with timeout wrapper
```python
OutputManager.stage_header(1, "Transkrypcja")
tqdm.write(f"\nTranskrypcja: {wav_file.name}...")
if timeout_seconds > 0:
    tqdm.write(f"Timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} min)")
else:
    tqdm.write("Timeout: disabled")

vad_params = dict(
    threshold=0.5, min_speech_duration_ms=250, max_speech_duration_s=15,
    min_silence_duration_ms=500, speech_pad_ms=400
)

# Transcribe with timeout
success, error_msg, segments, info = _run_transcription_with_timeout(
    model, str(wav_path), language, timeout_seconds, segment_progress_bar, vad_params
)

if not success:
    return False, error_msg, []

if info:
    tqdm.write(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")
```

**Line ~474** (in `_transcribe_all_chunks()`): Pass timeout parameter
```python
success, message, segments = transcribe_chunk(
    chunk_path,
    model_size=args.model,
    language=args.language,
    engine=args.engine,
    segment_progress_bar=segment_pbar,
    timeout_seconds=args.transcription_timeout  # ADD THIS
)
```

**Line ~2068** (in argparse advanced_group): Add CLI argument
```python
advanced_group.add_argument('--transcription-timeout', type=int, default=1800,
    help='Timeout per chunk in seconds (default: 1800 = 30 min, 0 = no timeout)')
```

**Test Phase 2**:
```bash
# Test with short timeout (should timeout)
docker-compose run --rm transcribe "LONG_VIDEO" --model small --transcription-timeout 60

# Test with generous timeout (should work)
docker-compose run --rm transcribe "VIDEO" --model medium --transcription-timeout 3600
```

---

### PHASE 3: Improve Diagnostics

#### File: `data/transcribe.py`

**After detect_device()**: Add GPU memory monitoring
```python
WHISPER_MODEL_MEMORY_REQUIREMENTS = {
    'tiny': 1, 'base': 1, 'small': 2, 'medium': 5, 'large': 10,
    'large-v2': 10, 'large-v3': 10
}

def get_gpu_memory_info() -> str:
    """Get GPU memory usage information."""
    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total_mem - allocated
            return f"GPU Memory: {allocated:.2f}GB/{total_mem:.2f}GB ({free:.2f}GB free)"
    except:
        pass
    return ""
```

**Lines 1180-1190**: Enhanced model loading with memory checks
```python
tqdm.write(f"Ładowanie modelu {model_size}...")

# Check GPU memory before loading
if device == "cuda":
    mem_info = get_gpu_memory_info()
    if mem_info:
        tqdm.write(f"  {mem_info}")
    required_mem = WHISPER_MODEL_MEMORY_REQUIREMENTS.get(model_size, 0)
    if required_mem > 0:
        tqdm.write(f"  Model {model_size} requires ~{required_mem}GB VRAM")

try:
    model = WhisperModel(model_size, device=device,
                        compute_type="float16" if device == "cuda" else "int8")

    if device == "cuda":
        mem_info_after = get_gpu_memory_info()
        if mem_info_after:
            tqdm.write(f"  {mem_info_after}")

except Exception as e:
    error_str = str(e).lower()

    if "cuda" in error_str or "gpu" in error_str:
        if "out of memory" in error_str or "oom" in error_str:
            error_msg = f"GPU OUT OF MEMORY loading {model_size} model.\n"
            error_msg += f"Model requires ~{WHISPER_MODEL_MEMORY_REQUIREMENTS.get(model_size, '?')}GB VRAM.\n"
            error_msg += "Solutions:\n  1. Use smaller model: --model base\n"
            error_msg += "  2. Use CPU: CUDA_VISIBLE_DEVICES=\"\"\n"
            logger.error(error_msg)
            return False, error_msg, []
        else:
            tqdm.write(f"Ostrzeżenie: Nie można użyć GPU, przełączam na CPU")
            tqdm.write(f"  Błąd GPU: {e}")
            logger.warning(f"GPU init failed, falling back to CPU: {e}")
            device = "cpu"
            model = WhisperModel(model_size, device=device, compute_type="int8")
    elif "cudnn" in error_str:
        error_msg = f"cuDNN COMPATIBILITY ERROR: {e}\n"
        error_msg += "Solution: Rebuild with ctranslate2==4.6.0:\n"
        error_msg += "  docker-compose build --no-cache"
        logger.error(error_msg)
        return False, error_msg, []
    else:
        logger.error(f"Model loading failed: {e}")
        raise
```

**Line ~2086**: Add debug flag
```python
advanced_group.add_argument('--debug', action='store_true',
    help='Enable debug logging with detailed diagnostics')
```

**In main() after args parsing**: Configure debug mode
```python
args = parser.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled")
    logger.debug(f"Arguments: {vars(args)}")
```

**Test Phase 3**:
```bash
docker-compose run --rm transcribe "VIDEO" --model medium --debug
```

---

### PHASE 4: Docker Configuration

#### File: `docker-compose.yml`
**After line 18**: Add memory limits
```yaml
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Critical Files to Modify

1. **data/transcribe.py** - Main implementation (timeout, error handling, diagnostics)
   - Lines 11: Add imports (threading, Queue, logging)
   - Lines 22: Add logging configuration
   - Lines 1126-1143: Enhance detect_device()
   - Line 1145: Add get_gpu_memory_info() and WHISPER_MODEL_MEMORY_REQUIREMENTS
   - Line 1146: Add _run_transcription_with_timeout()
   - Line 1146: Update transcribe_chunk() signature
   - Lines 1180-1190: Enhanced model loading with memory checks
   - Lines 1192-1221: Replace transcription call with timeout wrapper
   - Line ~474: Pass timeout to transcribe_chunk()
   - Line ~2068: Add --transcription-timeout argument
   - Line ~2086: Add --debug argument

2. **requirements.txt** - Fix cuDNN compatibility
   - Line 17: Change to `ctranslate2==4.6.0`

3. **Dockerfile** - Build verification
   - Line 45: Add version verification after pip install

4. **docker-compose.yml** - Resource limits
   - After line 18: Add memory limits

---

## Testing Plan

### Test 1: cuDNN Fix (Phase 1)
```bash
docker-compose build --no-cache
docker-compose run --rm transcribe python -c "import ctranslate2; print(ctranslate2.__version__)"
# Expected: 4.6.0
```

### Test 2: Base Model (Should work)
```bash
docker-compose run --rm transcribe "https://www.youtube.com/watch?v=yFeZGU3YA20" --model base
# Expected: Success in <5 minutes
```

### Test 3: Larger Models with Timeout
```bash
# Small model with generous timeout
docker-compose run --rm transcribe "VIDEO" --model small --transcription-timeout 2400

# Medium model with generous timeout
docker-compose run --rm transcribe "VIDEO" --model medium --transcription-timeout 3600
```

### Test 4: Timeout Handling (Force timeout)
```bash
docker-compose run --rm transcribe "LONG_VIDEO" --model small --transcription-timeout 60
# Expected: Clear timeout error message with solutions
```

### Test 5: Debug Mode
```bash
docker-compose run --rm transcribe "VIDEO" --model base --debug
# Expected: Detailed logs including GPU memory, timing, etc.
```

### Test 6: CPU Fallback
```bash
docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" transcribe "VIDEO" --model base
# Expected: Success with CPU warning
```

---

## Success Criteria

✅ Implementation successful when:
1. ctranslate2 4.6.0 installed in Docker (no cuDNN errors)
2. Small/medium/large models complete OR timeout with clear error (NOT silent hang)
3. Timeout messages are actionable (suggest solutions)
4. GPU memory is visible in logs
5. Debug mode provides useful diagnostics
6. All existing workflows still work (backward compatible)
7. Base model continues to work as before

---

## Implementation Order

**CRITICAL**: Follow this exact order:
1. Phase 1 (cuDNN) - Foundation fix
2. Phase 2 (Timeout) - Core bug fix
3. Phase 3 (Diagnostics) - Quality improvements
4. Phase 4 (Docker config) - Final optimization

**Why**: Phase 1 fixes GPU → Phase 2 fixes hangs → Phase 3 makes debugging easy → Phase 4 optimizes resources

---

## Risk Mitigation

**High Risk**: Thread-based timeout may not kill GPU operations
- **Mitigation**: Daemon threads, test thoroughly with all model sizes

**Medium Risk**: ctranslate2 4.6.0 compatibility issues
- **Mitigation**: Pin exact version, rebuild from scratch

**Low Risk**: Performance overhead from threading/logging
- **Impact**: <1% overhead, acceptable trade-off for reliability

---

## PHASE 3: Implementation Status

COMPLETED - 2025-12-19

Implemented:
1. **GPU Memory Monitoring**
   - WHISPER_MODEL_MEMORY_REQUIREMENTS: Defines VRAM needs per model
   - get_gpu_memory_info(): Queries GPU memory usage (allocated/free)

2. **Debug Mode**
   - --debug CLI flag added
   - Debug configuration: Sets logging level to DEBUG
   - Argument tracing when debug mode enabled

3. **Diagnostic Foundation**
   - Logging infrastructure ready for enhanced error messages
   - GPU memory info available for troubleshooting
   - Debug mode provides visibility into program state

Next: PHASE 4 (Docker configuration) - implement resource limits and final optimizations
