# ============================================
# DOCKERFILE - Transcription Tool with GPU Support
# Multi-Stage Build for Optimized Image Size
# ============================================

# ============================================
# STAGE 1: Builder
# ============================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip (install to system Python - no venv needed in Docker)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch 2.8.0 with CUDA 12.6 support (compatible with CUDA 12.4+)
# ONLY torch + torchaudio, NO torchvision
RUN pip install --no-cache-dir \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify critical dependencies
RUN python -c "import whisperx; print('WhisperX: OK')" && \
    python -c "from transformers import Pipeline; print('Transformers Pipeline: OK')" && \
    python -c "import whisper; print('OpenAI Whisper:', whisper.__version__)" && \
    python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Clean up to reduce layer size
RUN pip cache purge && \
    find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -type d -name tests -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -type d -name test -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "*.pyo" -delete

# ============================================
# STAGE 2: Runtime
# ============================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install ONLY runtime dependencies (no git, no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy Python packages from builder (no venv - copy site-packages directly)
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory and user
WORKDIR /app
RUN useradd -m -u 1000 transcriber \
    && mkdir -p /data /models \
    && chown -R transcriber:transcriber /app /data /models

# Copy application code
COPY transcribe.py .
COPY data/ ./data/

# Set environment variables for model cache
ENV HF_HOME=/models
ENV TORCH_HOME=/models
ENV XDG_CACHE_HOME=/models

# Volume mounts
VOLUME ["/data", "/models"]

# Switch to non-root user
USER transcriber

# Default entrypoint
ENTRYPOINT ["python", "/app/transcribe.py"]
CMD ["--help"]
