# ============================================
# DOCKERFILE - Transcription Tool with GPU Support
# ============================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 transcriber \
    && mkdir -p /data /models \
    && chown -R transcriber:transcriber /app /data /models

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.8)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import ctranslate2; print('CTranslate2:', ctranslate2.__version__)"

# Copy application code
COPY data/transcribe.py .

# Set environment variables for model cache
ENV HF_HOME=/models
ENV TORCH_HOME=/models
ENV XDG_CACHE_HOME=/models

# Volume mounts
VOLUME ["/data", "/models"]

# Switch to non-root user
USER transcriber

# Default entrypoint
ENTRYPOINT ["python", "transcribe.py"]
CMD ["--help"]
