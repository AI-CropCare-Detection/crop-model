# ─────────────────────────────────────────────
# Base Image
# ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# ─────────────────────────────────────────────
# System Dependencies (including git + git-lfs)
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    libopencv-dev \
    libopencv-core-dev \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# ─────────────────────────────────────────────
# Install Python Dependencies
# ─────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

# ─────────────────────────────────────────────
# Copy Project Files
# ─────────────────────────────────────────────
COPY . .

# Pull actual files from Git LFS
RUN git lfs pull

# Ensure checkpoints folder exists
RUN mkdir -p /app/checkpoints

# Optional: verify model size during build
RUN ls -lh /app/checkpoints

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.cache

# ─────────────────────────────────────────────
# Railway Port Configuration
# ─────────────────────────────────────────────
EXPOSE 8000

# Start application using Railway PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]