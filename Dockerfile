# ─────────────────────────────────────────────
# Base Image
# ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# ─────────────────────────────────────────────
# System Dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopencv-dev \
    libopencv-core-dev \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

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

# Install gdown for Google Drive downloads
RUN pip install --no-cache-dir gdown

# ─────────────────────────────────────────────
# Copy Project Files
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# Ensure Checkpoints Folder Exists
# ─────────────────────────────────────────────
RUN mkdir -p /app/checkpoints

# ─────────────────────────────────────────────
# Download Model Files from Google Drive
# ─────────────────────────────────────────────

# best_model.pt
RUN gdown --fuzzy "https://drive.google.com/file/d/1RaLO5MrJ-8H_Q4XGEGY8TXRvfqXdUZd1/view?usp=drive_link" \
    -O /app/checkpoints/best_model.pt

# model_meta.json
RUN gdown --fuzzy "https://drive.google.com/file/d/1KmsorvSj6zEE1GAHEnRUCmxb9p6X4HiO/view?usp=drive_link" \
    -O /app/checkpoints/model_meta.json

# yolov7.onnx
RUN gdown --fuzzy "https://drive.google.com/file/d/1Fx2TXACx_IFXvPTiCR-Lw5qG-HLP9B9D/view?usp=drive_link" \
    -O /app/checkpoints/yolov7.onnx

# yolov7_plant_disease.torchscript.pt
RUN gdown --fuzzy "https://drive.google.com/file/d/1EIwz1J-9bXUXrZGk35ef8159dAG-56JM/view?usp=drive_link" \
    -O /app/checkpoints/yolov7_plant_disease.torchscript.pt

# ─────────────────────────────────────────────
# Verify All Downloads Succeeded
# ─────────────────────────────────────────────
RUN python -c "import os; files = {'/app/checkpoints/best_model.pt': 1000000, '/app/checkpoints/model_meta.json': 100, '/app/checkpoints/yolov7.onnx': 1000000, '/app/checkpoints/yolov7_plant_disease.torchscript.pt': 1000000}; errors = [f'{p}: {os.path.getsize(p)} bytes' for p, m in files.items() if os.path.getsize(p) < m]; [print('OK: ' + p) for p in files]; assert not errors, 'Download failed: ' + str(errors)"

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

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
