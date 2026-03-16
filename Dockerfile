# ─────────────────────────────────────────────
# Base Image
# ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

ENV TORCH_HOME=/app/.cache

# ─────────────────────────────────────────────
# System Dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

# ─────────────────────────────────────────────
# Copy Application Code
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# IMPORTANT:
# Copy your model files directly into the project folder:
# checkpoints/
#     yolov7_plant_disease.torchscript.pt
#     best_model.pt
#     model_meta.json
#     yolov7_plant_disease.onnx
# ─────────────────────────────────────────────
COPY checkpoints /app/checkpoints

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────────
# Expose Port (Railway uses $PORT)
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# Start Application
# ─────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]