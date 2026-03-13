# Use multi-stage build to minimize image size
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU version (lighter than CUDA for web deployment)
RUN pip install --user --no-cache-dir \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --user --no-cache-dir -r requirements.txt

# Install gdown for Google Drive downloads
RUN pip install --user --no-cache-dir gdown

# ─────────────────────────────────────────────
# Download model files from Google Drive
# Files must be shared as "Anyone with the link can view"
# ─────────────────────────────────────────────
RUN mkdir -p /app/checkpoints

RUN /root/.local/bin/gdown "1EIwz1J-9bXUXrZGk35ef8159dAG-56JM" \
    -O /app/checkpoints/yolov7_plant_disease.torchscript.pt \
    --quiet 2>&1 || echo "[WARN] TorchScript gdown issue - checking file..."

RUN /root/.local/bin/gdown "1RaLO5MrJ-8H_Q4XGEGY8TXRvfqXdUZd1" \
    -O /app/checkpoints/best_model.pt \
    --quiet 2>&1 || echo "[WARN] Checkpoint gdown issue - checking file..."

RUN /root/.local/bin/gdown "1KmsorvSj6zEE1GAHEnRUCmxb9p6X4HiO" \
    -O /app/checkpoints/model_meta.json \
    --quiet 2>&1 || echo "[WARN] Metadata gdown issue - checking file..."

RUN /root/.local/bin/gdown "1Fx2TXACx_IFXvPTiCR-Lw5qG-HLP9B9D" \
    -O /app/checkpoints/yolov7_plant_disease.onnx \
    --quiet 2>&1 || echo "[WARN] ONNX gdown issue - checking file..."

# Verify downloads succeeded
RUN echo "[CHECK] Downloaded model files:" && \
    ls -lh /app/checkpoints/ && \
    torchscript_size=$(stat -c%s /app/checkpoints/yolov7_plant_disease.torchscript.pt 2>/dev/null || echo "0") && \
    checkpoint_size=$(stat -c%s /app/checkpoints/best_model.pt 2>/dev/null || echo "0") && \
    echo "[CHECK] TorchScript: ${torchscript_size}B" && \
    echo "[CHECK] Checkpoint: ${checkpoint_size}B" && \
    if [ "$torchscript_size" -lt 1000000 ] || [ "$checkpoint_size" -lt 1000000 ]; then \
        echo "[CRITICAL ERROR] Model files too small - Google Drive download likely failed!"; \
        echo "[CRITICAL ERROR] Check that files are shared as 'Anyone with the link'"; \
        exit 1; \
    else \
        torchscript_mb=$((torchscript_size / 1048576)); \
        checkpoint_mb=$((checkpoint_size / 1048576)); \
        echo "[OK] TorchScript: ${torchscript_mb}MB"; \
        echo "[OK] Checkpoint: ${checkpoint_mb}MB"; \
    fi


# ─────────────────────────────────────────────
# Final stage
# ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (no git-lfs needed anymore)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core-dev \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy downloaded model files from builder
COPY --from=builder /app/checkpoints /app/checkpoints

# Copy application code (no .git or git-lfs needed)
COPY . .

# Copy entrypoint
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]