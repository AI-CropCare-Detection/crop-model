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
# If GPU is available on Railway, you can modify this
RUN pip install --user --no-cache-dir \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies including git and git-lfs (in case files are LFS-tracked)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core-dev \
    libsm6 \
    libxext6 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code (includes .gitattributes and .git) 
COPY . .

# CRITICAL: Initialize git-lfs and pull actual model files instead of pointers
# On Railway, this ensures large model files are fetched from LFS storage
RUN cd /app && \
    echo "[STEP 1] Initializing git-lfs in docker context..." && \
    git lfs install --local --force && \
    echo "[STEP 2] Configuring git to accept all certificates..." && \
    git config --global http.sslverify false && \
    echo "[STEP 3] Fetching git-lfs model files (pt, onnx)..." && \
    git lfs pull --include="checkpoints/*.pt" --include="checkpoints/*.onnx" --include="checkpoints/*.json" 2>&1 || \
    (echo "[WARN] Git-LFS pull had issues, attempting direct checkout..."; git checkout checkpoints/ 2>&1 || true) && \
    echo "[STEP 4] Verifying model files..." && \
    ls -lh /app/checkpoints/ && \
    echo ""

# Strict model file validation - Docker build FAILS if files are only pointers
RUN \
    torchscript_size=$(stat -c%s /app/checkpoints/yolov7_plant_disease.torchscript.pt 2>/dev/null || echo "0") && \
    checkpoint_size=$(stat -c%s /app/checkpoints/best_model.pt 2>/dev/null || echo "0") && \
    echo "[CHECK] TorchScript: ${torchscript_size}B" && \
    echo "[CHECK] Checkpoint: ${checkpoint_size}B" && \
    if [ "$torchscript_size" -lt 1000000 ] || [ "$checkpoint_size" -lt 1000000 ]; then \
        echo ""; \
        echo "[CRITICAL ERROR] Model files are too small!"; \
        echo "[CRITICAL ERROR] Files appear to be git-lfs pointers (text), not actual binary files."; \
        echo "[CRITICAL ERROR] This means: git lfs pull did not work correctly in Docker build."; \
        echo "[CRITICAL ERROR] "; \
        echo "[CRITICAL ERROR] On Railway, this could happen because:"; \
        echo "[CRITICAL ERROR]   1. Git LFS credentials not configured"; \
        echo "[CRITICAL ERROR]   2. Network timeout during git lfs pull"; \
        echo "[CRITICAL ERROR]   3. Git LFS server unreachable"; \
        echo "[CRITICAL ERROR] "; \
        echo "[CRITICAL ERROR] Solution:"; \
        echo "[CRITICAL ERROR]   - Ensure .gitattributes file has LFS config"; \
        echo "[CRITICAL ERROR]   - Verify model files are in LFS storage"; \
        echo "[CRITICAL ERROR]   - Check Railway can access GitHub LFS"; \
        echo ""; \
        exit 1; \
    else \
        echo "[OK] Model files verified - sizes acceptable"; \
        torchscript_mb=$((torchscript_size / 1048576)); \
        checkpoint_mb=$((checkpoint_size / 1048576)); \
        echo "[OK] TorchScript: ${torchscript_mb}MB"; \
        echo "[OK] Checkpoint: ${checkpoint_mb}MB"; \
    fi

# Create entrypoint script for validation
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

# Use entrypoint script to validate models before starting
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

# Run with uvicorn (use environment variable for port, defaulting to 8000)
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]
