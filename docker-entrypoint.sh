#!/bin/bash
# Docker entrypoint script - Railway deployment validation
# Ensures large model files are actual binaries (not git-lfs pointers)
# Fails immediately if any critical models are missing or corrupted

set -e

echo "[ENTRYPOINT] $(date '+%Y-%m-%d %H:%M:%S') - Starting model validation..."
echo ""

# CRITICAL: Runtime safety pull attempt (in case Railway runtime has LFS access)
# This complements the build-time pull, providing a failsafe
if command -v git-lfs &> /dev/null; then
    echo "[STEP 1] Attempting runtime git-lfs pull (failsafe)..."
    cd /app && \
    git lfs install --local --force 2>&1 | grep -v "Hooks have" || true && \
    git lfs pull --include="checkpoints/*.pt" --include="checkpoints/*.onnx" 2>&1 | head -5 || \
    echo "[WARN] Runtime git-lfs pull skipped (expected if already resolved)"
    echo ""
fi

echo "[STEP 2] VALIDATING MODEL FILES..."
echo ""

# Define model paths
TORCHSCRIPT_PATH="/app/checkpoints/yolov7_plant_disease.torchscript.pt"
CHECKPOINT_PATH="/app/checkpoints/best_model.pt"
METADATA_PATH="/app/checkpoints/model_meta.json"
ONNX_PATH="/app/checkpoints/yolov7_plant_disease.onnx"

# Check if checkpoints directory exists
if [ ! -d "/app/checkpoints" ]; then
    echo "[ERROR] /app/checkpoints directory not found!"
    echo "[ERROR] Model files must be present in Docker image"
    exit 1
fi

echo "Critical Model Files:"
echo "  TorchScript: $TORCHSCRIPT_PATH"
echo "  Checkpoint:  $CHECKPOINT_PATH"
echo "  Metadata:    $METADATA_PATH"
echo ""
echo "Checking file sizes and validity..."
echo ""

# Track validation status
VALIDATION_FAILED=0

# Check TorchScript
if [ -f "$TORCHSCRIPT_PATH" ]; then
    TS_SIZE=$(stat -c%s "$TORCHSCRIPT_PATH" 2>/dev/null || echo "0")
    if [ "$TS_SIZE" -lt 500 ]; then
        echo "[ERROR] TorchScript: ${TS_SIZE}B (GIT-LFS POINTER - NOT REAL FILE!)"
        VALIDATION_FAILED=1
    else
        TS_MB=$((TS_SIZE / 1048576))
        echo "[OK] TorchScript: ${TS_MB}MB"
    fi
else
    echo "[ERROR] TorchScript: FILE NOT FOUND"
    VALIDATION_FAILED=1
fi

# Check Checkpoint  
if [ -f "$CHECKPOINT_PATH" ]; then
    CP_SIZE=$(stat -c%s "$CHECKPOINT_PATH" 2>/dev/null || echo "0")
    if [ "$CP_SIZE" -lt 500 ]; then
        echo "[ERROR] Checkpoint: ${CP_SIZE}B (GIT-LFS POINTER - NOT REAL FILE!)"
        VALIDATION_FAILED=1
    else
        CP_MB=$((CP_SIZE / 1048576))
        echo "[OK] Checkpoint: ${CP_MB}MB"
    fi
else
    echo "[ERROR] Checkpoint: FILE NOT FOUND"
    VALIDATION_FAILED=1
fi

# Check Metadata
if [ -f "$METADATA_PATH" ]; then
    echo "[OK] Metadata: present"
else
    echo "[ERROR] Metadata: FILE NOT FOUND"
    VALIDATION_FAILED=1
fi

# Check ONNX (optional)
if [ -f "$ONNX_PATH" ]; then
    ONNX_SIZE=$(stat -c%s "$ONNX_PATH" 2>/dev/null || echo "0")
    if [ "$ONNX_SIZE" -ge 1000000 ]; then
        ONNX_MB=$((ONNX_SIZE / 1048576))
        echo "[OK] ONNX: ${ONNX_MB}MB"
    else
        echo "[WARN] ONNX: ${ONNX_SIZE}B (small/optional)"
    fi
fi

echo ""

# Fail if validation failed
if [ $VALIDATION_FAILED -eq 1 ]; then
    echo "[CRITICAL] Model validation FAILED!"
    echo ""
    echo "ERROR: Git-LFS pointer files detected instead of real model files."
    echo ""
    echo "This happens when:"
    echo "  1. Docker build clones the repo but git-lfs pull doesn't resolve pointers"
    echo "  2. Git-LFS credentials unavailable or server unreachable"
    echo "  3. .gitattributes missing or git-lfs not installed in Docker"
    echo ""
    echo "On Railway:"
    echo "  - Ensure .gitattributes file exists with LFS config"
    echo "  - Verify model files are in GitHub LFS storage"
    echo "  - Check Railway build logs for 'git lfs pull' output"
    echo "  - Rebuild: Push changes to trigger Railway rebuild"
    echo ""
    exit 1
fi

echo "[OK] All model files validated successfully!"
echo "[INFO] Starting application..."
echo ""

# Pass control to the CMD (uvicorn)
exec "$@"

