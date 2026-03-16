#!/bin/bash
# Docker entrypoint script - Railway deployment validation
# Models are downloaded from Google Drive during Docker build (not git-lfs)
# This script validates that all model files are present and correctly sized

set -e

echo "[ENTRYPOINT] $(date '+%Y-%m-%d %H:%M:%S') - Starting model validation..."
echo ""

echo "[STEP 1] VALIDATING MODEL FILES..."
echo ""

# Define model paths
TORCHSCRIPT_PATH="/app/checkpoints/yolov7_plant_disease.torchscript.pt"
CHECKPOINT_PATH="/app/checkpoints/best_model.pt"
METADATA_PATH="/app/checkpoints/model_meta.json"
ONNX_PATH="/app/checkpoints/yolov7_plant_disease.onnx"

# Check if checkpoints directory exists
if [ ! -d "/app/checkpoints" ]; then
    echo "[ERROR] /app/checkpoints directory not found!"
    echo "[ERROR] Model files should have been downloaded from Google Drive during Docker build."
    echo "[ERROR] Rebuild the Docker image to re-trigger the download step."
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
    if [ "$TS_SIZE" -lt 1000000 ]; then
        echo "[ERROR] TorchScript: ${TS_SIZE}B (TOO SMALL - download may have failed!)"
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
    if [ "$CP_SIZE" -lt 1000000 ]; then
        echo "[ERROR] Checkpoint: ${CP_SIZE}B (TOO SMALL - download may have failed!)"
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
        echo "[WARN] ONNX: ${ONNX_SIZE}B (unexpectedly small)"
    fi
else
    echo "[WARN] ONNX: not found (optional - skipping)"
fi

echo ""

# Fail if validation failed
if [ $VALIDATION_FAILED -eq 1 ]; then
    echo "[CRITICAL] Model validation FAILED!"
    echo ""
    echo "ERROR: One or more model files are missing or too small."
    echo ""
    echo "This happens when:"
    echo "  1. Google Drive download failed silently during Docker build"
    echo "  2. File was not shared as 'Anyone with the link can view'"
    echo "  3. Google Drive returned an HTML error page instead of the file"
    echo ""
    echo "Fix:"
    echo "  - Verify all 4 files in Google Drive are shared publicly"
    echo "  - Check Railway build logs for gdown output errors"
    echo "  - Rebuild: push a commit to trigger a fresh Railway build"
    echo ""
    exit 1
fi

echo "[OK] All model files validated successfully!"
echo "[INFO] Starting application..."
echo ""

# Pass control to the CMD (uvicorn)
exec "$@"