#!/bin/bash
# Docker entrypoint script - validates model files before starting the app
# This ensures the API only starts if critical model files are present and valid

set -e

echo "[START] Docker Entrypoint Script"
echo "[INFO] Checking model files..."
echo ""

# Define model paths
TORCHSCRIPT_PATH="/app/checkpoints/yolov7_plant_disease.torchscript.pt"
CHECKPOINT_PATH="/app/checkpoints/best_model.pt"
METADATA_PATH="/app/checkpoints/model_meta.json"

# Check if checkpoints directory exists
if [ ! -d "/app/checkpoints" ]; then
    echo "[ERROR] Critical: /app/checkpoints directory not found!"
    echo "[ERROR] Model files must be present in the Docker image"
    echo "[ERROR] Check that checkpoints/ is committed to git and not excluded in .dockerignore"
    exit 1
fi

# Function to check file size and validity
check_file() {
    local filepath=$1
    local filename=$(basename "$filepath")
    local min_size=${2:-1000000}  # Default 1MB minimum
    
    if [ ! -f "$filepath" ]; then
        echo "[ERROR] Missing: $filename"
        return 1
    fi
    
    # Get file size (works on both Linux and macOS)
    if command -v stat &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            local size=$(stat -f%z "$filepath" 2>/dev/null || echo "0")
        else
            # Linux
            local size=$(stat -c%s "$filepath" 2>/dev/null || echo "0")
        fi
    else
        # Fallback using ls + awk
        local size=$(ls -L "$filepath" 2>/dev/null | awk '{print $5}')
    fi
    
    size=${size:-0}
    local size_mb=$((size / 1048576))
    
    # Check if file is a git-lfs pointer (very small, text content)
    if [ "$size" -lt 500 ]; then
        echo "[ERROR] $filename appears to be git-lfs pointer (${size}B) - not actual file"
        echo "[ERROR]   This happens when git-lfs is configured but not pulled"
        echo "[ERROR]   Solution: Run 'git lfs pull --include=\"checkpoints/*.pt\"' in your repo"
        echo "[ERROR]   Or: Remove git-lfs: 'git rm .gitattributes && git add checkpoints/'"
        return 1
    elif [ "$size" -lt "$min_size" ]; then
        echo "[ERROR] $filename is too small (${size_mb}MB) - likely corrupted or not copied"
        return 1
    else
        echo "[OK] $filename: ${size_mb}MB"
        return 0
    fi
}

# Check metadata (required)
echo "Checking critical files:"
if ! check_file "$METADATA_PATH" 100; then
    echo "[ERROR] Model metadata file is missing or invalid!"
    echo "[ERROR] Cannot load model without metadata"
    exit 1
fi

echo ""
echo "Checking model files (at least one required):"

TORCHSCRIPT_OK=0
CHECKPOINT_OK=0

if check_file "$TORCHSCRIPT_PATH" 100000000; then  # 100MB minimum for TorchScript
    echo "[INFO] TorchScript model will be loaded"
    TORCHSCRIPT_OK=1
else
    echo "[WARN] TorchScript model not available"
fi

if check_file "$CHECKPOINT_PATH" 300000000; then  # 300MB minimum for checkpoint
    echo "[INFO] Checkpoint model will be loaded as fallback"
    CHECKPOINT_OK=1
else
    echo "[WARN] Checkpoint model not available"
fi

echo ""

# Check that at least one model file is present
if [ $TORCHSCRIPT_OK -eq 0 ] && [ $CHECKPOINT_OK -eq 0 ]; then
    echo "[ERROR] CRITICAL: No valid model files found!"
    echo "[ERROR] At least one of the following is required:"
    echo "[ERROR]   - $TORCHSCRIPT_PATH (108+ MB)"
    echo "[ERROR]   - $CHECKPOINT_PATH (324+ MB)"
    echo ""
    echo "[ERROR] Troubleshooting:"
    echo "[ERROR] 1. Verify model files are committed to git (not in .gitignore or .dockerignore)"
    echo "[ERROR] 2. Check that git has the full files (not corrupted or truncated)"
    echo "[ERROR] 3. For large files > 100MB, ensure git-lfs is configured"
    echo "[ERROR] 4. Rebuild Docker image: docker build -t crop-model ."
    echo ""
    exit 1
fi

echo "[OK] All critical model files are present and valid"
echo "[INFO] Starting application..."
echo ""

# Pass control to the CMD (uvicorn)
exec "$@"

