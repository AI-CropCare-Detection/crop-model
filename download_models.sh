#!/bin/bash
# Download model files from Google Drive
# Used during Docker build to fetch large model files

set -e

echo "[DOWNLOAD_MODELS] Starting model file download from Google Drive..."
echo ""

# Create output directory
mkdir -p /app/checkpoints

# Google Drive file IDs (replace with actual IDs from your Google Drive shares)
# Format: https://drive.google.com/uc?id=FILE_ID&export=download

# Function to download from Google Drive
download_from_gdrive() {
    local file_id=$1
    local filename=$2
    local expected_size=$3
    
    echo "[DOWNLOAD] Fetching $filename (expecting $expected_size bytes)..."
    
    # Use curl to download from Google Drive
    curl -L "https://drive.google.com/uc?id=${file_id}&export=download" \
         -o "/app/checkpoints/${filename}" \
         --progress-bar \
         --retry 3 \
         --retry-delay 2 || {
        echo "[ERROR] Failed to download $filename"
        return 1
    }
    
    # Check file size
    actual_size=$(stat -c%s "/app/checkpoints/${filename}" 2>/dev/null || echo "0")
    if [ "$actual_size" -lt 1000000 ]; then
        echo "[WARNING] $filename is only ${actual_size}B (expected ~$expected_size)"
        if grep -q "Too many users" "/app/checkpoints/${filename}" 2>/dev/null; then
            echo "[ERROR] Google Drive quota exceeded - file not downloaded"
            return 1
        fi
    else
        size_mb=$((actual_size / 1048576))
        echo "[OK] $filename downloaded: ${size_mb}MB"
    fi
    
    return 0
}

# Download model files
# NOTE: Replace these FILE_IDs with actual Google Drive share links
echo "[STEP 1] Downloading TorchScript model..."
download_from_gdrive "TORCHSCRIPT_FILE_ID" "yolov7_plant_disease.torchscript.pt" "108600000" || {
    echo "[WARN] TorchScript download failed, proceeding..."
}

echo ""
echo "[STEP 2] Downloading PyTorch checkpoint..."
download_from_gdrive "CHECKPOINT_FILE_ID" "best_model.pt" "324400000" || {
    echo "[WARN] Checkpoint download failed, proceeding..."
}

echo ""
echo "[STEP 3] Downloading ONNX model..."
download_from_gdrive "ONNX_FILE_ID" "yolov7_plant_disease.onnx" "108000000" || {
    echo "[WARN] ONNX download failed, proceeding..."
}

echo ""
echo "[STEP 4] Creating metadata file..."
cat > /app/checkpoints/model_meta.json << 'EOF'
{
  "model_name": "yolov7_plant_disease",
  "num_classes": 38,
  "input_size": 640,
  "framework": "pytorch",
  "model_formats": ["torchscript", "checkpoint", "onnx"]
}
EOF

echo ""
echo "[VERIFY] Checking downloaded files..."
ls -lh /app/checkpoints/

echo ""
echo "[DOWNLOAD_MODELS] Model download complete!"
