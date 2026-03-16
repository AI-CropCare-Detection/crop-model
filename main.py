import os
import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
from PIL import Image
import cv2
from datetime import datetime
import gc
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
# Checkpoint Paths  (must match Dockerfile downloads)
# ─────────────────────────────────────────────
CHECKPOINTS_DIR   = Path(__file__).parent / 'checkpoints'
MODEL_META_PATH   = CHECKPOINTS_DIR / 'model_meta.json'
checkpoint_path   = CHECKPOINTS_DIR / 'best_model.pt'
torchscript_path  = CHECKPOINTS_DIR / 'yolov7_plant_disease.torchscript.pt'
onnx_path         = CHECKPOINTS_DIR / 'yolov7.onnx'

# Load class names and metadata from model_meta.json
try:
    with open(MODEL_META_PATH, 'r') as f:
        model_meta = json.load(f)
    CLASS_NAMES = model_meta.get('class_names', [])
    NUM_CLASSES = model_meta.get('num_classes', len(CLASS_NAMES))
    IMG_SIZE = model_meta.get('img_size', 128)
    CLASS_MAP = {i: class_name for i, class_name in enumerate(CLASS_NAMES)}
    CLASS_TO_IDX = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
    logger.info(f"[OK] Loaded model_meta.json — {NUM_CLASSES} classes, img_size={IMG_SIZE}")
except FileNotFoundError:
    logger.warning(f"[WARN] {MODEL_META_PATH} not found. Using default classes.")
    CLASS_NAMES = ['healthy', 'diseased']
    NUM_CLASSES = len(CLASS_NAMES)
    IMG_SIZE = 128
    CLASS_MAP = {i: class_name for i, class_name in enumerate(CLASS_NAMES)}
    CLASS_TO_IDX = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────
class YOLOv7Classifier(torch.nn.Module):
    """Simple YOLOv7-based classifier for plant disease detection."""

    def __init__(self, num_classes=2, img_size=128, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────
def _check_file(path: Path, label: str, min_bytes: int = 1_000_000) -> bool:
    """Return True if file exists and is large enough to be valid."""
    if not path.exists():
        logger.warning(f"[WARN] {label} not found at {path}")
        return False
    size = path.stat().st_size
    size_mb = size / (1024 * 1024)
    if size < min_bytes:
        logger.error(f"[ERROR] {label} too small ({size_mb:.3f} MB) — download likely failed")
        return False
    logger.info(f"[OK] {label} found ({size_mb:.1f} MB)")
    return True


def load_model():
    """Load the trained model. Tries TorchScript first, then checkpoint."""

    # ── 1. TorchScript ──────────────────────────────────────────────────
    if _check_file(torchscript_path, "TorchScript model"):
        try:
            logger.info(f"[LOAD] Loading TorchScript: {torchscript_path}")
            model = torch.jit.load(str(torchscript_path), map_location=DEVICE)
            model.eval()
            logger.info(f"[OK] TorchScript model loaded on {DEVICE}")
            return model
        except Exception as e:
            logger.warning(f"[WARN] TorchScript load failed: {type(e).__name__}: {str(e)[:150]}")

    # ── 2. Checkpoint (.pt) ─────────────────────────────────────────────
    if _check_file(checkpoint_path, "Checkpoint (best_model.pt)"):
        try:
            logger.info(f"[LOAD] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                    logger.info(f"   Epoch: {checkpoint.get('epoch', 'N/A')}  "
                                f"Best F1: {checkpoint.get('best_f1', 'N/A')}")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            logger.info(f"[LOAD] Building YOLOv7Classifier ({NUM_CLASSES} classes)…")
            model = YOLOv7Classifier(num_classes=NUM_CLASSES, dropout=0.0)
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"[WARN] {len(incompatible.missing_keys)} missing keys")
            if incompatible.unexpected_keys:
                logger.warning(f"[WARN] {len(incompatible.unexpected_keys)} unexpected keys")

            model.to(DEVICE)
            model.eval()
            logger.info(f"[OK] Checkpoint model loaded on {DEVICE}")
            return model

        except Exception as e:
            logger.error(f"[ERROR] Checkpoint load failed: {type(e).__name__}: {str(e)[:150]}")

    # ── Nothing worked ───────────────────────────────────────────────────
    def _fmt(p: Path) -> str:
        return f"{p} (exists={p.exists()}, size={p.stat().st_size if p.exists() else 0}B)"

    raise RuntimeError(
        "CRITICAL: Could not load any model file!\n"
        f"  TorchScript : {_fmt(torchscript_path)}\n"
        f"  Checkpoint  : {_fmt(checkpoint_path)}\n\n"
        "Troubleshooting:\n"
        "1. Check Railway build logs for gdown download errors\n"
        "2. Confirm Google Drive files are shared as 'Anyone with the link → Viewer'\n"
        "3. Verify file IDs in Dockerfile match actual Drive links\n"
        "4. Rebuild the Docker image to retry downloads"
    )


# ─────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def process_image(image_data: bytes, img_size: int = None) -> torch.Tensor:
    """Validate, resize, and convert an uploaded image to a model-ready tensor."""
    img_size = img_size or IMG_SIZE
    try:
        if len(image_data) > MAX_FILE_SIZE:
            raise ValueError("File too large. Max size: 50 MB")
        if len(image_data) == 0:
            raise ValueError("File is empty")

        img = Image.open(BytesIO(image_data)).convert('RGB')

        if img.size[0] < 16 or img.size[1] < 16:
            raise ValueError(f"Image too small (minimum 16×16). Got: {img.size}")

        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0

        if np.isnan(img_array).any() or np.isinf(img_array).any():
            raise ValueError("Invalid image data detected")

        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contains NaN or Inf values")

        return tensor.to(DEVICE)

    except Exception as e:
        logger.error(f"[ERROR] Image processing error: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detection API",
    description="YOLOv7-based API for plant disease classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
model_loaded = False


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load model on startup. Application will NOT start if model fails to load."""
    global model, model_loaded

    logger.info("[START] Starting Plant Disease Detection API...")
    logger.info(f"[INFO] Classes : {NUM_CLASSES}")
    logger.info(f"[INFO] Device  : {DEVICE}")
    logger.info(f"[INFO] Checkpoints dir: {CHECKPOINTS_DIR}")

    # Log what files are present so Railway logs are self-diagnosing
    for p in [torchscript_path, checkpoint_path, MODEL_META_PATH, onnx_path]:
        exists = p.exists()
        size   = p.stat().st_size if exists else 0
        logger.info(f"[FILE] {p.name}: exists={exists}, size={size/1024:.1f} KB")

    try:
        if not model_loaded:
            model = load_model()
            model_loaded = True

            if model is None:
                raise RuntimeError("load_model() returned None")

            logger.info(f"[OK] Model ready on {DEVICE}")

            if torch.cuda.is_available():
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"[INFO] GPU memory: {mem_gb:.1f} GB")

    except Exception as e:
        logger.error(f"[CRITICAL] Model loading failed during startup: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"[CRITICAL] Model loading failed during startup: {str(e)}") from e


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown."""
    try:
        logger.info("[STOP] Shutting down API…")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[OK] Cleanup complete")
    except Exception as e:
        logger.error(f"[ERROR] Shutdown error: {str(e)}")


# ─────────────────────────────────────────────
# Exception Handlers
# ─────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"[WARN] Validation error: {exc}")
    return JSONResponse(status_code=400, content={"detail": "Invalid request"})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"[ERROR] Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /":        "This info",
            "GET /health":  "Health check",
            "GET /classes": "List disease classes",
            "POST /predict":"Classify disease from image",
            "GET /test":    "Interactive testing interface",
            "GET /docs":    "Swagger UI",
            "GET /redoc":   "ReDoc",
        },
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES,
        "classes": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from an uploaded image."""
    start_time = datetime.now()

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="File is empty")

        try:
            tensor = process_image(image_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            with torch.no_grad():
                outputs = model(tensor)

                if torch.isnan(outputs).any():
                    raise ValueError("Model produced NaN output")

                probabilities = torch.softmax(outputs, dim=1)

            probs_np = probabilities.cpu().numpy()[0]

            if not np.isfinite(probs_np).all():
                raise ValueError("Invalid probability values")

            top_3_idx   = np.argsort(probs_np)[::-1][:3]
            top_3_probs = probs_np[top_3_idx]

            if not all(0 <= idx < len(CLASS_NAMES) for idx in top_3_idx):
                raise ValueError("Model returned invalid class indices")

            predictions = [
                {
                    "class_id":          int(idx),
                    "class_name":        CLASS_NAMES[idx],
                    "confidence":        float(prob),
                    "confidence_percent": f"{float(prob) * 100:.2f}%"
                }
                for idx, prob in zip(top_3_idx, top_3_probs)
            ]

            predicted_class = CLASS_NAMES[top_3_idx[0]]
            confidence      = float(top_3_probs[0])

            response = {
                "success":           True,
                "predicted_class":   predicted_class,
                "confidence":        confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "all_predictions":   predictions,
                "is_healthy":        "healthy" in predicted_class.lower(),
                "requires_treatment": (
                    "disease" in predicted_class.lower() or
                    "diseased" in predicted_class.lower()
                )
            }

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"[PREDICT] {file.filename} ({len(image_data)/1024:.1f} KB) → "
                        f"{predicted_class} ({confidence*100:.2f}%) in {elapsed:.2f}s")

            return response

        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Inference error")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Request error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.get("/classes")
async def get_classes():
    return {
        "num_classes": NUM_CLASSES,
        "classes":     CLASS_NAMES,
        "class_map":   CLASS_MAP
    }


@app.get("/test", response_class=HTMLResponse)
async def get_test():
    test_file = Path(__file__).parent / "test_interface.html"
    if test_file.exists():
        try:
            return test_file.read_text(encoding='utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to load test interface")
    return """
    <html>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>Testing Interface Not Found</h1>
            <p>test_interface.html is missing.</p>
            <p><a href="/docs">Go to API Documentation</a></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    import sys
    try:
        port    = int(os.getenv("PORT", "8000"))
        workers = int(os.getenv("WORKERS", "1"))
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=workers,
            access_log=True,
            limit_concurrency=10,
            limit_max_requests=1000,
            timeout_keep_alive=15,
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Fatal: {str(e)}")
        sys.exit(1)