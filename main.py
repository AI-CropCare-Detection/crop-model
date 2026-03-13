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

# Load configuration
MODEL_META_PATH = Path(__file__).parent / 'checkpoints' / 'model_meta.json'
checkpoint_path = Path(__file__).parent / 'checkpoints' / 'best_model.pt'
torchscript_path = Path(__file__).parent / 'checkpoints' / 'yolov7_plant_disease.torchscript.pt'

# Load class names and metadata from model_meta.json
try:
    with open(MODEL_META_PATH, 'r') as f:
        model_meta = json.load(f)
    CLASS_NAMES = model_meta.get('class_names', [])
    NUM_CLASSES = model_meta.get('num_classes', len(CLASS_NAMES))
    IMG_SIZE = model_meta.get('img_size', 128)
    # Create mappings: index -> name and name -> index
    CLASS_MAP = {i: class_name for i, class_name in enumerate(CLASS_NAMES)}
    CLASS_TO_IDX = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
except FileNotFoundError:
    print(f"Warning: {MODEL_META_PATH} not found. Using default classes.")
    CLASS_NAMES = ['healthy', 'diseased']  # Default classes
    NUM_CLASSES = len(CLASS_NAMES)
    IMG_SIZE = 128
    CLASS_MAP = {i: class_name for i, class_name in enumerate(CLASS_NAMES)}
    CLASS_TO_IDX = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}


# Load Model
def load_model():
    """Load the trained YOLOv7 model from checkpoint with multiple fallbacks."""
    
    # Try TorchScript version first (fully compiled, no architecture needed)
    if torchscript_path.exists():
        file_size = torchscript_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        # Check if file is valid (should be 100+ MB)
        if file_size < 1_000_000:  # Less than 1MB = corrupted/download failed
            logger.warning(f"[WARN] TorchScript file too small ({size_mb:.2f}MB) - download may have failed")
        else:
            try:
                logger.info(f"[LOAD] Loading TorchScript model from: {torchscript_path} ({size_mb:.1f}MB)")
                model = torch.jit.load(str(torchscript_path), map_location=DEVICE)
                model.eval()
                logger.info(f"[OK] TorchScript model loaded on {DEVICE}")
                return model
            except Exception as e:
                logger.warning(f"[WARN] TorchScript load failed: {type(e).__name__}: {str(e)[:150]}")
    else:
        logger.warning(f"[WARN] TorchScript model not found at {torchscript_path}")
    
    # Try checkpoint with improved loading (weights_only=False for compatibility)
    if checkpoint_path.exists():
        file_size = checkpoint_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        # Check if file is valid (should be 300+ MB)
        if file_size < 1_000_000:  # Less than 1MB = corrupted/download failed
            logger.error(f"[ERROR] Checkpoint file too small ({size_mb:.2f}MB) - download may have failed")
        else:
            try:
                logger.info(f"[LOAD] Loading checkpoint from: {checkpoint_path} ({size_mb:.1f}MB)")
                # Use weights_only=False to handle pickled objects in older checkpoints
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                
                # Extract model state from checkpoint structure
                if isinstance(checkpoint, dict):
                    if 'model_state' in checkpoint:
                        logger.info(f"[OK] Found 'model_state' key in checkpoint")
                        state_dict = checkpoint['model_state']
                        if 'epoch' in checkpoint:
                            logger.info(f"   Training Epoch: {checkpoint['epoch']}")
                        if 'best_f1' in checkpoint:
                            logger.info(f"   Best F1 Score: {checkpoint['best_f1']:.4f}")
                    elif 'model' in checkpoint:
                        logger.info(f"[OK] Found 'model' key in checkpoint")
                        state_dict = checkpoint['model']
                    else:
                        logger.info(f"[OK] Using checkpoint as state_dict directly")
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Create architecture and load weights
                logger.info(f"[LOAD] Creating YOLOv7Classifier with {NUM_CLASSES} classes...")
                model = YOLOv7Classifier(num_classes=NUM_CLASSES, dropout=0.0)
                
                # Load with strict=False for flexibility
                incompatible = model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    logger.info(f"[WARN] {len(incompatible.missing_keys)} missing keys (expected)")
                if incompatible.unexpected_keys:
                    logger.info(f"[WARN] {len(incompatible.unexpected_keys)} unexpected keys (architecture mismatch)")
                
                model.to(DEVICE)
                model.eval()
                logger.info(f"[OK] Checkpoint model loaded on {DEVICE}")
                return model
                
            except Exception as e:
                logger.error(f"[ERROR] Checkpoint load failed: {type(e).__name__}: {str(e)[:150]}")
    else:
        logger.warning(f"[WARN] Checkpoint not found at {checkpoint_path}")
    
    # All model loading attempts failed - this is a critical failure  
    error_msg = (
        f"CRITICAL: Could not load any model file!\n"
        f"TorchScript exists: {torchscript_path.exists()} (size: {torchscript_path.stat().st_size if torchscript_path.exists() else 0}B)\n"
        f"Checkpoint exists: {checkpoint_path.exists()} (size: {checkpoint_path.stat().st_size if checkpoint_path.exists() else 0}B)\n"
        f"\nTroubleshooting:\n"
        f"1. If files show <1MB, Google Drive download likely failed\n"
        f"2. Check that Google Drive files are shared as 'Anyone with the link can view'\n"
        f"3. Verify download file IDs in Dockerfile (gdown commands)\n"
        f"4. Check Railway build logs for gdown download errors\n"
        f"5. Rebuild Docker image to retry downloads"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


class YOLOv7Classifier(torch.nn.Module):
    """Simple YOLOv7-based classifier for plant disease detection."""
    
    def __init__(self, num_classes=2, img_size=128, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Simplified backbone
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


# Image Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

def process_image(image_data: bytes, img_size: int = 128) -> torch.Tensor:
    """Process uploaded image for inference with validation."""
    try:
        # Validate file size
        if len(image_data) > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Max size: 50MB")
        
        if len(image_data) == 0:
            raise ValueError("File is empty")
        
        # Open and validate image
        img = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Validate image dimensions (at least 16x16)
        if img.size[0] < 16 or img.size[1] < 16:
            raise ValueError(f"Image too small. Minimum: 16x16, Got: {img.size}")
        
        # Resize
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor with validation
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Validate array values
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            raise ValueError("Invalid image data detected")
        
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        # Validate tensor
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contains NaN or Inf values")
        
        return tensor.to(DEVICE)
    
    except Exception as e:
        logger.error(f"[ERROR] Image processing error: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")


# FastAPI App
app = FastAPI(
    title="Plant Disease Detection API",
    description="YOLOv7-based API for plant disease classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded at startup)
model = None
model_loaded = False

# Startup & Shutdown
@app.on_event("startup")
async def startup_event():
    """Handle startup - load model once on first startup. MANDATORY: application fails if model cannot load."""
    global model, model_loaded
    
    logger.info("[START] Starting Plant Disease Detection API...")
    logger.info(f"[INFO] Classes: {NUM_CLASSES}")
    logger.info(f"[INFO] Device: {DEVICE}")
    
    try:
        # Load model only once (prevents race conditions with multiple workers)
        if not model_loaded:
            # Load model - this MUST succeed or the application MUST fail
            model = load_model()
            model_loaded = True
            
            if model is None:
                raise RuntimeError("Model loading returned None")
            
            logger.info(f"[OK] Model loaded successfully on {DEVICE}")
            logger.info(f"[INFO] Model ready: {model is not None}")
            
            if torch.cuda.is_available():
                logger.info(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    except Exception as e:
        # CRITICAL: Model loading failed - application cannot continue
        error_msg = f"[CRITICAL] Model loading failed during startup: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Re-raise to prevent application startup
        raise RuntimeError(error_msg) from e


@app.on_event("shutdown")
async def shutdown_event():
    """Handle graceful shutdown."""
    try:
        logger.info("[STOP] Shutting down API...")
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[OK] Cleanup complete")
    except Exception as e:
        logger.error(f"[ERROR] Shutdown error: {str(e)}")


# Global Exception Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    logger.warning(f"[WARN] Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unhandled exceptions."""
    logger.error(f"[ERROR] Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "device": str(DEVICE),
            "model_loaded": model is not None,
            "num_classes": NUM_CLASSES,
            "classes": len(CLASS_NAMES),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[ERROR] Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with predictions, confidence, and disease info
    """
    start_time = datetime.now()
    
    try:
        if model is None:
            logger.warning("[WARN] Prediction attempted but model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Process image with error handling
        try:
            tensor = process_image(image_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Run inference with validation
        try:
            with torch.no_grad():
                outputs = model(tensor)
                
                # Validate outputs
                if torch.isnan(outputs).any():
                    logger.error("[ERROR] Model returned NaN values")
                    raise ValueError("Model produced invalid output")
                
                probabilities = torch.softmax(outputs, dim=1)
            
            probs_np = probabilities.cpu().numpy()[0]
            
            # Validate probabilities
            if not np.isfinite(probs_np).all():
                logger.error("[ERROR] Probabilities contain NaN/Inf")
                raise ValueError("Invalid probability values")
            
            # Ensure probabilities sum to ~1
            prob_sum = probs_np.sum()
            if prob_sum < 0.9 or prob_sum > 1.1:
                logger.warning(f"[WARN] Probabilities sum to {prob_sum:.4f} (expected ~1.0)")
            
            # Get top predictions
            top_3_idx = np.argsort(probs_np)[::-1][:3]
            top_3_probs = probs_np[top_3_idx]
            
            # Validate indices
            if not all(0 <= idx < len(CLASS_NAMES) for idx in top_3_idx):
                logger.error("[ERROR] Invalid class indices detected")
                raise ValueError("Model returned invalid class indices")
            
            predictions = [
                {
                    "class_id": int(idx),
                    "class_name": CLASS_NAMES[idx],
                    "confidence": float(prob),
                    "confidence_percent": f"{float(prob) * 100:.2f}%"
                }
                for idx, prob in zip(top_3_idx, top_3_probs)
            ]
            
            predicted_class_idx = top_3_idx[0]
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(top_3_probs[0])
            
            response = {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "all_predictions": predictions,
                "is_healthy": "healthy" in predicted_class.lower(),
                "requires_treatment": "disease" in predicted_class.lower() or "diseased" in predicted_class.lower()
            }
            
            # Log prediction results
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"[PREDICT] Image: {file.filename} ({len(image_data)/1024:.1f}KB)")
            logger.info(f"[PREDICT] Predicted: {predicted_class} (Confidence: {confidence * 100:.2f}%)")
            logger.info(f"[PREDICT] Top 3 predictions:")
            for i, pred in enumerate(predictions, 1):
                logger.info(f"   {i}. {pred['class_name']}: {pred['confidence_percent']}")
            logger.info(f"[PREDICT] Health Status: {'HEALTHY' if response['is_healthy'] else 'DISEASED'}")
            logger.info(f"[PREDICT] Processing time: {elapsed:.2f}s")
            logger.info("-" * 80)
            
            return response
        
        except ValueError as e:
            logger.error(f"[ERROR] Inference error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"[ERROR] Unexpected inference error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Inference error")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Request error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[WARN] Cleanup error: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get list of all disease classes."""
    try:
        return {
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "class_map": CLASS_MAP
        }
    except Exception as e:
        logger.error(f"[ERROR] Classes endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve classes")


@app.get("/")
async def root():
    """API information endpoint."""
    try:
        return {
            "name": "Plant Disease Detection API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "GET /": "This info",
                "GET /health": "Health check",
                "GET /classes": "List disease classes",
                "POST /predict": "Classify disease from image",
                "GET /test": "Interactive testing interface",
                "GET /docs": "API documentation (Swagger UI)",
                "GET /redoc": "API documentation (ReDoc)"
            },
            "device": str(DEVICE),
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[ERROR] Root endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/test", response_class=HTMLResponse)
async def get_test():
    """Serve interactive testing interface."""
    test_file = Path(__file__).parent / "test_interface.html"
    if test_file.exists():
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[ERROR] Test interface error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load test interface")
    else:
        logger.warning("[WARN] test_interface.html not found")
        return """
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>Testing Interface Not Found</h1>
                <p>The test_interface.html file is missing.</p>
                <p><a href="/docs">Go to API Documentation</a></p>
            </body>
        </html>
        """


if __name__ == "__main__":
    import sys
    
    try:
        port = int(os.getenv("PORT", "8000"))
        workers = int(os.getenv("WORKERS", "1"))
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        logger.info("[START] Starting Plant Disease Detection API")
        logger.info(f"   Port: {port}")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Log Level: {log_level}")
        logger.info(f"   Device: {DEVICE}")
        logger.info(f"   Model Loaded: {model is not None}")
        
        # Run with production settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True,
            use_colors=True,
            limit_concurrency=10,  # Max concurrent requests
            limit_max_requests=1000,  # Restart workers after 1000 requests for memory safety
            timeout_keep_alive=15,
            timeout_notify=30,
        )
    except KeyboardInterrupt:
        logger.info("[STOP] Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

