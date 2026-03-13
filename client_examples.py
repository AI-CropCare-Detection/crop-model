"""
Example client code for interacting with the Plant Disease Detection API.
Shows different ways to use the API.
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any
import time


class PlantDiseaseClient:
    """Client for Plant Disease Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_classes(self) -> Dict[str, Any]:
        """Get list of disease classes."""
        response = self.session.get(f"{self.base_url}/classes")
        response.raise_for_status()
        return response.json()
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction results with confidence scores
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/predict",
                files=files
            )
        
        response.raise_for_status()
        return response.json()
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict disease from image bytes.
        
        Args:
            image_bytes: Image file as bytes
            
        Returns:
            Prediction results with confidence scores
        """
        files = {'file': ('image.jpg', image_bytes)}
        response = self.session.post(
            f"{self.base_url}/predict",
            files=files
        )
        
        response.raise_for_status()
        return response.json()


def example_basic_usage():
    """Basic example of using the API."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60 + "\n")
    
    client = PlantDiseaseClient("http://localhost:8000")
    
    try:
        # Check health
        health = client.health_check()
        print(f"✅ API Health: {health['status']}")
        print(f"   Device: {health['device']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        
        # Get available classes
        classes = client.get_classes()
        print(f"\n📊 Available disease classes ({classes['num_classes']}):")
        for cls in classes['classes']:
            print(f"   - {cls}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_single_image_prediction(image_path: str):
    """Predict disease from a single image."""
    print("="*60)
    print("EXAMPLE 2: Single Image Prediction")
    print("="*60 + "\n")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    client = PlantDiseaseClient("http://localhost:8000")
    
    try:
        print(f"🖼️  Analyzing: {image_path}")
        result = client.predict(image_path)
        
        print(f"\n✅ Prediction Complete")
        print(f"   Disease: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence_percent']}")
        print(f"   Healthy: {result['is_healthy']}")
        print(f"   Requires Treatment: {result['requires_treatment']}")
        
        print(f"\n📈 Top 3 Predictions:")
        for i, pred in enumerate(result['all_predictions'], 1):
            print(f"   {i}. {pred['class_name']}: {pred['confidence_percent']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_prediction(image_directory: str):
    """Predict diseases for multiple images."""
    print("="*60)
    print("EXAMPLE 3: Batch Prediction")
    print("="*60 + "\n")
    
    image_dir = Path(image_directory)
    if not image_dir.exists():
        print(f"❌ Directory not found: {image_directory}")
        return
    
    client = PlantDiseaseClient("http://localhost:8000")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in image_dir.glob('**/*') 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in {image_directory}")
        return
    
    print(f"🖼️  Found {len(images)} images. Processing...\n")
    
    results = []
    for i, img_path in enumerate(images, 1):
        try:
            print(f"[{i}/{len(images)}] {img_path.name}... ", end='', flush=True)
            result = client.predict(str(img_path))
            results.append({
                'file': img_path.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence_percent'],
                'is_healthy': result['is_healthy']
            })
            print(f"✅ {result['predicted_class']}")
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'file': img_path.name,
                'error': str(e)
            })
        
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Batch Summary")
    print("="*60)
    
    healthy_count = sum(1 for r in results if r.get('is_healthy'))
    diseased_count = len(results) - healthy_count
    
    print(f"Total Images: {len(results)}")
    print(f"Healthy: {healthy_count}")
    print(f"Diseased: {diseased_count}")
    
    return results


def example_using_requests_directly():
    """Show how to use requests library directly."""
    print("="*60)
    print("EXAMPLE 4: Direct requests Library Usage")
    print("="*60 + "\n")
    
    print("""
# Simple example
import requests

# Upload image and get prediction
with open('disease_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence_percent']}")

# Get classes
response = requests.get('http://localhost:8000/classes')
classes = response.json()
print(f"Available classes: {classes['classes']}")
""")


def example_production_usage():
    """Example of production-ready code."""
    print("="*60)
    print("EXAMPLE 5: Production-Ready Code")
    print("="*60 + "\n")
    
    print("""
import requests
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseasePredictionService:
    def __init__(self, api_url: str, retries: int = 3):
        self.api_url = api_url
        self.retries = retries
    
    def predict_disease(self, image_path: str) -> Optional[Dict[str, Any]]:
        \"\"\"Predict disease with error handling and retries.\"\"\"
        for attempt in range(self.retries):
            try:
                with open(image_path, 'rb') as f:
                    response = requests.post(
                        f"{self.api_url}/predict",
                        files={'file': f},
                        timeout=30
                    )
                
                response.raise_for_status()
                result = response.json()
                
                if result.get('success'):
                    logger.info(f"Prediction: {result['predicted_class']}")
                    return result
                else:
                    logger.error(f"API returned error: {result}")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt+1}/{self.retries})")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == self.retries - 1:
                    raise
        
        return None

# Usage
service = DiseasePredictionService("https://your-api.railway.app")
result = service.predict_disease("plant_image.jpg")

if result and result.get('requires_treatment'):
    print(f"Treatment recommended for {result['predicted_class']}")
""")


def main():
    import sys
    
    print("\\n" + "🌾"*30)
    print("  Plant Disease Detection API - Client Examples")
    print("🌾"*30 + "\n")
    
    # Run examples
    example_basic_usage()
    print("\n")
    
    # Uncomment to test with actual image
    # example_single_image_prediction("path/to/image.jpg")
    
    # example_batch_prediction("path/to/image/directory")
    # print("\n")
    
    example_using_requests_directly()
    print("\n")
    
    example_production_usage()
    
    print("\n" + "="*60)
    print("📚 See README_DEPLOYMENT.md for more information")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
