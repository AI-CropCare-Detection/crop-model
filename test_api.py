#!/usr/bin/env python
"""
Test script for Plant Disease Detection API.
Tests all endpoints and example predictions.
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 10
TEST_FOLDER = Path(__file__).parent / "test"


def get_test_image() -> Optional[Path]:
    """Find first test image in test folder."""
    if TEST_FOLDER.exists():
        for ext in ['*.JPG', '*.jpg', '*.PNG', '*.png']:
            images = list(TEST_FOLDER.glob(ext))
            if images:
                return images[0]
    return None


class APITester:
    """Test harness for the API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.passed = 0
        self.failed = 0
    
    def print_header(self, text: str):
        print(f"\n{'='*60}")
        print(f"  {text}")
        print('='*60)
    
    def print_result(self, test_name: str, result: bool, details: str = ""):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} | {test_name}")
        if details:
            print(f"     -> {details}")
        
        if result:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_health_check(self) -> bool:
        """Test /health endpoint."""
        self.print_header("TEST 1: Health Check")
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            data = resp.json()
            
            success = resp.status_code == 200
            self.print_result(
                "Health check endpoint",
                success,
                f"Status: {data.get('status', 'unknown')}"
            )
            
            if success:
                print(f"     -> Device: {data.get('device')}")
                print(f"     -> Model Loaded: {data.get('model_loaded')}")
                print(f"     -> Classes: {data.get('num_classes')}")
            
            return success
        except Exception as e:
            self.print_result("Health check endpoint", False, str(e))
            return False
    
    def test_api_info(self) -> bool:
        """Test / endpoint."""
        self.print_header("TEST 2: API Information")
        try:
            resp = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            data = resp.json()
            
            success = resp.status_code == 200 and "endpoints" in data
            self.print_result(
                "API info endpoint",
                success,
                f"Version: {data.get('version')}"
            )
            
            if success:
                print(f"     -> Name: {data.get('name')}")
                print(f"     -> Endpoints available: {len(data.get('endpoints', {}))}")
            
            return success
        except Exception as e:
            self.print_result("API info endpoint", False, str(e))
            return False
    
    def test_get_classes(self) -> bool:
        """Test /classes endpoint."""
        self.print_header("TEST 3: Get Available Classes")
        try:
            resp = self.session.get(f"{self.base_url}/classes", timeout=TIMEOUT)
            data = resp.json()
            
            success = resp.status_code == 200 and "classes" in data
            self.print_result(
                "Get classes endpoint",
                success,
                f"Classes found: {data.get('num_classes')}"
            )
            
            if success:
                classes = data.get('classes', [])
                for cls in classes:
                    print(f"     -> {cls}")
            
            return success
        except Exception as e:
            self.print_result("Get classes endpoint", False, str(e))
            return False
    
    def test_predict_with_sample(self, image_path: Optional[str] = None) -> bool:
        """Test /predict endpoint with a sample image."""
        self.print_header("TEST 4: Image Prediction")
        
        # Auto-detect test image if not provided
        if not image_path:
            test_image = get_test_image()
            if test_image:
                image_path = str(test_image)
                print(f"Auto-detected test image: {test_image.name}")
            else:
                print("[WARN] No test image found in test/ folder.")
                print("   Usage: python test_api.py --image /path/to/image.jpg")
                return None
        
        image_file = Path(image_path)
        if not image_file.exists():
            self.print_result("Prediction test", False, f"Image not found: {image_path}")
            return False
        
        try:
            print(f"Testing with image: {image_file.name}")
            with open(image_file, 'rb') as f:
                # Explicitly set content type for the file
                files = {'file': (image_file.name, f, 'image/jpeg')}
                resp = self.session.post(
                    f"{self.base_url}/predict",
                    files=files,
                    timeout=TIMEOUT
                )
            
            data = resp.json()
            success = resp.status_code == 200 and data.get('success')
            
            self.print_result(
                "Prediction endpoint",
                success,
                f"Predicted: {data.get('predicted_class')}"
            )
            
            if success:
                print(f"     -> Confidence: {data.get('confidence_percent')}")
                print(f"     -> Is Healthy: {data.get('is_healthy')}")
                print(f"     -> Requires Treatment: {data.get('requires_treatment')}")
                
                predictions = data.get('all_predictions', [])
                print(f"     -> Top 3 Predictions:")
                for i, pred in enumerate(predictions[:3], 1):
                    print(f"        {i}. {pred['class_name']}: {pred['confidence_percent']}")
            
            return success
        except Exception as e:
            self.print_result("Prediction endpoint", False, str(e))
            return False
    
    def test_connection(self) -> bool:
        """Test basic connectivity."""
        self.print_header("CONNECTIVITY TEST")
        try:
            resp = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            success = resp.status_code == 200
            self.print_result(
                f"Connect to {self.base_url}",
                success
            )
            return success
        except requests.exceptions.ConnectionError:
            self.print_result(
                f"Connect to {self.base_url}",
                False,
                "Connection refused. Ensure API is running."
            )
            return False
        except Exception as e:
            self.print_result(f"Connect to {self.base_url}", False, str(e))
            return False
    
    def run_all(self, image_path: Optional[str] = None):
        """Run all tests."""
        print("\n" + "[START] " * 15)
        print("        PLANT DISEASE DETECTION API - TEST SUITE")
        print("[START] " * 15)
        
        # Check connectivity first
        if not self.test_connection():
            print("\n[FAIL] Cannot connect to API. Aborting tests.")
            print(f"   Make sure the API is running on {self.base_url}")
            sys.exit(1)  # Auto-detects from test/ if not provided
        
        # Run tests
        self.test_health_check()
        self.test_api_info()
        self.test_get_classes()
        self.test_predict_with_sample(image_path)
        
        # Summary
        self.print_header("TEST SUMMARY")
        total = self.passed + self.failed
        print(f"Passed: {self.passed}/{total}")
        print(f"Failed: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\n[OK] All tests passed!")
        else:
            print(f"\n[WARN] {self.failed} test(s) failed")
        
        return self.failed == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Plant Disease Detection API"
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"Base URL of API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--image",
        help="Path to test image for prediction"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = tester.run_all(args.image)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
