#!/usr/bin/env python
"""
Setup helper script for local development and deployment preparation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

def check_directories():
    """Verify required directories exist."""
    required_dirs = [
        'checkpoints',
        'processed_data',
        'results',
        'runs'
    ]
    
    print("📁 Checking directories...")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ⚠️  {dir_name}/ missing")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"     Created {dir_name}/")


def check_model_files():
    """Verify model and configuration files."""
    files_to_check = {
        'checkpoints/best_model.pt': 'Model checkpoint',
        'processed_data/class_map.json': 'Class mapping'
    }
    
    print("\n🤖 Checking model files...")
    for file_path, description in files_to_check.items():
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {description}: {file_path} NOT FOUND")
            print(f"     This file is required for the API to work")


def check_python_environment():
    """Check Python version and basic packages."""
    print(f"\n🐍 Python Environment")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")
    
    print(f"\n📦 Checking key packages...")
    packages = {
        'fastapi': 'FastAPI framework',
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  [OK] {description} ({package})")
        except ImportError:
            print(f"  [FAIL] {description} ({package}) - NOT INSTALLED")


def check_docker():
    """Check if Docker is installed."""
    print(f"\n[CHECK] Docker Status")
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  [OK] {result.stdout.strip()}")
        else:
            print(f"  [FAIL] Docker found but error: {result.stderr}")
    except FileNotFoundError:
        print(f"  [INFO] Docker not found (optional for local development)")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    print(f"\n⚙️  Environment Configuration")
    if env_file.exists():
        print(f"  ✅ .env already exists")
    elif env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print(f"  ✅ Created .env from .env.example")
    else:
        print(f"  ℹ️  No .env configuration needed")


def print_next_steps():
    """Print deployment instructions."""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    
    print("""
1. LOCAL DEVELOPMENT:
   python main.py
   # API will be at http://localhost:8000
   
2. TEST THE API:
   python test_api.py --image /path/to/test/image.jpg
   
3. DOCKER LOCAL TESTING:
   docker-compose up -d
   # View logs: docker-compose logs -f api
   
4. RAILWAY DEPLOYMENT:
   a) Option A - GitHub Integration (Easiest):
      - Push code to GitHub
      - Go to https://railway.app
      - Create new project
      - Connect to your GitHub repo
      - Railway auto-deploys!
      
   b) Option B - Using Railway CLI:
      - Install: npm i -g @railway/cli
      - railway login
      - railway init
      - railway up
      
5. VERIFY DEPLOYMENT:
   curl https://your-railway-url.railway.app/health
   
6. MONITOR:
   - Railway Dashboard: https://railway.app/dashboard
   - Logs: railway logs
   - Metrics: Check in dashboard

DOCUMENTATION:
   - Full guide: README_DEPLOYMENT.md
   - API docs: https://your-api-url/docs
""")


def main():
    """Run all checks and setup."""
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  Plant Disease Detection API - Setup Helper".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    check_directories()
    check_model_files()
    check_python_environment()
    check_docker()
    create_env_file()
    print_next_steps()
    
    print("\n" + "="*60)
    print("✅ Setup check complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
