#!/bin/bash
# CI/CD Setup Script for Railway Deployment
# This script helps configure GitHub Actions for automatic Railway deployment

set -e

echo "[INFO] Railway CI/CD Setup Script"
echo "[INFO] =============================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "${BLUE}[STEP 1]${NC} Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Git is not installed"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Git found"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Not in a git repository"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Git repository detected"

# Step 2: Verify workflow files exist
echo ""
echo -e "${BLUE}[STEP 2]${NC} Verifying workflow files..."

WORKFLOWS_DIR=".github/workflows"
if [ ! -d "$WORKFLOWS_DIR" ]; then
    mkdir -p "$WORKFLOWS_DIR"
    echo -e "${GREEN}[OK]${NC} Created $WORKFLOWS_DIR directory"
else
    echo -e "${GREEN}[OK]${NC} Workflows directory exists"
fi

if [ -f "$WORKFLOWS_DIR/railway-deploy.yml" ]; then
    echo -e "${GREEN}[OK]${NC} railway-deploy.yml exists"
else
    echo -e "${RED}[WARN]${NC} railway-deploy.yml not found"
fi

if [ -f "$WORKFLOWS_DIR/validate-pr.yml" ]; then
    echo -e "${GREEN}[OK]${NC} validate-pr.yml exists"
else
    echo -e "${RED}[WARN]${NC} validate-pr.yml not found"
fi

# Step 3: Verify project structure
echo ""
echo -e "${BLUE}[STEP 3]${NC} Verifying project structure..."

REQUIRED_FILES=("main.py" "requirements.txt" "Dockerfile" "railway.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}[OK]${NC} $file exists"
    else
        echo -e "${RED}[ERROR]${NC} $file not found"
        exit 1
    fi
done

if [ -d "checkpoints" ]; then
    echo -e "${GREEN}[OK]${NC} checkpoints/ directory exists"
    
    if [ -f "checkpoints/model_meta.json" ]; then
        echo -e "${GREEN}[OK]${NC} model_meta.json found"
    else
        echo -e "${YELLOW}[WARN]${NC} model_meta.json not found"
    fi
    
    if [ -f "checkpoints/yolov7_plant_disease.torchscript.pt" ]; then
        echo -e "${GREEN}[OK]${NC} TorchScript model found"
    else
        echo -e "${YELLOW}[WARN]${NC} TorchScript model not found"
    fi
else
    echo -e "${RED}[ERROR]${NC} checkpoints/ directory not found"
    exit 1
fi

# Step 4: Summary and next steps
echo ""
echo -e "${GREEN}=============================="
echo "[SUCCESS] Setup verification complete!"
echo "==============================${NC}"
echo ""
echo "Next Steps:"
echo "1. Go to GitHub Repository Settings"
echo "2. Navigate to: Secrets and variables → Actions"
echo "3. Add new secret: RAILWAY_TOKEN"
echo "   - Get token from: https://railway.app → Account Settings → Tokens"
echo "4. Commit and push your changes:"
echo "   git add ."
echo "   git commit -m 'ci: add Railway CI/CD pipeline'"
echo "   git push origin main"
echo "5. Go to Actions tab and watch the deployment"
echo ""
echo "Useful Commands:"
echo "  • View workflow runs: Check GitHub Actions tab"
echo "  • Manual deploy: railway up --detach"
echo "  • Check logs: railway logs"
echo "  • Test locally: python test_api.py"
echo ""
echo "Documentation:"
echo "  • See DEPLOYMENT_CI_CD.md for detailed guide"
echo "  • See railway.json for deployment config"
echo "  • See Dockerfile for container config"
echo ""
